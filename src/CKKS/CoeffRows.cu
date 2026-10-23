// Plaintext-weight inner sums from coefficient rows (see CoeffRows.cuh).
#include "CKKS/CoeffRows.cuh"

#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "ModMult.cuh"
#include "NTTcluster.cuh"
#include "NTTcore.cuh"

#include <cooperative_groups.h>

namespace FIDESlib::CKKS {

namespace {

constexpr int kMaxChunk = 32;

struct CtTables {
    void** c0[kMaxChunk];   // each ciphertext's device limb-pointer table (LimbPartition::limbptr)
    void** c1[kMaxChunk];
};

// m = round(row[k] * scale) reduced into limbs 0..L-1 of plaintext r: scratch[(r * L + l) * N + k]
__global__ void coeffRowsScatter_(const float* __restrict__ rows, size_t rowStride, double scale, int N, int L,
                                  uint32_t* __restrict__ scratch, unsigned long long* overflow) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (k >= N) return;
    const double x = (double)rows[(size_t)r * rowStride + k] * scale;
    long long m;
    if (fabs(x) < 9.2e18) {
        m = llrint(x);
    } else {
        m = x < 0 ? -(1ll << 62) : (1ll << 62);
        atomicAdd(overflow, 1ull);
    }
    const unsigned long long a = (unsigned long long)(m < 0 ? -m : m);
    uint32_t* out = scratch + (size_t)r * L * N + k;
    for (int l = 0; l < L; ++l) {
        const int primeid = C_.primeid_flattened[l];
        uint32_t v = modreduce_lazy(a, primeid);
        if (m < 0 && v != 0) v = (uint32_t)C_.primes[primeid] - v;
        out[(size_t)l * N] = v;
    }
}

// out_c{0,1}[l][k] (+)= sum_r ct_r.c{0,1}[l][k] * pt_r[l][k] mod p_l; `accumulate` keeps out's previous value
__global__ void coeffRowsMac_(CtTables t, int n, const uint32_t* __restrict__ scratch, int N, int L, void** out0,
                              void** out1, bool accumulate) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (k >= N) return;
    const int primeid = C_.primeid_flattened[l];
    uint32_t* o0 = (uint32_t*)out0[l];
    uint32_t* o1 = (uint32_t*)out1[l];
    unsigned long long a0 = accumulate ? o0[k] : 0, a1 = accumulate ? o1[k] : 0;
    // products are < 2^62 (primes < 2^31): reduce after every second one
    for (int r = 0; r < n; ++r) {
        const unsigned long long w = scratch[((size_t)r * L + l) * N + k];
        a0 += w * ((const uint32_t*)t.c0[r][l])[k];
        a1 += w * ((const uint32_t*)t.c1[r][l])[k];
        if (r & 1) {
            a0 = modreduce_lazy(a0, primeid);
            a1 = modreduce_lazy(a1, primeid);
        }
    }
    o0[k] = modreduce_lazy(a0, primeid);
    o1[k] = modreduce_lazy(a1, primeid);
}

// Fused path (FIDESLIB_COEFF_FUSED): one cluster per (limb, row group) rounds each row, transforms it in shared
// memory and accumulates the products in registers. Row groups write partial sums (coeffRowsReduce_).
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(ntc::CL, 1, 1) __launch_bounds__(ntc::GROUPS* ntc::BD, 2)
    coeffRowsFused_(const Global::Globals* Globals, const float* __restrict__ rows, size_t rowStride, int n, int groups,
                    double scale, int L, CtTables t, void** out0, void** out1, uint32_t* partial, bool accumulate,
                    unsigned long long* overflow) {
    using namespace ntc;
    __shared__ __align__(16) T region[REG_I2 * 2];
    __shared__ T psi[BD], psi_sh[BD];
    cg::cluster_group cluster = cg::this_cluster();
    const int c = blockIdx.x / CL;
    const int limb = c % L, group = c / L;
    const int rank = (int)cluster.block_rank();
    const int grp = threadIdx.x >> LOGBD;
    const int tid = threadIdx.x & (BD - 1);
    const int j = tid << 1;
    const int bx = rank * GROUPS + grp;
    T* gbuf = region + grp * TILE_T;
    const int primeid = C_.primeid_flattened[limb];
    const uint32_t p = (uint32_t)C_.primes[primeid];
    if (threadIdx.x < BD) {
        psi[tid] = ((T*)G_->psi[primeid])[tid];
        psi_sh[tid] = ((T*)G_->psi_shoup[primeid])[tid];
    }
    const int per = (n + groups - 1) / groups;
    const int r0 = group * per, r1 = min(n, r0 + per);
    uint32_t a0[2 * M], a1[2 * M];
    const bool direct = groups == 1;
#pragma unroll
    for (int i = 0; i < M; ++i) {
        const int q = TILE_I2 * bx + BD * i + tid;
        if (direct && accumulate) {
            const uint2 v0 = ((const uint2*)out0[limb])[q], v1 = ((const uint2*)out1[limb])[q];
            a0[2 * i] = v0.x; a0[2 * i + 1] = v0.y; a1[2 * i] = v1.x; a1[2 * i + 1] = v1.y;
        } else {
            a0[2 * i] = a0[2 * i + 1] = a1[2 * i] = a1[2 * i + 1] = 0;
        }
    }
    const int col_init = j & ~2;
    for (int r = r0; r < r1; ++r) {
        const float* row = rows + (size_t)r * rowStride;
        int4 temp[4];
        for (int i = 0; i < M / 2; ++i) {
            const int base = 4 * (transp_i2(col_init + i, bx, j) >> 1);
            const float4 f = *(const float4*)(row + base);
            const float fv[4] = {f.x, f.y, f.z, f.w};
            uint32_t rv[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const double x = (double)fv[e] * scale;
                long long m;
                if (fabs(x) < 9.2e18) {
                    m = llrint(x);
                } else {
                    m = x < 0 ? -(1ll << 62) : (1ll << 62);
                    atomicAdd(overflow, 1ull);
                }
                uint32_t v = modreduce_lazy((unsigned long long)(m < 0 ? -m : m), primeid);
                rv[e] = (m < 0 && v != 0) ? p - v : v;
            }
            ((T*)&temp[0])[i] = rv[0];
            ((T*)&temp[1])[i] = rv[1];
            ((T*)&temp[2])[i] = rv[2];
            ((T*)&temp[3])[i] = rv[3];
        }
        __syncthreads();
        store_transposed_regs(gbuf, temp, j);
        __syncthreads();
        fwd_negacyclic(gbuf, primeid, psi, psi_sh, Globals, tid, bx);
        ct_stages(gbuf, psi, psi_sh, primeid, tid);
        ntt_eot_store_inplace(gbuf, psi, psi_sh, primeid, Globals, tid, bx);
        cluster.sync();
        for (int i = 0; i < M / 2; ++i) {
            const int q2 = transp_i2(col_init + i, bx, j);
            const T* owner = cluster.map_shared_rank(region, q2 / REG_I2);
            const int4 v = ((const int4*)owner)[(q2 % REG_I2) >> 1];
            ((T*)&temp[0])[i] = v.x;
            ((T*)&temp[1])[i] = v.y;
            ((T*)&temp[2])[i] = v.z;
            ((T*)&temp[3])[i] = v.w;
        }
        cluster.sync();
        store_transposed_regs(gbuf, temp, j);
        __syncthreads();
        ct_stages(gbuf, psi, psi_sh, primeid, tid);
        __syncthreads();
        const uint2* c0 = (const uint2*)t.c0[r][limb];
        const uint2* c1 = (const uint2*)t.c1[r][limb];
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const int pos = swz_pos<T>(i, j);
            int2 o = ((int2*)GA(i))[pos >> 1];
            if (pos & 1) {
                const int t_ = o.x;
                o.x = o.y;
                o.y = t_;
            }
            const int q = TILE_I2 * bx + BD * i + tid;
            const uint2 x0 = c0[q], x1 = c1[q];
            a0[2 * i] = modreduce_lazy(a0[2 * i] + (unsigned long long)(uint32_t)o.x * x0.x, primeid);
            a0[2 * i + 1] = modreduce_lazy(a0[2 * i + 1] + (unsigned long long)(uint32_t)o.y * x0.y, primeid);
            a1[2 * i] = modreduce_lazy(a1[2 * i] + (unsigned long long)(uint32_t)o.x * x1.x, primeid);
            a1[2 * i + 1] = modreduce_lazy(a1[2 * i + 1] + (unsigned long long)(uint32_t)o.y * x1.y, primeid);
        }
    }
#pragma unroll
    for (int i = 0; i < M; ++i) {
        const int q = TILE_I2 * bx + BD * i + tid;
        if (direct) {
            ((uint2*)out0[limb])[q] = make_uint2(a0[2 * i], a0[2 * i + 1]);
            ((uint2*)out1[limb])[q] = make_uint2(a1[2 * i], a1[2 * i + 1]);
        } else {
            uint2* p0 = (uint2*)(partial + ((size_t)(group * L + limb) * 2) * (2 * TILE_I2 * TILES));
            uint2* p1 = p0 + TILE_I2 * TILES;
            p0[q] = make_uint2(a0[2 * i], a0[2 * i + 1]);
            p1[q] = make_uint2(a1[2 * i], a1[2 * i + 1]);
        }
    }
}

__global__ void coeffRowsReduce_(const uint32_t* __restrict__ partial, int groups, int L, int N, void** out0, void** out1,
                                 bool accumulate) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (k >= N) return;
    const int primeid = C_.primeid_flattened[l];
    unsigned long long s0 = accumulate ? ((uint32_t*)out0[l])[k] : 0, s1 = accumulate ? ((uint32_t*)out1[l])[k] : 0;
    for (int g = 0; g < groups; ++g) {
        const uint32_t* b = partial + ((size_t)(g * L + l) * 2) * N;
        s0 += b[k];
        s1 += b[N + k];
    }
    ((uint32_t*)out0[l])[k] = modreduce_lazy(s0, primeid);
    ((uint32_t*)out1[l])[k] = modreduce_lazy(s1, primeid);
}

bool fusedOn() {
    static const bool v = [] {
        const char* e = std::getenv("FIDESLIB_COEFF_FUSED");
        return !(e && e[0] == '0');
    }();
    return v;
}

// Two buffer sets used in turn by consecutive chunks (across calls too). Chunk c on set b = c % 2:
//   copy stream:  wait scattered[b]; rows -> stage[b] (DMA, pinned host rows only); record copied[b]
//   enc stream:   wait copied[b], macked[b]; scatter stage[b] (or the rows in place) -> res[b]; record scattered[b];
//                 NTT res[b]; record encoded[b]
//   out stream:   wait encoded[b]; MAC res[b] into the output; record macked[b]
// so the transfer of chunk c+1 and its encode overlap the MAC of chunk c, and every buffer is reused only after its
// last reader is done, whichever ciphertext streams the calls run on.
struct Set {
    float* stage = nullptr;                   // chunk x N rows (device copy of pinned host rows)
    uint32_t* res = nullptr;                  // chunk x L x N residues (room for Lcap limbs)
    std::map<int, void**> ptrs;               // per limb count: res's limb pointers, for the NTT
    cudaEvent_t copied = nullptr, scattered = nullptr, encoded = nullptr, macked = nullptr;
};

struct Scratch {
    int N = 0, Lcap = 0, chunk = 0;
    uint32_t* partial = nullptr;
    size_t partialWords = 0;
    cudaEvent_t partialDone = nullptr;
    Set set[2];
    int next = 0;
    cudaStream_t copy = nullptr, enc = nullptr;
    unsigned long long* overflow = nullptr;   // mapped pinned host counter
};

std::mutex g_mutex;
Scratch g_s;

int chunkSize() {
    static const int c = [] {
        const char* e = std::getenv("FIDESLIB_COEFF_CHUNK");
        const int v = e ? std::atoi(e) : (fusedOn() ? kMaxChunk : 16);
        return v < 1 ? 1 : v > kMaxChunk ? kMaxChunk : v;
    }();
    return c;
}

// buffers for `L` limbs (regrown, with a device sync, only when a call needs more room than any before)
void ensure(Scratch& s, int N, int L, int chunk) {
    if (!s.overflow) {
        cudaHostAlloc(&s.overflow, sizeof(unsigned long long), cudaHostAllocMapped);
        *s.overflow = 0;
        cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s.enc, cudaStreamNonBlocking);
        for (auto& b : s.set)
            for (cudaEvent_t* e : {&b.copied, &b.scattered, &b.encoded, &b.macked})
                cudaEventCreateWithFlags(e, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.partialDone, cudaEventDisableTiming);
    }
    if (s.N == N && s.chunk == chunk && s.Lcap >= L) return;
    cudaDeviceSynchronize();   // earlier calls' kernels may still use the old buffers
    for (auto& b : s.set) {
        if (b.stage) cudaFree(b.stage);
        if (b.res) cudaFree(b.res);
        for (auto& [l, p] : b.ptrs) cudaFree(p);
        b.ptrs.clear();
        cudaMalloc(&b.stage, sizeof(float) * (size_t)chunk * N);
        if (!fusedOn())
            cudaMalloc(&b.res, sizeof(uint32_t) * (size_t)chunk * L * N);
    }
    s.N = N;
    s.Lcap = L;
    s.chunk = chunk;
}

void** ptrsFor(Set& b, int N, int L, int chunk) {
    auto it = b.ptrs.find(L);
    if (it != b.ptrs.end()) return it->second;
    std::vector<void*> h((size_t)chunk * L);
    for (size_t i = 0; i < h.size(); ++i) h[i] = b.res + i * N;
    void** d = nullptr;
    cudaMalloc(&d, sizeof(void*) * h.size());
    cudaMemcpy(d, h.data(), sizeof(void*) * h.size(), cudaMemcpyHostToDevice);
    return b.ptrs.emplace(L, d).first->second;
}

void joinLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) P.s.wait(STREAM(P.limb[i]));
}
void forkLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) STREAM(P.limb[i]).wait(P.s);
}

}  // namespace

unsigned long long coeffRowsOverflows() {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_s.overflow ? *(volatile unsigned long long*)g_s.overflow : 0ull;
}

void multCoeffRowsSum(Ciphertext& out, const std::vector<const Ciphertext*>& cts, const float* rows, size_t rowStride,
                      double scale) {
    if (cts.empty())
        return;
    ContextData& C = out.cc;
    const int N = C.N;
    const int L = cts[0]->getLevel() + 1;   // limbs 0..level
    const int chunk = chunkSize();
    std::lock_guard<std::mutex> g(g_mutex);
    Scratch& s = g_s;
    ensure(s, N, L, chunk);

    LimbPartition& O0 = out.c0.GPU.at(0);
    LimbPartition& O1 = out.c1.GPU.at(0);
    joinLimbs(O0);
    joinLimbs(O1);
    O0.s.wait(O1.s);
    Stream& E = O0.s;
    // limb streams are shared across ciphertexts: wait on each distinct one once
    {
        std::vector<Stream*> seen;
        seen.reserve(64);
        auto waitOnce = [&](Stream& st) {
            for (Stream* x : seen)
                if (x == &st) return;
            seen.push_back(&st);
            E.wait(st);
        };
        for (const Ciphertext* c : cts) {
            for (const RNSPoly* poly : {&c->c0, &c->c1}) {
                auto& P = const_cast<LimbPartition&>(poly->GPU.at(0));
                waitOnce(P.s);
                const int nl = P.getLimbSize(*P.level);
                for (int i = 0; i < nl; ++i) waitOnce(STREAM(P.limb[i]));
            }
        }
    }
    const auto* G = O0.getGlobals();
    const unsigned long long* ovf = nullptr;
    cudaHostGetDevicePointer((void**)&ovf, s.overflow, 0);
    // pinned host rows go through the copy engine; device rows are read in place
    cudaPointerAttributes at{};
    const bool host = cudaPointerGetAttributes(&at, rows) == cudaSuccess && at.type == cudaMemoryTypeHost;
    cudaGetLastError();
    if (fusedOn() && N == 2 * ntc::TILE_I2 * ntc::TILES) {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        static const int waves = [] { const char* e = std::getenv("FIDESLIB_COEFF_WAVES"); return e ? std::max(1, std::atoi(e)) : 2; }();
        const int clusters = std::max(1, 2 * waves * sms / ntc::CL);
        for (size_t r0 = 0; r0 < cts.size(); r0 += chunk) {
            const int n = (int)std::min<size_t>(chunk, cts.size() - r0);
            Set& b = s.set[s.next];
            s.next ^= 1;
            const float* src = rows + r0 * rowStride;
            size_t stride = rowStride;
            if (host) {
                cudaStreamWaitEvent(s.copy, b.scattered, 0);
                cudaMemcpy2DAsync(b.stage, sizeof(float) * N, src, sizeof(float) * rowStride, sizeof(float) * N, n,
                                  cudaMemcpyHostToDevice, s.copy);
                cudaEventRecord(b.copied, s.copy);
                cudaStreamWaitEvent(E.ptr(), b.copied, 0);
                src = b.stage;
                stride = N;
            }
            const int groups = std::max(1, std::min(n, clusters / L));
            if (groups > 1) {
                const size_t words = (size_t)groups * L * 2 * N;
                if (s.partialWords < words) {
                    cudaDeviceSynchronize();
                    if (s.partial) cudaFree(s.partial);
                    cudaMalloc(&s.partial, sizeof(uint32_t) * words);
                    s.partialWords = words;
                }
                cudaStreamWaitEvent(E.ptr(), s.partialDone, 0);
            }
            CtTables t{};
            for (int r = 0; r < n; ++r) {
                t.c0[r] = cts[r0 + r]->c0.GPU.at(0).limbptr.data;
                t.c1[r] = cts[r0 + r]->c1.GPU.at(0).limbptr.data;
            }
            coeffRowsFused_<<<groups * L * ntc::CL, ntc::GROUPS * ntc::BD, 0, E.ptr()>>>(
                G, src, stride, n, groups, scale, L, t, O0.limbptr.data, O1.limbptr.data, s.partial, r0 > 0,
                const_cast<unsigned long long*>(ovf));
            if (groups > 1) {
                coeffRowsReduce_<<<dim3((N + 255) / 256, L), 256, 0, E.ptr()>>>(s.partial, groups, L, N, O0.limbptr.data,
                                                                               O1.limbptr.data, r0 > 0);
                cudaEventRecord(s.partialDone, E.ptr());
            }
            cudaEventRecord(b.scattered, E.ptr());
        }
    } else
    for (size_t r0 = 0; r0 < cts.size(); r0 += chunk) {
        const int n = (int)std::min<size_t>(chunk, cts.size() - r0);
        Set& b = s.set[s.next];
        s.next ^= 1;
        void** ptrs = ptrsFor(b, N, L, chunk);
        const float* src = rows + r0 * rowStride;
        size_t stride = rowStride;
        if (host) {
            cudaStreamWaitEvent(s.copy, b.scattered, 0);
            cudaMemcpy2DAsync(b.stage, sizeof(float) * N, src, sizeof(float) * rowStride, sizeof(float) * N, n,
                              cudaMemcpyHostToDevice, s.copy);
            cudaEventRecord(b.copied, s.copy);
            cudaStreamWaitEvent(s.enc, b.copied, 0);
            src = b.stage;
            stride = N;
        }
        cudaStreamWaitEvent(s.enc, b.macked, 0);
        coeffRowsScatter_<<<dim3((N + 255) / 256, n), 256, 0, s.enc>>>(src, stride, scale, N, L, b.res,
                                                                       const_cast<unsigned long long*>(ovf));
        cudaEventRecord(b.scattered, s.enc);
        launchNTTcluster(G, false, ptrs, 0, ptrs, n * L, s.enc, L);
        cudaEventRecord(b.encoded, s.enc);
        CtTables t{};
        for (int r = 0; r < n; ++r) {
            t.c0[r] = cts[r0 + r]->c0.GPU.at(0).limbptr.data;
            t.c1[r] = cts[r0 + r]->c1.GPU.at(0).limbptr.data;
        }
        cudaStreamWaitEvent(E.ptr(), b.encoded, 0);
        coeffRowsMac_<<<dim3((N + 255) / 256, L), 256, 0, E.ptr()>>>(t, n, b.res, N, L, O0.limbptr.data,
                                                                     O1.limbptr.data, r0 > 0);
        cudaEventRecord(b.macked, E.ptr());
    }
    // the inputs are read on E: they live until E is past this point
    for (const Ciphertext* c : cts) {
        const_cast<LimbPartition&>(c->c0.GPU.at(0)).s.wait(E);
        const_cast<LimbPartition&>(c->c1.GPU.at(0)).s.wait(E);
    }
    O1.s.wait(E);
    forkLimbs(O0);
    forkLimbs(O1);

    const Ciphertext& a = *cts[0];
    out.slots = std::max(a.slots, N / 2);
    out.keyID = a.keyID;
    out.NoiseLevel = a.NoiseLevel + 1;
    out.NoiseFactor = a.NoiseFactor * scale;
}

}  // namespace FIDESlib::CKKS
