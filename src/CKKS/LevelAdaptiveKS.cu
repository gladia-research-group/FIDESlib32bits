#include "CKKS/LevelAdaptiveKS.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/KskSeedExpand.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "NTTconfig.cuh"

namespace FIDESlib::CKKS {

namespace {

constexpr int LAR_MAXM = 32;    // source limbs of one base conversion (a digit, or P')
constexpr int LAR_TG = 16;      // conversion targets per thread
constexpr int LAR_MAXJOBS = 4;  // digits per class (and the dot's per-digit accumulation bound)

struct LarConvJob {
    void** src;           // m source limbs, coefficient form
    void** dst;           // cnt target limbs, coefficient form
    const int* srcIds;    // m prime ids
    const int* dstIds;    // cnt prime ids
    const uint32_t* pre;  // m: (B/b_i)^{-1} mod b_i
    const uint32_t* mat;  // m x ld: (B/b_i) mod t_k
    int m, cnt, ld;
};
struct LarConvJobs {
    LarConvJob j[LAR_MAXJOBS];
    int chunk;  // products accumulated between lazy reductions (u64 headroom for the widest primes)
};

// Fast basis conversion B -> T (approximate, the standard hybrid-KS ModUp/ModDown conversion), one job per
// blockIdx.z. Thread = one coefficient, LAR_TG targets of which 4 accumulate together (ILP); the m pre-scaled sources
// stay in registers, the block's m x LAR_TG matrix tile sits in shared memory (all threads read the same entry), and
// the lazy reduction points are compile-time (every CH products; CH from the u64 headroom of the widest primes).
template <int CH>
__global__ void __launch_bounds__(128, 10) lar_bconv_(const __grid_constant__ LarConvJobs jobs) {
    const LarConvJob& J = jobs.j[blockIdx.z];
    const int k0 = blockIdx.y * LAR_TG;
    if (k0 >= J.cnt)
        return;
    extern __shared__ uint32_t lar_smem[];
    uint32_t* sy = lar_smem;                          // m x 128: this block's pre-scaled sources, thread-major
    uint32_t* smat = lar_smem + J.m * 128;            // m x LAR_TG matrix tile
    int* sdst = (int*)(smat + J.m * LAR_TG);          // LAR_TG target prime ids
    const int kn = min(LAR_TG, J.cnt - k0);
    for (int x = threadIdx.x; x < J.m * LAR_TG; x += blockDim.x) {
        const int i = x / LAR_TG, k = x - i * LAR_TG;
        smat[x] = k < kn ? J.mat[i * J.ld + k0 + k] : 0u;
    }
    if (threadIdx.x < LAR_TG)
        sdst[threadIdx.x] = threadIdx.x < kn ? J.dstIds[k0 + threadIdx.x] : J.dstIds[k0];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < J.m; ++i)
        sy[i * 128 + threadIdx.x] = modreduce_lazy((uint64_t)((const uint32_t*)J.src[i])[idx] * J.pre[i], J.srcIds[i]);
    __syncthreads();
    for (int k = 0; k < kn; k += 4) {
        const int t0 = sdst[k], t1 = sdst[k + 1], t2 = sdst[k + 2], t3 = sdst[k + 3];
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i0 = 0; i0 < J.m; i0 += CH) {
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                const int i = i0 + c;
                if (i < J.m) {
                    const uint64_t yi = sy[i * 128 + threadIdx.x];
                    const uint4 r = *(const uint4*)(smat + i * LAR_TG + k);
                    a0 += yi * r.x;
                    a1 += yi * r.y;
                    a2 += yi * r.z;
                    a3 += yi * r.w;
                }
            }
            a0 = modreduce_lazy(a0, t0);
            a1 = modreduce_lazy(a1, t1);
            a2 = modreduce_lazy(a2, t2);
            a3 = modreduce_lazy(a3, t3);
        }
        ((uint32_t*)J.dst[k0 + k])[idx] = (uint32_t)a0;
        if (k + 1 < kn)
            ((uint32_t*)J.dst[k0 + k + 1])[idx] = (uint32_t)a1;
        if (k + 2 < kn)
            ((uint32_t*)J.dst[k0 + k + 2])[idx] = (uint32_t)a2;
        if (k + 3 < kn)
            ((uint32_t*)J.dst[k0 + k + 3])[idx] = (uint32_t)a3;
    }
}

// the widest instantiated reduction interval the primes allow
void launchBconv(const LarConvJobs& jobs, dim3 grid, cudaStream_t q) {
    int m = 0;
    for (unsigned z = 0; z < grid.z; ++z)
        m = std::max(m, jobs.j[z].m);
    const size_t smem = (size_t)m * (128 + LAR_TG) * sizeof(uint32_t) + LAR_TG * sizeof(int);
    if (jobs.chunk >= 8)
        lar_bconv_<8><<<grid, 128, smem, q>>>(jobs);
    else if (jobs.chunk >= 4)
        lar_bconv_<4><<<grid, 128, smem, q>>>(jobs);
    else if (jobs.chunk >= 2)
        lar_bconv_<2><<<grid, 128, smem, q>>>(jobs);
    else
        lar_bconv_<1><<<grid, 128, smem, q>>>(jobs);
}

struct LarDotArgs {
    void** inLimbs;                       // in (NTT, Q): the digit's own limbs, overwritten with <d, a>
    void** auxLimbs;                      // aux (Q): <d, b>
    uint32_t* accP;                       // 2*np limbs: <d, a> then <d, b> on P'
    const uint32_t* modup[LAR_MAXJOBS];   // digit j: [P' | Q_nHi \ D_j] limbs, contiguous
    const uint32_t* keyA[LAR_MAXJOBS];    // digit j: limb of prime id k at k*N
    const uint32_t* keyB[LAR_MAXJOBS];
    const int* pIds;                      // np P' prime ids
    int start[LAR_MAXJOBS + 1];
    int nd, np, N;
    uint32_t seed[8];                     // seeded class: keyA unused, `a` regenerated
};

// Inner product with the class key over Q_n u P' (blockIdx.y < np: P' limb, else Q limb y - np).
__global__ void lar_dot_(const __grid_constant__ LarDotArgs A) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = blockIdx.y;
    const bool isP = e < A.np;
    const int t = isP ? -1 : e - A.np;
    const int pid = isP ? A.pIds[e] : t;
    const size_t off = (size_t)pid * A.N + idx;
    uint64_t accA = 0, accB = 0;  // <= LAR_MAXJOBS products of < 2^62: no wrap
    for (int j = 0; j < A.nd; ++j) {
        uint32_t v;
        if (isP) {
            v = A.modup[j][(size_t)e * A.N + idx];
        } else if (t >= A.start[j] && t < A.start[j + 1]) {
            v = ((const uint32_t*)A.inLimbs[t])[idx];
        } else {
            const int pos = A.np + (t < A.start[j] ? t : t - (A.start[j + 1] - A.start[j]));
            v = A.modup[j][(size_t)pos * A.N + idx];
        }
        accA += (uint64_t)v * A.keyA[j][off];
        accB += (uint64_t)v * A.keyB[j][off];
    }
    const uint32_t ra = modreduce_lazy(accA, pid);
    const uint32_t rb = modreduce_lazy(accB, pid);
    if (isP) {
        A.accP[(size_t)e * A.N + idx] = ra;
        A.accP[(size_t)(A.np + e) * A.N + idx] = rb;
    } else {
        ((uint32_t*)A.inLimbs[t])[idx] = ra;
        ((uint32_t*)A.auxLimbs[t])[idx] = rb;
    }
}

// the rare rejected keystream word: reference escalation, kept out of line so the hot loop stays in registers
__device__ __noinline__ uint32_t lar_expand_slow(const uint32_t* seed, uint32_t digit, uint32_t p, uint32_t slot,
                                                 uint32_t n16) {
    uint32_t key[8];
    for (int i = 0; i < 8; ++i)
        key[i] = seed[i];
    return kskexpand::expand_coeff(key, digit, p, slot, n16);
}

// lar_dot_ with a seed-expanded `a`: thread = 8 consecutive coefficients of one limb, the half of one ChaCha12 block of
// digit j's keystream for that prime (KskSeedExpand.cuh addressing: block = slot >> 4, word = slot & 15); the two
// threads of a block each compute it (2x keystream compute, half the accumulator registers). A rejected word (probability
// < 2^-4 per coefficient at 28-bit primes) takes the reference escalation path, bit-identical to the host.
__global__ void __launch_bounds__(128) lar_dot_seeded_(const __grid_constant__ LarDotArgs A) {
    const uint32_t th = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blk = th >> 1, half = th & 1u;
    const int idx0 = (int)th * 8;
    const int e = blockIdx.y;
    const bool isP = e < A.np;
    const int t = isP ? -1 : e - A.np;
    const int pid = isP ? A.pIds[e] : t;
    const uint32_t p = (uint32_t)C_.primes[pid];
    const uint32_t m_p = (0xFFFFFFFFu / p) * p;
    const uint32_t n16 = (uint32_t)A.N >> 4;
    const size_t off = (size_t)pid * A.N + idx0;
    uint64_t accA[8], accB[8];
#pragma unroll
    for (int w = 0; w < 8; ++w)
        accA[w] = accB[w] = 0;
    for (int j = 0; j < A.nd; ++j) {
        const uint32_t* vs;
        if (isP)
            vs = A.modup[j] + (size_t)e * A.N + idx0;
        else if (t >= A.start[j] && t < A.start[j + 1])
            vs = (const uint32_t*)A.inLimbs[t] + idx0;
        else
            vs = A.modup[j] + (size_t)(A.np + (t < A.start[j] ? t : t - (A.start[j + 1] - A.start[j]))) * A.N + idx0;
        uint32_t ks[16];
        kskexpand::chacha_block(A.seed, blk, (uint32_t)j, p, ks);
        uint32_t a[8];
        uint32_t rej = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t k = half ? ks[8 + w] : ks[w];
            rej |= (uint32_t)(k >= m_p) << w;
            a[w] = modreduce_lazy((uint64_t)k, pid);
        }
        if (rej) {
#pragma unroll
            for (int w = 0; w < 8; ++w)
                if (rej >> w & 1u)
                    a[w] = lar_expand_slow(A.seed, (uint32_t)j, p, (uint32_t)(idx0 + w), n16);
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint4 v = ((const uint4*)vs)[q];
            const uint4 b = ((const uint4*)(A.keyB[j] + off))[q];
            accA[4 * q + 0] += (uint64_t)v.x * a[4 * q + 0];
            accA[4 * q + 1] += (uint64_t)v.y * a[4 * q + 1];
            accA[4 * q + 2] += (uint64_t)v.z * a[4 * q + 2];
            accA[4 * q + 3] += (uint64_t)v.w * a[4 * q + 3];
            accB[4 * q + 0] += (uint64_t)v.x * b.x;
            accB[4 * q + 1] += (uint64_t)v.y * b.y;
            accB[4 * q + 2] += (uint64_t)v.z * b.z;
            accB[4 * q + 3] += (uint64_t)v.w * b.w;
        }
    }
    uint32_t* da = isP ? A.accP + (size_t)e * A.N + idx0 : (uint32_t*)A.inLimbs[t] + idx0;
    uint32_t* db = isP ? A.accP + (size_t)(A.np + e) * A.N + idx0 : (uint32_t*)A.auxLimbs[t] + idx0;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        ((uint4*)da)[q] = make_uint4(modreduce_lazy(accA[4 * q], pid), modreduce_lazy(accA[4 * q + 1], pid),
                                     modreduce_lazy(accA[4 * q + 2], pid), modreduce_lazy(accA[4 * q + 3], pid));
        ((uint4*)db)[q] = make_uint4(modreduce_lazy(accB[4 * q], pid), modreduce_lazy(accB[4 * q + 1], pid),
                                     modreduce_lazy(accB[4 * q + 2], pid), modreduce_lazy(accB[4 * q + 3], pid));
    }
}

// Fused-dot path (stored `a`): the accumulators start as the digits' own-limb terms (t in D_j uses d_t itself) and
// zero on P'; each digit's ModUp NTT then accumulates x * key into them in its second pass (NTT_KSK_DOT_ACC).
struct LarOwnArgs {
    void** inLimbs;
    void** auxLimbs;
    uint32_t* accP;
    const uint32_t* keyA[LAR_MAXJOBS];
    const uint32_t* keyB[LAR_MAXJOBS];
    int start[LAR_MAXJOBS + 1];
    int nd, np, N;
};
__global__ void lar_owndot_(const __grid_constant__ LarOwnArgs A) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int e = blockIdx.y;
    if (e < A.np) {
        A.accP[(size_t)e * A.N + idx] = 0;
        A.accP[(size_t)(A.np + e) * A.N + idx] = 0;
        return;
    }
    const int t = e - A.np;
    int j = 0;
    while (j + 1 < A.nd && t >= A.start[j + 1])
        ++j;
    const size_t off = (size_t)t * A.N + idx;
    uint32_t* in = (uint32_t*)A.inLimbs[t];
    const uint32_t v = in[idx];
    in[idx] = modreduce_lazy((uint64_t)v * A.keyA[j][off], t);
    ((uint32_t*)A.auxLimbs[t])[idx] = modreduce_lazy((uint64_t)v * A.keyB[j][off], t);
}

// the per-call output tables of the fused NTT: digit j's target y -> its A / B accumulator limb
struct LarTabArgs {
    void** inLimbs;
    void** auxLimbs;
    uint32_t* accP;
    void** resA[LAR_MAXJOBS];
    void** resB[LAR_MAXJOBS];
    const int* dstIds[LAR_MAXJOBS];
    int cap[LAR_MAXJOBS];
    int nd, np, nHi, N;
};
__global__ void lar_tables_(const __grid_constant__ LarTabArgs A) {
    const int j = blockIdx.x;
    for (int y = threadIdx.x; y < A.cap[j]; y += blockDim.x) {
        const int id = A.dstIds[j][y];
        if (y < A.np) {  // the P' part leads every target list, in pIds order
            A.resA[j][y] = A.accP + (size_t)y * A.N;
            A.resB[j][y] = A.accP + (size_t)(A.np + y) * A.N;
        } else {
            A.resA[j][y] = A.inLimbs[id];
            A.resB[j][y] = A.auxLimbs[id];
        }
    }
}

// ---------------------------------------------------------------- host side

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t p) {
    return (uint64_t)((unsigned __int128)a * b % p);
}
uint64_t powmod(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1;
    a %= p;
    while (e) {
        if (e & 1)
            r = mulmod(r, a, p);
        a = mulmod(a, a, p);
        e >>= 1;
    }
    return r;
}
uint64_t invmod(uint64_t a, uint64_t p) {
    return powmod(a, p - 2, p);
}

template <typename T>
T* devUpload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T));
    if (!v.empty())
        cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

void** limbTable(uint32_t* base, int count, size_t N) {
    std::vector<void*> h(count);
    for (int k = 0; k < count; ++k)
        h[k] = base + (size_t)k * N;
    return devUpload(h);
}

// Products of a sum-of-products accumulator that fit u64 after a reduction leaves < 2^32.
int chunkFor(uint64_t maxA, uint64_t maxB) {
    const unsigned __int128 prod = (unsigned __int128)(maxA - 1) * (maxB - 1);
    const unsigned __int128 room = ((unsigned __int128)1 << 64) - ((unsigned __int128)1 << 32);
    return (int)std::max<unsigned __int128>(1, std::min<unsigned __int128>(room / prod, 64));
}

struct LarClassDev {
    LarGeometry geo;
    int beta = 0, np = 0;
    int rowT[LAR_MAXJOBS] = {}, rowPP = 0;
    int cap[LAR_MAXJOBS] = {};
    uint32_t* keyA[LAR_MAXJOBS] = {};
    uint32_t* keyB[LAR_MAXJOBS] = {};
    uint32_t* modup[LAR_MAXJOBS] = {};
    void** modupPtrs[LAR_MAXJOBS] = {};
    int* dstIds[LAR_MAXJOBS] = {};
    std::map<std::pair<int, int>, std::pair<uint32_t*, uint32_t*>> upTab;  // (digit, live primes) -> (pre, mat)
    uint32_t *preP = nullptr, *matP = nullptr;
    int top = 0;  // FMD row holding P'^{-1} mod q_t: the ModDown tail rides NTT_MODDOWNR (rows >= nQ are unused)
    int* pIds = nullptr;
    uint32_t* accP = nullptr;
    void** accPPtrs = nullptr;
    int chunkUp = 1, chunkDown = 1;
    bool seeded = false;
    uint32_t seed[8] = {};
    void** ptA[LAR_MAXJOBS] = {};   // fused path: digit j's target y -> key limbs (static)
    void** ptB[LAR_MAXJOBS] = {};
    void** resA[LAR_MAXJOBS] = {};  // fused path: digit j's target y -> accumulator limbs (rebuilt per call)
    void** resB[LAR_MAXJOBS] = {};
};

struct LarState {
    const KeySwitchingKey* evalKey = nullptr;
    int nQ = 0, K = 0, N = 0;
    std::vector<LarClassDev> classes;
    uint32_t *coef = nullptr, *scratch = nullptr, *conv = nullptr;
    void **coefPtrs = nullptr, **scratchPtrs = nullptr, **convPtrs = nullptr;
    int* qIds = nullptr;  // 0..nQ-1
};

std::map<const ContextData*, std::unique_ptr<LarState>> g_lar;
bool g_active = true;

// FIDESLIB_LAR_STATS=1: eval-key relins per live-limb count, taken (lar) or not, printed at exit
struct LarStats {
    long hit[MAXP + 1] = {}, miss[MAXP + 1] = {};
    ~LarStats() {
        long h = 0, m = 0;
        for (int n = 0; n <= MAXP; ++n) {
            h += hit[n];
            m += miss[n];
        }
        if (h + m == 0)
            return;
        std::fprintf(stderr, "[lar] eval-key relins: %ld level-adaptive, %ld production; per live limbs n:lar/prod", h, m);
        for (int n = 0; n <= MAXP; ++n)
            if (hit[n] + miss[n])
                std::fprintf(stderr, " %d:%ld/%ld", n, hit[n], miss[n]);
        std::fprintf(stderr, "\n");
    }
};
LarStats* larStats() {
    static const bool on = [] {
        const char* e = std::getenv("FIDESLIB_LAR_STATS");
        return e && std::atoi(e) > 0;
    }();
    static LarStats st;  // printed by its destructor at exit
    return on ? &st : nullptr;
}

int envInt(const char* name, int dflt) {
    const char* e = std::getenv(name);
    return (e && *e) ? std::atoi(e) : dflt;
}

}  // namespace

bool larSeeded() {
    static const bool on = envInt("FIDESLIB_LAR_SEEDED", 0) > 0;
    return on;
}

bool larEnabled() {
    static const bool on = envInt("FIDESLIB_LAR", 0) > 0;
    return on;
}

void larSetActive(bool on) {
    g_active = on;
}

std::vector<LarGeometry> larPlanClasses(Context& ccp) {
    std::vector<LarGeometry> out;
    if (!larEnabled())
        return out;
    ContextData& cc = *ccp;
    const Constants& C = cc.precom.constants.at(0);
    if (cc.GPUid.size() != 1 || C.type != 0) {
        std::cerr << "[lar] disabled: single-GPU all-u32 chains only\n";
        return out;
    }
    const int nQ = C.L, K = cc.K;
    for (int i = 0; i < nQ; ++i)
        if (cc.meta.at(0).at(i).id != i)
            throw std::runtime_error("[lar] Q limb ids are not level-ordered");
    std::vector<std::pair<int, int>> spec;
    {
        const char* e = std::getenv("FIDESLIB_LAR_CLASSES");
        std::stringstream ss(e && *e ? e : "42:2");
        std::string item;
        while (std::getline(ss, item, ',')) {
            const auto c = item.find(':');
            if (c == std::string::npos)
                throw std::runtime_error("[lar] FIDESLIB_LAR_CLASSES wants nHi:beta[,nHi:beta]");
            spec.emplace_back(std::stoi(item.substr(0, c)), std::stoi(item.substr(c + 1)));
        }
    }
    std::sort(spec.begin(), spec.end());
    int nLo = envInt("FIDESLIB_LAR_NLO", 28), rows = 0;
    for (auto [nHi, beta] : spec) {
        LarGeometry g;
        g.nHi = nHi;
        g.nLo = std::max(nLo, 1);
        const int np = (nQ - nHi) + K;
        const int alpha = (nHi + beta - 1) / beta;
        for (int j = 0; j <= beta; ++j)
            g.start.push_back(std::min(j * alpha, nHi));
        for (int id = nHi; id < nQ + K; ++id)
            g.pPrimeIds.push_back(id);
        double digitBits = 0, pBits = 0;
        for (int j = 0; j < beta; ++j) {
            double b = 0;
            for (int i = g.start[j]; i < g.start[j + 1]; ++i)
                b += std::log2((double)C.primes[i]);
            digitBits = std::max(digitBits, b);
        }
        for (int id : g.pPrimeIds)
            pBits += std::log2((double)C.primes[id]);
        const bool ok = nHi > g.nLo && nHi < nQ && beta >= 1 && beta <= LAR_MAXJOBS && alpha <= LAR_MAXM &&
                        np <= LAR_MAXM && 2 * np <= MAXP && np + nHi <= MAXP && digitBits <= pBits + 1.0 &&
                        rows + beta + 1 <= MAXD - 1;
        std::cerr << "[lar] class nHi=" << nHi << " beta=" << beta << " alpha=" << alpha << " |P'|=" << np
                  << " digit " << digitBits << " bits vs P' " << pBits << " bits, serves n in [" << g.nLo << ", "
                  << nHi << "]" << (ok ? "" : "  REJECTED") << "\n";
        if (!ok)
            continue;
        rows += beta + 1;
        nLo = nHi + 1;
        out.push_back(std::move(g));
    }
    return out;
}

void InstallLarKeys(Context& ccp, std::vector<LarHostKey>&& keys, const KeyHash& evalKeyTag) {
    ContextData& cc = *ccp;
    SetCurrentContext(ccp);
    cudaSetDevice(cc.GPUid.at(0));
    Constants& C = cc.precom.constants.at(0);
    auto st = std::make_unique<LarState>();
    st->nQ = C.L;
    st->K = cc.K;
    st->N = cc.N;
    const int nQ = st->nQ, nTot = st->nQ + st->K;
    const size_t N = cc.N, limbBytes = N * sizeof(uint32_t);
    st->evalKey = &cc.GetEvalKey(evalKeyTag);

    cudaMalloc(&st->coef, nQ * limbBytes);
    cudaMalloc(&st->scratch, LAR_MAXJOBS * MAXP * limbBytes);  // one NTT scratch per concurrent lane
    cudaMalloc(&st->conv, 2 * nQ * limbBytes);
    st->coefPtrs = limbTable(st->coef, nQ, N);
    st->scratchPtrs = limbTable(st->scratch, LAR_MAXJOBS * MAXP, N);
    st->convPtrs = limbTable(st->conv, 2 * nQ, N);
    {
        std::vector<int> q(nQ);
        for (int i = 0; i < nQ; ++i)
            q[i] = i;
        st->qIds = devUpload(q);
    }
    uint64_t maxQ = 0, maxAll = 0;
    for (int id = 0; id < nTot; ++id) {
        maxAll = std::max<uint64_t>(maxAll, C.primes[id]);
        if (id < nQ)
            maxQ = std::max<uint64_t>(maxQ, C.primes[id]);
    }

    int row = 1;  // PARTITION rows 1..MAXD-1 are unused on a single GPU
    std::vector<uint32_t> tmp(N);
    for (auto& hk : keys) {
        LarClassDev d;
        d.geo = hk.geo;
        const LarGeometry& g = d.geo;
        d.beta = (int)g.start.size() - 1;
        d.np = (int)g.pPrimeIds.size();
        const int np = d.np, nHi = g.nHi;
        d.pIds = devUpload(g.pPrimeIds);
        cudaMalloc(&d.accP, 2 * np * limbBytes);
        d.accPPtrs = limbTable(d.accP, 2 * np, N);
        for (int j = 0; j < d.beta; ++j) {
            const int s0 = g.start[j], s1 = g.start[j + 1];
            std::vector<int> ids(g.pPrimeIds);
            for (int t = 0; t < nHi; ++t)
                if (t < s0 || t >= s1)
                    ids.push_back(t);
            d.cap[j] = (int)ids.size();
            d.rowT[j] = row++;
            for (int k = 0; k < d.cap[j]; ++k)
                C.primeid_partition[d.rowT[j]][k] = ids[k];
            d.dstIds[j] = devUpload(ids);
            cudaMalloc(&d.modup[j], d.cap[j] * limbBytes);
            d.modupPtrs[j] = limbTable(d.modup[j], d.cap[j], N);
            // ModUp tables for every live prefix m of this digit
            for (int m = 1; m <= s1 - s0; ++m) {
                std::vector<uint32_t> pre(m), mat((size_t)m * d.cap[j]);
                for (int i = 0; i < m; ++i) {
                    const uint64_t qi = C.primes[s0 + i];
                    uint64_t hat = 1;
                    for (int l = 0; l < m; ++l)
                        if (l != i)
                            hat = mulmod(hat, C.primes[s0 + l] % qi, qi);
                    pre[i] = (uint32_t)invmod(hat, qi);
                    for (int k = 0; k < d.cap[j]; ++k) {
                        const uint64_t pk = C.primes[ids[k]];
                        uint64_t h = 1;
                        for (int l = 0; l < m; ++l)
                            if (l != i)
                                h = mulmod(h, C.primes[s0 + l] % pk, pk);
                        mat[(size_t)i * d.cap[j] + k] = (uint32_t)h;
                    }
                }
                d.upTab[{j, m}] = {devUpload(pre), devUpload(mat)};
            }
            // keys: every prime id 0..nTot-1, EVAL form, u32 (a seeded class stores no `a`)
            d.seeded = hk.seed.size() == 8;
            if (d.seeded)
                std::copy(hk.seed.begin(), hk.seed.end(), d.seed);
            else
                cudaMalloc(&d.keyA[j], nTot * limbBytes);
            cudaMalloc(&d.keyB[j], nTot * limbBytes);
            for (int id = 0; id < nTot; ++id) {
                for (int half = d.seeded ? 1 : 0; half < 2; ++half) {
                    const auto& src = (half == 0 ? hk.a : hk.b).at(j).at(id);
                    if (src.size() != N)
                        throw std::runtime_error("[lar] key limb size mismatch");
                    for (size_t c = 0; c < N; ++c)
                        tmp[c] = (uint32_t)src[c];
                    cudaMemcpy((half == 0 ? d.keyA[j] : d.keyB[j]) + (size_t)id * N, tmp.data(), limbBytes,
                               cudaMemcpyHostToDevice);
                }
            }
        }
        if (!d.seeded)
            for (int j = 0; j < d.beta; ++j) {
                std::vector<int> ids(d.cap[j]);
                cudaMemcpy(ids.data(), d.dstIds[j], d.cap[j] * sizeof(int), cudaMemcpyDeviceToHost);
                std::vector<void*> a(d.cap[j]), b(d.cap[j]);
                for (int y = 0; y < d.cap[j]; ++y) {
                    a[y] = d.keyA[j] + (size_t)ids[y] * N;
                    b[y] = d.keyB[j] + (size_t)ids[y] * N;
                }
                d.ptA[j] = devUpload(a);
                d.ptB[j] = devUpload(b);
                cudaMalloc(&d.resA[j], d.cap[j] * sizeof(void*));
                cudaMalloc(&d.resB[j], d.cap[j] * sizeof(void*));
            }
        d.rowPP = row++;
        for (int k = 0; k < 2 * np; ++k)
            C.primeid_partition[d.rowPP][k] = g.pPrimeIds[k % np];
        // ModDown P' -> Q_nHi
        {
            std::vector<uint32_t> pre(np), mat((size_t)np * nHi);
            std::vector<uint64_t> binv(MAXP, 0), binvSh(MAXP, 0);
            for (int i = 0; i < np; ++i) {
                const uint64_t pi = C.primes[g.pPrimeIds[i]];
                uint64_t hat = 1;
                for (int l = 0; l < np; ++l)
                    if (l != i)
                        hat = mulmod(hat, C.primes[g.pPrimeIds[l]] % pi, pi);
                pre[i] = (uint32_t)invmod(hat, pi);
                for (int t = 0; t < nHi; ++t) {
                    const uint64_t qt = C.primes[t];
                    uint64_t h = 1;
                    for (int l = 0; l < np; ++l)
                        if (l != i)
                            h = mulmod(h, C.primes[g.pPrimeIds[l]] % qt, qt);
                    mat[(size_t)i * nHi + t] = (uint32_t)h;
                }
            }
            for (int t = 0; t < nHi; ++t) {
                const uint64_t qt = C.primes[t];
                uint64_t P = 1;
                for (int l = 0; l < np; ++l)
                    P = mulmod(P, C.primes[g.pPrimeIds[l]] % qt, qt);
                binv[t] = invmod(P, qt);
                binvSh[t] = shoup_precomp(binv[t], t, C);
            }
            d.preP = devUpload(pre);
            d.matP = devUpload(mat);
            d.top = MAXP - 1 - (int)st->classes.size();
            if (d.top < nQ)  // FMD rows are indexed by a composite top q id < nQ
                throw std::runtime_error("[lar] no free FMD row for the ModDown tail");
            char* gp = (char*)cc.precom.globals->globals[0];
            cudaMemcpy(gp + offsetof(Global::Globals, FMD_Binv) + (size_t)d.top * MAXP * sizeof(uint64_t), binv.data(),
                       MAXP * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(gp + offsetof(Global::Globals, FMD_Binv_shoup) + (size_t)d.top * MAXP * sizeof(uint64_t),
                       binvSh.data(), MAXP * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }
        d.chunkUp = chunkFor(maxQ, maxAll);
        d.chunkDown = chunkFor(maxAll, maxQ);
        st->classes.push_back(std::move(d));
    }
    // the new prime-id rows: device copy now, host copy for every later SetCurrentContext
    cudaMemcpyToSymbol(constants, &C, sizeof(Constants), 0, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CudaCheckErrorMod;
    size_t bytes = 0;
    for (auto& d : st->classes)
        bytes += (size_t)(d.seeded ? 1 : 2) * d.beta * nTot * limbBytes;
    std::cerr << "[lar] installed " << st->classes.size() << " class(es), keys " << (bytes >> 20) << " MiB\n";
    g_lar[&cc] = std::move(st);
}

bool larTryKeySwitch(RNSPoly& in, RNSPoly& aux, const KeySwitchingKey& ksk) {
    if (!g_active || g_lar.empty())
        return false;
    LimbPartition& P = in.GPU.at(0);
    ContextData& cc = P.cc;
    const auto it = g_lar.find(&cc);
    if (it == g_lar.end())
        return false;
    LarState& st = *it->second;
    if (&ksk != st.evalKey)
        return false;
    const int level = in.getLevel();
    const int n = level + 1;
    LarClassDev* cl = nullptr;
    for (auto& d : st.classes)
        if (n >= d.geo.nLo && n <= d.geo.nHi) {
            cl = &d;
            break;
        }
    if (LarStats* stats = larStats())
        (cl ? stats->hit : stats->miss)[std::min(n, MAXP)]++;
    if (!cl)
        return false;

    aux.setLevel(level);
    LimbPartition& X = aux.GPU.at(0);
    cudaSetDevice(P.device);
    Stream& s = P.s;
    s.wait(X.s);
    cudaStream_t cs = s.ptr();
    Global::Globals* G = cc.precom.globals->globals[0];
    const LarGeometry& g = cl->geo;
    const int np = cl->np, N = st.N;
    int nd = 0;
    while (nd < cl->beta && g.start[nd] < n)
        ++nd;

    constexpr ALGO algo = ALGO_SHOUP;
    const int M = FIDES_NTT_M32;  // all-u32 chain (larPlanClasses)
    const dim3 bdF{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
    const dim3 bdS{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
    const int bytesF = 4 * bdF.x * (2 * M + 1 + 1);
    const int bytesS = 4 * bdS.x * (2 * M + 1 + 1);
    const uint32_t gxF = N / (bdF.x * M * 2), gxS = N / (bdS.x * M * 2);
    // Lanes: lane 0 is the input's stream, lane k > 0 the context's digit stream k (relins are host-serialized, so the
    // digit streams are idle here; fork / join through their events). Each lane owns an NTT scratch.
    auto lane = [&](int k) -> Stream& { return k == 0 ? s : cc.digitStream.at(k).at(0); };
    auto scr = [&](int k) { return st.scratchPtrs + k * MAXP; };
    auto ntt = [&](cudaStream_t q, void** tmp, void** buf, int row, int cnt) {
        NTT_<false, algo, NTT_NONE><<<dim3{gxF, (uint32_t)cnt}, bdF, bytesF, q>>>(G, buf, row, tmp);
        NTT_<true, algo, NTT_NONE><<<dim3{gxS, (uint32_t)cnt}, bdS, bytesS, q>>>(G, tmp, row, buf);
    };
    auto intt = [&](cudaStream_t q, void** tmp, void** src, int row, int cnt, void** dst) {
        INTT_<false, algo, INTT_NONE><<<dim3{gxF, (uint32_t)cnt}, bdF, bytesF, q>>>(G, src, row, tmp);
        INTT_<true, algo, INTT_NONE><<<dim3{gxS, (uint32_t)cnt}, bdS, bytesS, q>>>(G, tmp, row, dst);
    };

    // 1. d -> coefficient form
    intt(cs, scr(0), P.limbptr.data, PARTITION(0, 0), n, st.coefPtrs);
    if (!cl->seeded) {
        // 2. accumulators <- own-limb terms (zero on P'); this call's output tables for the fused NTT
        LarOwnArgs oa{};
        oa.inLimbs = P.limbptr.data;
        oa.auxLimbs = X.limbptr.data;
        oa.accP = cl->accP;
        LarTabArgs ta{};
        ta.inLimbs = P.limbptr.data;
        ta.auxLimbs = X.limbptr.data;
        ta.accP = cl->accP;
        for (int j = 0; j < nd; ++j) {
            oa.keyA[j] = cl->keyA[j];
            oa.keyB[j] = cl->keyB[j];
            ta.resA[j] = cl->resA[j];
            ta.resB[j] = cl->resB[j];
            ta.dstIds[j] = cl->dstIds[j];
            ta.cap[j] = cl->cap[j];
        }
        for (int j = 0; j <= cl->beta; ++j)
            oa.start[j] = g.start[j];
        oa.nd = ta.nd = nd;
        oa.np = ta.np = np;
        oa.N = ta.N = N;
        ta.nHi = g.nHi;
        lar_tables_<<<nd, 64, 0, cs>>>(ta);
        lar_owndot_<<<dim3{(uint32_t)N / 256, (uint32_t)(np + n)}, 256, 0, cs>>>(oa);
        // 3. ModUp, one lane per live digit: digit j to [P' | Q_n \ D_j], NTT whose second pass accumulates x * key
        // into the accumulators (the second passes chain digit after digit: they share the P' targets)
        for (int j = 1; j < nd; ++j)  // fork every lane before any digit's work lands on lane 0
            lane(j).wait(s);
        for (int j = 0; j < nd; ++j) {
            Stream& q = lane(j);
            const int s0 = g.start[j], s1 = g.start[j + 1];
            const int m = std::min(s1, n) - s0;
            const int cnt = np + std::min(n, s0) + std::max(0, n - s1);
            const auto& tab = cl->upTab.at({j, m});
            LarConvJobs up{};
            up.chunk = cl->chunkUp;
            up.j[0] = LarConvJob{st.coefPtrs + s0, cl->modupPtrs[j], st.qIds + s0, cl->dstIds[j], tab.first,
                                 tab.second, m, cnt, cl->cap[j]};
            launchBconv(up, dim3{(uint32_t)N / 128, (uint32_t)((cnt + LAR_TG - 1) / LAR_TG), 1}, q.ptr());
            NTT_<false, algo, NTT_NONE><<<dim3{gxF, (uint32_t)cnt}, bdF, bytesF, q.ptr()>>>(
                G, cl->modupPtrs[j], PARTITION(cl->rowT[j], 0), scr(j));
            if (j)
                q.wait(lane(j - 1));
            NTT_<true, algo, NTT_KSK_DOT_ACC><<<dim3{gxS, (uint32_t)cnt}, bdS, bytesS, q.ptr()>>>(
                G, scr(j), PARTITION(cl->rowT[j], 0), cl->resB[j], cl->ptA[j], 0, cl->resA[j], cl->ptB[j]);
        }
        if (nd > 1)
            s.wait(lane(nd - 1));
    } else {
    // 2. ModUp, one lane per live digit: digit j to [P' | Q_n \ D_j], then NTT
    for (int j = 1; j < nd; ++j)
        lane(j).wait(s);
    for (int j = 0; j < nd; ++j) {
        Stream& q = lane(j);
        const int s0 = g.start[j], s1 = g.start[j + 1];
        const int m = std::min(s1, n) - s0;
        const int cnt = np + std::min(n, s0) + std::max(0, n - s1);
        const auto& tab = cl->upTab.at({j, m});
        LarConvJobs up{};
        up.chunk = cl->chunkUp;
        up.j[0] = LarConvJob{st.coefPtrs + s0, cl->modupPtrs[j], st.qIds + s0, cl->dstIds[j], tab.first, tab.second, m,
                             cnt, cl->cap[j]};
        launchBconv(up, dim3{(uint32_t)N / 128, (uint32_t)((cnt + LAR_TG - 1) / LAR_TG), 1}, q.ptr());
        ntt(q.ptr(), scr(j), cl->modupPtrs[j], PARTITION(cl->rowT[j], 0), cnt);
    }
    for (int j = 1; j < nd; ++j)
        s.wait(lane(j));
    // 3. inner product with the class key over Q_n u P'
    LarDotArgs da{};
    da.inLimbs = P.limbptr.data;
    da.auxLimbs = X.limbptr.data;
    da.accP = cl->accP;
    for (int j = 0; j < nd; ++j) {
        da.modup[j] = cl->modup[j];
        da.keyA[j] = cl->keyA[j];
        da.keyB[j] = cl->keyB[j];
    }
    for (int j = 0; j <= cl->beta; ++j)
        da.start[j] = g.start[j];
    da.pIds = cl->pIds;
    da.nd = nd;
    da.np = np;
    da.N = N;
    if (cl->seeded) {
        std::copy(cl->seed, cl->seed + 8, da.seed);
        lar_dot_seeded_<<<dim3{(uint32_t)N / (8 * 128), (uint32_t)(np + n)}, 128, 0, cs>>>(da);
    } else {
        lar_dot_<<<dim3{(uint32_t)N / 256, (uint32_t)(np + n)}, 256, 0, cs>>>(da);
    }
    }
    // 4. ModDown by P', the <d, a> half on lane 0 and the <d, b> half on lane 1: INTT of the P' part, conversion to
    // Q_n, NTT with the second pass fused into x <- (x - conv) * P'^{-1}
    Stream& q1 = lane(1);
    q1.wait(s);
    for (int h = 0; h < 2; ++h) {
        cudaStream_t q = h ? q1.ptr() : cs;
        void** acc = cl->accPPtrs + h * np;
        void** conv = st.convPtrs + h * st.nQ;
        intt(q, scr(h), acc, PARTITION(cl->rowPP, 0), np, acc);
        LarConvJobs down{};
        down.chunk = cl->chunkDown;
        down.j[0] = LarConvJob{acc, conv, cl->pIds, st.qIds, cl->preP, cl->matP, np, n, g.nHi};
        launchBconv(down, dim3{(uint32_t)N / 128, (uint32_t)((n + LAR_TG - 1) / LAR_TG), 1}, q);
        NTT_<false, algo, NTT_NONE><<<dim3{gxF, (uint32_t)n}, bdF, bytesF, q>>>(G, conv, PARTITION(0, 0), scr(h));
        NTT_<true, algo, NTT_MODDOWNR><<<dim3{gxS, (uint32_t)n}, bdS, bytesS, q>>>(
            G, scr(h), PARTITION(0, 0), h == 0 ? P.limbptr.data : X.limbptr.data, nullptr, cl->top);
    }
    s.wait(q1);
    CudaCheckErrorModNoSync;
    X.s.wait(s);
    in.SetModUp(false);
    aux.SetModUp(false);
    return true;
}

}  // namespace FIDESlib::CKKS
