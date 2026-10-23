#include "AddSub.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "NTTtc_core.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "NTTcore.cuh"

namespace FIDESlib {

static int g_tc_ntt = -1;
int tcNttFlag() {
    if (g_tc_ntt < 0) {
        const char* e = std::getenv("FIDESLIB_TC_NTT");
        g_tc_ntt = e ? std::atoi(e) : 0;
    }
    return g_tc_ntt;
}

namespace {
// One block = one 8-row tile; block bx holds the unit vectors e_{8*bx + i} in rows i and runs the butterfly
// core, so after it row i holds column (8*bx + i) of the core matrix: out planes [l][m][k].
template <bool INVERSE>
__global__ void __launch_bounds__(128) tc_probe_(const Global::Globals* Globals, const int primeid, uint8_t* out) {
    using T = uint32_t;
    __shared__ __align__(16) T tile[ntc::TILE_T];
    __shared__ T psi[ntc::BD], psi_sh[ntc::BD];
    const int tid = threadIdx.x;
    T* gbuf = tile;
    psi[tid] = ((T*)(INVERSE ? G_->inv_psi[primeid] : G_->psi[primeid]))[tid];
    psi_sh[tid] = ((T*)(INVERSE ? G_->inv_psi_shoup[primeid] : G_->psi_shoup[primeid]))[tid];
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + ntc::BD * q;
        const int i = idx >> 8, e = idx & 255;
        GAS(i, e) = (e == 8 * (int)blockIdx.x + i) ? (T)1 : (T)0;
    }
    __syncthreads();
    if constexpr (INVERSE)
        ntc::gs_stages(gbuf, psi, psi_sh, primeid, tid);
    else
        ntc::ct_stages(gbuf, psi, psi_sh, primeid, tid);
    __syncthreads();
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + ntc::BD * q;
        const int i = idx >> 8, m = idx & 255;
        const T v = GAS(i, m);
        const int k = 8 * (int)blockIdx.x + i;
        for (int l = 0; l < 4; ++l)
            out[l * 65536 + m * 256 + k] = (uint8_t)((v >> (8 * l)) & 0xFFu);
    }
}

// Self-test: random tile -> butterflies vs tensor cores, counts mismatching words (one block).
template <bool INVERSE>
__global__ void __launch_bounds__(128) tc_selftest_(const Global::Globals* Globals, const int primeid,
                                                    const uint8_t* Rplanes, const uint64_t mu, long* bad,
                                                    const int version) {
    using T = uint32_t;
    extern __shared__ char buffer[];
    __shared__ __align__(16) T ref[ntc::TILE_T];
    __shared__ T psi[ntc::BD], psi_sh[ntc::BD];
    const int tid = threadIdx.x;
    const T p = (T)C_.primes[primeid];
    psi[tid] = ((T*)(INVERSE ? G_->inv_psi[primeid] : G_->psi[primeid]))[tid];
    psi_sh[tid] = ((T*)(INVERSE ? G_->inv_psi_shoup[primeid] : G_->psi_shoup[primeid]))[tid];
    T* gbuf = ref;
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + ntc::BD * q;
        uint32_t h = (uint32_t)idx * 2654435761u ^ (uint32_t)primeid * 40503u;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        const T v = h % p;
        GAS(idx >> 8, idx & 255) = v;
        AS(idx >> 8, idx & 255) = v;
    }
    __syncthreads();
    if constexpr (INVERSE)
        ntc::gs_stages(gbuf, psi, psi_sh, primeid, tid);
    else
        ntc::ct_stages(gbuf, psi, psi_sh, primeid, tid);
    if (version == 2)
        tc::tc_core2<T>(buffer, Rplanes, primeid, mu, tid);
    else
        tc::tc_core<T>(buffer, Rplanes, primeid, mu, tid);
    __syncthreads();
    long mine = 0;
    for (int q = 0; q < 16; ++q) {
        const int idx = tid + ntc::BD * q;
        const int i = idx >> 8, e = idx & 255;
        if (AS(i, e) != GAS(i, e)) ++mine;
    }
    atomicAdd((unsigned long long*)bad, (unsigned long long)mine);
}
__global__ void tc_primes_(uint64_t* out) {
    for (int i = threadIdx.x; i < MAXP; i += blockDim.x)
        out[i] = C_.primes[i];
}
}  // namespace

static TcTables g_host_tables[8] = {};

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t p) { return (uint64_t)((__uint128_t)a * b % p); }
uint64_t powmod(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1;
    for (a %= p; e; e >>= 1, a = mulmod(a, a, p))
        if (e & 1) r = mulmod(r, a, p);
    return r;
}
// R[m][k] == c * g^(sigma(m) tau(k)) ? (g a primitive 256th root, sigma/tau permutations of 0..255)
struct Sep {
    bool ok = false;
    uint64_t g = 0, c = 0;
    uint8_t sigma[256], tau[256];
};
Sep analyzeCore(const std::vector<uint32_t>& R, uint64_t p) {
    Sep r;
    r.c = R[0];
    if (r.c == 0) return r;
    const uint64_t cinv = powmod(r.c, p - 2, p);
    std::vector<uint64_t> N(65536);
    for (int e = 0; e < 65536; ++e) N[e] = mulmod(R[e], cinv, p);
    for (int e = 0; e < 65536 && !r.g; ++e)
        if (powmod(N[e], 256, p) == 1 && powmod(N[e], 128, p) != 1) r.g = N[e];
    if (!r.g) return r;
    std::vector<std::pair<uint64_t, int>> pw(256);
    for (int e = 0; e < 256; ++e) pw[e] = {powmod(r.g, e, p), e};
    std::sort(pw.begin(), pw.end());
    std::vector<int> L(65536);
    for (int e = 0; e < 65536; ++e) {
        auto it = std::lower_bound(pw.begin(), pw.end(), std::make_pair(N[e], -1));
        if (it == pw.end() || it->first != N[e]) return r;
        L[e] = it->second;
    }
    int m1 = -1, k1 = -1;
    for (int e = 0; e < 65536 && m1 < 0; ++e)
        if (L[e] & 1) { m1 = e >> 8; k1 = e & 255; }
    if (m1 < 0) return r;
    int u = 1;  // inverse of L[m1][k1] mod 256
    while (((L[m1 * 256 + k1] * u) & 255) != 1) ++u;
    for (int m = 0; m < 256; ++m) r.sigma[m] = (uint8_t)((L[m * 256 + k1] * u) & 255);
    for (int k = 0; k < 256; ++k) r.tau[k] = (uint8_t)L[m1 * 256 + k];
    for (int m = 0; m < 256; ++m)
        for (int k = 0; k < 256; ++k)
            if (((r.sigma[m] * r.tau[k]) & 255) != L[m * 256 + k]) return r;
    bool seen_s[256] = {}, seen_t[256] = {};
    for (int i = 0; i < 256; ++i) { seen_s[r.sigma[i]] = true; seen_t[r.tau[i]] = true; }
    for (int i = 0; i < 256; ++i)
        if (!seen_s[i] || !seen_t[i]) return r;
    r.ok = true;
    return r;
}
// Build the v2 blob (see NTTtc_core.cuh) for a separable core; nullptr if the map is not separable.
const uint8_t* buildBlob(const uint8_t* dplanes, uint64_t p, int pid, bool inverse) {
    std::vector<uint8_t> planes(4 * 65536);
    cudaMemcpy(planes.data(), dplanes, planes.size(), cudaMemcpyDeviceToHost);
    std::vector<uint32_t> R(65536);
    for (int e = 0; e < 65536; ++e) {
        uint32_t v = 0;
        for (int l = 0; l < 4; ++l) v |= (uint32_t)planes[l * 65536 + e] << (8 * l);
        R[e] = v;
    }
    const Sep sp = analyzeCore(R, p);
    if (!sp.ok) {
        std::fprintf(stderr, "[tc_ntt] prime %d %s core is not separable: v2 unavailable for it\n", pid,
                     inverse ? "inverse" : "forward");
        return nullptr;
    }
    std::vector<uint8_t> blob(tc::TC2_BLOB, 0);
    uint64_t gp[256];
    for (int e = 0; e < 256; ++e) gp[e] = powmod(sp.g, e, p);
    for (int K1 = 0; K1 < 16; ++K1)
        for (int M0 = 0; M0 < 16; ++M0)
            for (int K0 = 0; K0 < 16; ++K0) {
                const uint64_t v = gp[(M0 * (K1 + 16 * K0)) & 255];
                for (int l = 0; l < 4; ++l)
                    blob[tc::TC2_A1 + l * 4096 + K1 * 256 + M0 * 16 + K0] = (uint8_t)((v >> (8 * l)) & 255);
            }
    for (int M1 = 0; M1 < 16; ++M1)
        for (int K1 = 0; K1 < 16; ++K1) {
            const uint64_t v = mulmod(sp.c, gp[(16 * M1 * K1) & 255], p);
            for (int l = 0; l < 4; ++l)
                blob[tc::TC2_A2 + l * 256 + M1 * 16 + K1] = (uint8_t)((v >> (8 * l)) & 255);
        }
    for (int k = 0; k < 256; ++k) blob[tc::TC2_TAU + k] = sp.tau[k];
    for (int m = 0; m < 256; ++m) blob[tc::TC2_SIGINV + sp.sigma[m]] = (uint8_t)m;
    for (int k = 0; k < 256; ++k) blob[tc::TC2_TAUINV + sp.tau[k]] = (uint8_t)k;
    uint8_t* d = nullptr;
    cudaMalloc(&d, blob.size());
    cudaMemcpy(d, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    return d;
}
}  // namespace

void buildTcNttTables(const Global::Globals* G, int device) {
    cudaSetDevice(device);
    TcTables& t = g_host_tables[device];
    uint64_t primes[MAXP];
    void* psi[MAXP];
    {
        uint64_t* d = nullptr;
        cudaMalloc(&d, sizeof(primes));
        tc_primes_<<<1, 64>>>(d);
        cudaMemcpy(primes, d, sizeof(primes), cudaMemcpyDeviceToHost);
        cudaFree(d);
        cudaMemcpy(psi, (const char*)G + offsetof(Global::Globals, psi), sizeof(psi), cudaMemcpyDeviceToHost);
    }
    const int nprimes = MAXP;
    for (int pid = 0; pid < nprimes; ++pid) {
        if (psi[pid] == nullptr || primes[pid] == 0 || primes[pid] >= (1ull << 28))
            continue;  // u32 chains only (4 chunks)
        uint8_t* R = nullptr;
        uint8_t* Rinv = nullptr;
        cudaMalloc(&R, 4 * 65536);
        cudaMalloc(&Rinv, 4 * 65536);
        tc_probe_<false><<<32, 128>>>(G, pid, R);
        tc_probe_<true><<<32, 128>>>(G, pid, Rinv);
        t.R[pid] = R;
        t.Rinv[pid] = Rinv;
        t.mu[pid] = ~0ull / primes[pid];
        cudaDeviceSynchronize();
        t.B[pid] = buildBlob(R, primes[pid], pid, false);
        t.Binv[pid] = buildBlob(Rinv, primes[pid], pid, true);
    }
    cudaDeviceSynchronize();
    setTcTables(t, device);
    if (const char* path = std::getenv("FIDESLIB_TC_NTT_DUMP"); path && *path) {
        // Structure study: prime 0's forward / inverse core maps as raw u32 [m][k], plus the prime.
        int pid = 0;
        while (pid < MAXP && !t.R[pid]) ++pid;
        if (pid < MAXP) {
            std::vector<uint8_t> planes(4 * 65536);
            std::vector<uint32_t> R(65536), Ri(65536);
            for (int which = 0; which < 2; ++which) {
                cudaMemcpy(planes.data(), which ? t.Rinv[pid] : t.R[pid], planes.size(), cudaMemcpyDeviceToHost);
                for (int e = 0; e < 65536; ++e) {
                    uint32_t v = 0;
                    for (int l = 0; l < 4; ++l) v |= (uint32_t)planes[l * 65536 + e] << (8 * l);
                    (which ? Ri : R)[e] = v;
                }
            }
            if (FILE* f = std::fopen(path, "wb")) {
                const uint64_t pr = primes[pid];
                std::fwrite(&pr, sizeof(pr), 1, f);
                std::fwrite(R.data(), 4, R.size(), f);
                std::fwrite(Ri.data(), 4, Ri.size(), f);
                std::fclose(f);
                std::printf("[tc_ntt] dumped prime %d (%lu) core maps to %s\n", pid, (unsigned long)pr, path);
            }
        }
    }
    if (const char* e = std::getenv("FIDESLIB_TC_NTT_SELFTEST"); e && std::atoi(e)) {
        for (int pid = 0; pid < nprimes && pid < MAXP; pid += 7) {
            if (!t.R[pid]) continue;
            const long bad = tcNttSelfTest(G, pid, device, 1);
            const long bad2 = tcNttSelfTest(G, pid, device, 2);
            std::printf("[tc_ntt] selftest prime %d: v1 %ld, v2 %ld mismatching words (fwd*1e6+inv)\n", pid, bad, bad2);
        }
    }
}

long tcNttSelfTest(const Global::Globals* G, int primeid, int device, int version) {
    cudaSetDevice(device);
    const TcTables& t = g_host_tables[device];
    long* d_bad = nullptr;
    cudaMalloc(&d_bad, 2 * sizeof(long));
    cudaMemset(d_bad, 0, 2 * sizeof(long));
    const int bytes = 4 * 128 * (2 * 8 + 1 + 1);  // the two-pass kernel's dynamic smem
    const uint8_t* tf = version == 2 ? t.B[primeid] : t.R[primeid];
    const uint8_t* ti = version == 2 ? t.Binv[primeid] : t.Rinv[primeid];
    if (!tf || !ti) return -1;
    tc_selftest_<false><<<1, 128, bytes>>>(G, primeid, tf, t.mu[primeid], d_bad, version);
    tc_selftest_<true><<<1, 128, bytes>>>(G, primeid, ti, t.mu[primeid], d_bad + 1, version);
    long h[2] = {0, 0};
    cudaMemcpy(h, d_bad, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d_bad);
    return h[0] * 1000000 + h[1];  // fwd * 1e6 + inv
}

}  // namespace FIDESlib
