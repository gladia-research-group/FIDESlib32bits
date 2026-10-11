#include "CKKS/ModUpConv2.cuh"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include "ModMult.cuh"

namespace FIDESlib::CKKS {

namespace {

constexpr int MU_TG = 16;    // targets per block
constexpr int MU_BX = 128;   // coefficients per block

std::atomic<int> g_on{[] {
    const char* e = std::getenv("FIDESLIB_BCONV2");
    return e ? std::atoi(e) : 0;
}()};

// Body for digit d at n live limbs; writes target j only where the original does (primeid_j < n or a special prime).
template <int CH>
__device__ __forceinline__ void modUpConv2Body(void** __restrict__ a, const int n, void** __restrict__ b, const int d,
                                               const Global::Globals* Globals) {
    const int m = C_.num_primeid_digit_from[d][n - 1];
    const int cnt = C_.num_primeid_digit_to[d][n - 1];
    const int k0 = blockIdx.y * MU_TG;
    if (k0 >= cnt)
        return;
    const int kn = min(MU_TG, cnt - k0);
    extern __shared__ uint32_t mu_smem[];
    uint32_t* sy = mu_smem;                       // m x MU_BX pre-scaled digit limbs
    uint32_t* smat = mu_smem + m * MU_BX;          // m x MU_TG matrix tile (zero past kn)
    int* sdst = (int*)(smat + m * MU_TG);          // MU_TG target prime ids
    const int tid = threadIdx.x;
    for (int x = tid; x < m * MU_TG; x += MU_BX) {
        const int i = x / MU_TG, k = x - i * MU_TG;
        smat[x] = k < kn ? G_->DecompAndModUp_matrix32[MODUPIDX_MATRIX(n - 1, d, C_.primeid_digit_from[d][i],
                                                                       C_.primeid_digit_to[d][k0 + k])]
                         : 0u;
    }
    if (tid < MU_TG)
        sdst[tid] = C_.primeid_digit_to[d][k0 + (tid < kn ? tid : 0)];
    const int idx = blockIdx.x * MU_BX + tid;
    for (int i = 0; i < m; ++i) {
        const int pi = C_.primeid_digit_from[d][i];
        sy[i * MU_BX + tid] = modreduce_lazy(
            (uint64_t)((const uint32_t*)a[i])[idx] * (uint32_t)G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, m - 1, pi)],
            pi);
    }
    __syncthreads();
    for (int k = 0; k < kn; k += 4) {
        const int t0 = sdst[k], t1 = sdst[k + 1], t2 = sdst[k + 2], t3 = sdst[k + 3];
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int i0 = 0; i0 < m; i0 += CH) {
#pragma unroll
            for (int c = 0; c < CH; ++c) {
                const int i = i0 + c;
                if (i < m) {
                    const uint64_t yi = sy[i * MU_BX + tid];
                    const uint4 r = *(const uint4*)(smat + i * MU_TG + k);
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
        const uint64_t acc[4] = {a0, a1, a2, a3};
        const int tt[4] = {t0, t1, t2, t3};
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (k + q < kn && (tt[q] < n || tt[q] >= C_.L))
                ((uint32_t*)b[k0 + k + q])[idx] = (uint32_t)acc[q];
    }
}

template <int CH>
__global__ void __launch_bounds__(MU_BX, 8) modUpConv2_(void** __restrict__ a, const int __grid_constant__ n,
                                                         void** __restrict__ b, const int __grid_constant__ d,
                                                         const Global::Globals* Globals) {
    modUpConv2Body<CH>(a, n, b, d, Globals);
}

template <int CH>
__global__ void __launch_bounds__(MU_BX, 8) modUpConv2Multi_(void*** __restrict__ atabs, const int __grid_constant__ n,
                                                              void*** __restrict__ btabs, const int* __restrict__ dids,
                                                              const Global::Globals* Globals) {
    modUpConv2Body<CH>(atabs[blockIdx.z], n, btabs[blockIdx.z], dids[blockIdx.z], Globals);
}

// products accumulated between lazy reductions: (max source prime) x (max target prime) headroom in u64 after a
// reduction leaves < 2^32; instantiated 8 / 4 / 2 / 1
int chunkOf(const Constants& hc) {
    uint64_t maxQ = 0, maxAll = 0;
    for (int i = 0; i < hc.L + hc.K; ++i) {
        maxAll = std::max<uint64_t>(maxAll, hc.primes[i]);
        if (i < hc.L)
            maxQ = std::max<uint64_t>(maxQ, hc.primes[i]);
    }
    const unsigned __int128 prod = (unsigned __int128)(maxQ - 1) * (maxAll - 1);
    const unsigned __int128 room = ((unsigned __int128)1 << 64) - ((unsigned __int128)1 << 32);
    const unsigned __int128 c = room / prod;
    return c >= 8 ? 8 : c >= 4 ? 4 : c >= 2 ? 2 : 1;
}

size_t smemFor(int m) {
    return (size_t)m * (MU_BX + MU_TG) * sizeof(uint32_t) + MU_TG * sizeof(int);
}

}  // namespace

bool modUpConv2On() {
    return g_on.load(std::memory_order_relaxed) > 0;
}

void setModUpConv2(int on) {
    g_on.store(on, std::memory_order_relaxed);
}

void launchModUpConv2(void** a, int n, void** b, int d, const Global::Globals* G, const Constants& hc, int maxSize,
                      cudaStream_t s) {
    const int cnt = hc.num_primeid_digit_to[d][n - 1];
    const dim3 grid{(uint32_t)hc.N / MU_BX, (uint32_t)std::max(1, (cnt + MU_TG - 1) / MU_TG)};
    const size_t smem = smemFor(maxSize);
    switch (chunkOf(hc)) {
        case 8: modUpConv2_<8><<<grid, MU_BX, smem, s>>>(a, n, b, d, G); break;
        case 4: modUpConv2_<4><<<grid, MU_BX, smem, s>>>(a, n, b, d, G); break;
        case 2: modUpConv2_<2><<<grid, MU_BX, smem, s>>>(a, n, b, d, G); break;
        default: modUpConv2_<1><<<grid, MU_BX, smem, s>>>(a, n, b, d, G); break;
    }
}

void launchModUpConv2Multi(void*** atabs, int n, void*** btabs, const int* dids, const int* dids_host, int ndig,
                           const Global::Globals* G, const Constants& hc, int maxSize, cudaStream_t s) {
    int cnt = 0;
    for (int z = 0; z < ndig; ++z)
        cnt = std::max(cnt, hc.num_primeid_digit_to[dids_host[z]][n - 1]);
    const dim3 grid{(uint32_t)hc.N / MU_BX, (uint32_t)std::max(1, (cnt + MU_TG - 1) / MU_TG), (uint32_t)ndig};
    const size_t smem = smemFor(maxSize);
    switch (chunkOf(hc)) {
        case 8: modUpConv2Multi_<8><<<grid, MU_BX, smem, s>>>(atabs, n, btabs, dids, G); break;
        case 4: modUpConv2Multi_<4><<<grid, MU_BX, smem, s>>>(atabs, n, btabs, dids, G); break;
        case 2: modUpConv2Multi_<2><<<grid, MU_BX, smem, s>>>(atabs, n, btabs, dids, G); break;
        default: modUpConv2Multi_<1><<<grid, MU_BX, smem, s>>>(atabs, n, btabs, dids, G); break;
    }
}

}  // namespace FIDESlib::CKKS
