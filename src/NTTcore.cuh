#pragma once
// Shared 256-point NTT/INTT core on an 8-row shared-memory tile (u32, Shoup), lifted from the two-pass
// kernel: used by the cluster kernel (NTTcluster.cu) and the tensor-core core's table probe (NTTtc.cu).
#include <cuda_runtime.h>
#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"
#include "NTThelper.cuh"

#define NTC_LD(p) __ldcs(p)

namespace FIDESlib {
namespace ntc {
using T = uint32_t;
constexpr ALGO algo = ALGO_SHOUP;
constexpr int M = 8;                 // int2 pairs per thread per row set (u32)
constexpr int BD = 128;              // threads per tile group (== the two-pass kernel's blockDim.x)
constexpr int GROUPS = 4;            // tile groups per block
constexpr int CL = 8;                // blocks per cluster
constexpr int TILES = GROUPS * CL;   // 32 tiles per limb (== the two-pass kernel's gridDim.x)
constexpr int TILE_T = 2 * BD * M;   // 2048 u32 per tile
constexpr int TILE_I2 = BD * M;      // 1024 int2 per tile
constexpr int REG_I2 = GROUPS * TILE_I2;  // 4096 int2 per block region (32 KB)
constexpr int LOGBD = 7;             // log2(BD)
constexpr int LOGBD1 = 8;            // 32 - __clz(BD): the two-pass kernel's `logBD` in the twiddle math

// group-local row accessors (the two-pass kernel's A/AS with buffer -> gbuf, blockDim.x -> BD)
#define GA(i) ((T*)gbuf + 2 * ntc::BD * (i))
#define GAS(i, e) (GA(i)[FIDESlib::swz_base<T>(e) ^ FIDESlib::swz_row<T>(i)])

// -------- forward (CT) butterfly stages: iteration 0, shared stages, then the warp-shuffle tail --------
__device__ __forceinline__ void ct_stages(T* gbuf, const T* psi, const T* psi_sh, const int primeid, const int tid) {
    const int j = tid << 1;
    int m = BD;
    int maskPsi = m;
    for (int i = 0; i < M; ++i) {
        const T a0 = GAS(i, tid);
        const T a1 = GAS(i, tid + m);
        GAS(i, tid) = modadd(a0, a1, primeid);
        GAS(i, tid + m) = modsub(a0, a1, primeid);
    }
    m >>= 1;
    maskPsi |= (maskPsi >> 1);
    int log_psi = LOGBD - 1;
#if FIDESLIB_NTT_WARP_SHFL
    constexpr bool use_shfl = (BD >= (1 << NTT_SHFL_STAGES));
#else
    constexpr bool use_shfl = false;
#endif
    constexpr int m_stop = use_shfl ? (1 << NTT_SHFL_STAGES) : 1;
    for (; m >= m_stop; m >>= 1, log_psi--, maskPsi |= (maskPsi >> 1)) {
        const int mask = m - 1;
        const int j1 = (mask & tid) | ((~mask & tid) << 1);
        const int j2 = j1 + m;
        const int psiid = (tid & maskPsi) >> log_psi;
        const T psiaux = psi[psiid];
        const T psiaux_sh = psi_sh[psiid];
        if (m >= warpSize)
            __syncthreads();
        else
            __syncwarp();
        for (int i = 0; i < M; ++i) {
            T& x0 = GAS(i, j1);
            T& x1 = GAS(i, j2);
            CT_butterfly<T, algo>(x0, x1, psiaux, primeid, psiaux_sh);
        }
    }
    if constexpr (use_shfl) {
        T psis[NTT_SHFL_STAGES], psis_sh[NTT_SHFL_STAGES];
#pragma unroll
        for (int k = 0; k < NTT_SHFL_STAGES; ++k) {
            const int psiid = (tid & maskPsi) >> log_psi;
            psis[k] = psi[psiid];
            psis_sh[k] = psi_sh[psiid];
            log_psi--;
            maskPsi |= (maskPsi >> 1);
        }
        const int j1_in = ((m - 1) & tid) | ((~(m - 1) & tid) << 1);
        __syncwarp();
        for (int i = 0; i < M; ++i) {
            T a0 = GAS(i, j1_in);
            T a1 = GAS(i, j1_in + m);
#pragma unroll
            for (int k = 0; k < NTT_SHFL_STAGES; ++k) {
                if (k)
                    warp_pair_exchange<T>(a0, a1, tid, NTT_SHFL_STAGES - 1 - k);
                CT_butterfly<T, algo>(a0, a1, psis[k], primeid, psis_sh[k]);
            }
            GAS(i, j) = a0;
            GAS(i, j + 1) = a1;
        }
    }
}

// -------- inverse (GS) stages: warp-shuffle head, shared stages, final add/sub --------
__device__ __forceinline__ void gs_stages(T* gbuf, const T* psi, const T* psi_sh, const int primeid, const int tid) {
    const int j = tid << 1;
    int m = 1;
    int maskPsi = (BD - 1);
    uint32_t log_psi = 0;
#if FIDESLIB_NTT_WARP_SHFL
    constexpr bool use_shfl = (BD >= (1 << NTT_SHFL_STAGES));
#else
    constexpr bool use_shfl = false;
#endif
    if constexpr (use_shfl) {
        T psis[NTT_SHFL_STAGES], psis_sh[NTT_SHFL_STAGES];
#pragma unroll
        for (int k = 0; k < NTT_SHFL_STAGES; ++k) {
            const int psiid = (tid & maskPsi) >> log_psi;
            psis[k] = psi[psiid];
            psis_sh[k] = psi_sh[psiid];
            maskPsi &= (maskPsi << 1);
            ++log_psi;
        }
        __syncwarp();
        constexpr int m_out = 1 << (NTT_SHFL_STAGES - 1);
        const int j1_out = ((m_out - 1) & tid) | ((~(m_out - 1) & tid) << 1);
        for (int i = 0; i < M; ++i) {
            T a0 = GAS(i, j);
            T a1 = GAS(i, j + 1);
#pragma unroll
            for (int k = 0; k < NTT_SHFL_STAGES; ++k) {
                if (k)
                    warp_pair_exchange<T>(a0, a1, tid, k - 1);
                GS_butterfly<T, algo>(a0, a1, psis[k], primeid, psis_sh[k]);
            }
            GAS(i, j1_out) = a0;
            GAS(i, j1_out + m_out) = a1;
        }
        m = 1 << NTT_SHFL_STAGES;
    }
    for (; m < BD; m <<= 1, maskPsi &= (maskPsi << 1), ++log_psi) {
        if (m >= warpSize)
            __syncthreads();
        else
            __syncwarp();
        const int mask = m - 1;
        const int j1 = (mask & tid) | (((~mask) << 1) & (tid << 1));
        const int j2 = j1 | m;
        const int psiid = (tid & maskPsi) >> log_psi;
        const T psiaux = psi[psiid];
        const T psiaux_sh = psi_sh[psiid];
        for (int i = 0; i < M; ++i) {
            T& a0 = GAS(i, j1);
            T& a1 = GAS(i, j2);
            GS_butterfly<T, algo>(a0, a1, psiaux, primeid, psiaux_sh);
        }
    }
    __syncthreads();
    for (int i = 0; i < M; ++i) {
        const T a0 = GAS(i, tid);
        const T a1 = GAS(i, tid + m);
        GAS(i, tid) = modadd(a0, a1, primeid);
        GAS(i, tid + m) = modsub(a0, a1, primeid);
    }
}

// -------- negacyclic scales (NTTfusions.cuh with blockDim.x/gridDim.x/blockIdx.x -> BD/TILES/bx) --------
__device__ __forceinline__ void fwd_negacyclic(T* gbuf, const int primeid, const T* psi, const T* psi_sh,
                                               const Global::Globals* Globals, const int tid, const int bx) {
    uint32_t pos1 = tid & (~1);
    pos1 = __brev(pos1) >> __clz((unsigned)BD);
    const T aux_3 = ((T*)G_->psi_no[primeid])[(tid & 1) * (TILES * M) + M * bx];
    T aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_sh[pos1]);
    const T root = C_.root[primeid];
    const T root_sh = C_.root_shoup[primeid];
    const T fourth = psi[1];
    const T fourth_sh = psi_sh[1];
    for (int i = 0; i < M; ++i) {
        if (i > 0)
            aux = modmult<algo>(aux, root, primeid, root_sh);
        const T aux2 = modmult<algo>(aux, fourth, primeid, fourth_sh);
        GAS(i, tid) = modmult<ALGO_BARRETT>(GAS(i, tid), aux, primeid);
        GAS(i, tid + BD) = modmult<ALGO_BARRETT>(GAS(i, tid + BD), aux2, primeid);
    }
}

__device__ __forceinline__ void bwd_negacyclic(T* gbuf, const int primeid, const T* psi, const T* psi_sh,
                                               const Global::Globals* Globals, const int tid, const int bx) {
    uint32_t pos1 = tid & (~1);
    pos1 = __brev(pos1) >> __clz((unsigned)BD);
    T aux_3 = ((T*)G_->inv_psi_no[primeid])[(tid & 1) * (TILES * M) + M * bx];
    aux_3 = modmult<ALGO_SHOUP>(aux_3, (T)C_.N, primeid, (T)C_.N_shoup[primeid]);
    T aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_sh[pos1]);
    const T root = C_.inv_root[primeid];
    const T root_sh = C_.inv_root_shoup[primeid];
    const T fourth = psi[1];
    const T fourth_sh = psi_sh[1];
    for (int i = 0; i < M; ++i) {
        if (i > 0)
            aux = modmult<algo>(aux, root, primeid, root_sh);
        const T aux2 = modmult<algo>(aux, fourth, primeid, fourth_sh);
        GAS(i, tid) = modmult<ALGO_BARRETT>(GAS(i, tid), aux, primeid);
        GAS(i, tid + BD) = modmult<ALGO_BARRETT>(GAS(i, tid + BD), aux2, primeid);
    }
}

// -------- the middle-scale (EOT) twiddles of the 4-step transform --------
// forward pass-1 epilogue: multiply and store the tile IN PLACE in its own region slot, in the contiguous
// int2 order pass 2 of the other blocks will gather (register-staged: the swizzled read positions differ
// from the write positions, so every thread reads first, barrier, then writes).
__device__ __forceinline__ void ntt_eot_store_inplace(T* gbuf, const T* psi, const T* psi_sh, const int primeid,
                                                      const Global::Globals* Globals, const int tid, const int bx) {
    const int j = tid << 1;
    T eot_tw[2], eot_step[2];
    {
        const uint32_t mask_lo_exp = (((C_.N) >> 1) | ((C_.N >> LOGBD1) - 1));
        const uint32_t clzN = __clz(C_.N) + 2;
        const uint32_t block_pos0 = bx * M;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t br_j = __brev(j + k) >> (32 - LOGBD1);
            const uint32_t exp = block_pos0 * br_j;
            const uint32_t hi_exp_br = __brev(exp << clzN) & (BD - 1);
            const uint32_t lo_exp = exp & mask_lo_exp;
            eot_tw[k] = modmult<algo>(((T*)G_->psi_no[primeid])[lo_exp * 2], psi[hi_exp_br], primeid, psi_sh[hi_exp_br]);
            eot_step[k] = ((T*)G_->psi_no[primeid])[br_j * 2];
        }
    }
    int2 out[M];
    for (int i = 0; i < M; ++i) {
        out[i].x = (int)modmult<ALGO_BARRETT>(GAS(i, j), eot_tw[0], primeid);
        out[i].y = (int)modmult<ALGO_BARRETT>(GAS(i, j + 1), eot_tw[1], primeid);
        eot_tw[0] = modmult<ALGO_BARRETT>(eot_tw[0], eot_step[0], primeid);
        eot_tw[1] = modmult<ALGO_BARRETT>(eot_tw[1], eot_step[1], primeid);
    }
    __syncthreads();
    for (int i = 0; i < M; ++i)
        ((int2*)gbuf)[BD * i + tid] = out[i];  // == the two-pass kernel's res[OFFSET_2T(i)] within the tile
}

// inverse pass-2 prologue: the middle-scale pre-multiply on the (swizzled) tile.
__device__ __forceinline__ void intt_eot_premult(T* gbuf, const T* psi, const T* psi_sh, const int primeid,
                                                 const Global::Globals* Globals, const int tid, const int bx) {
    const int j = tid << 1;
    T eot_tw[2], eot_step[2];
    {
        const uint32_t mask_lo_exp = (((C_.N) >> 1) | ((C_.N >> LOGBD1) - 1));
        const uint32_t clzN = __clz(C_.N) + 2;
        const uint32_t block_pos0 = bx * M;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint32_t br_j = __brev(j + k) >> (32 - LOGBD1);
            const uint32_t exp = block_pos0 * br_j;
            const uint32_t hi_exp_br = __brev(exp << clzN) & (BD - 1);
            const uint32_t lo_exp = exp & mask_lo_exp;
            eot_tw[k] = modmult<algo>(((T*)G_->inv_psi_no[primeid])[lo_exp << 1], psi[hi_exp_br], primeid,
                                      psi_sh[hi_exp_br]);
            eot_step[k] = br_j ? ((T*)G_->psi_no[primeid])[(C_.N - br_j) << 1] : (T)1;
        }
    }
    for (int i = 0; i < M; ++i) {
        GAS(i, j) = modmult<ALGO_BARRETT>(GAS(i, j), eot_tw[0], primeid);
        GAS(i, j + 1) = modmult<ALGO_BARRETT>(GAS(i, j + 1), eot_tw[1], primeid);
        eot_tw[0] = modmult<ALGO_BARRETT>(eot_tw[0], eot_step[0], primeid);
        eot_tw[1] = modmult<ALGO_BARRETT>(eot_tw[1], eot_step[1], primeid);
    }
}

// transposed int2 index of (row r = col_init+i, tile bx, lane pair j&2): the two-pass kernel's pos_transp
__device__ __forceinline__ int transp_i2(const int r, const int bx, const int j) {
    return (M / 2) * (TILES * r + bx) + (j & 2);
}

// write 4 int4 gathered in transposed order into the swizzled tile rows (the two-pass u32 load epilogue)
__device__ __forceinline__ void store_transposed_regs(T* gbuf, const int4* temp, const int j) {
    const int col_init = j & ~2;
#pragma unroll
    for (int t_ = 0; t_ < 4; ++t_) {
        const int row = 2 * (j & 2) + t_;
        ((int4*)GA(row))[swz_quad<T>(row, col_init)] = swz_perm4(temp[t_], swz_lx<T>(row, col_init));
    }
}

}  // namespace ntc
}  // namespace FIDESlib
