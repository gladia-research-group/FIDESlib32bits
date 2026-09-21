//
// Created by carlosad on 27/09/24.
//

#include <printf.h>
#include <stdexcept>
#include <string>
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/KskSeedExpand.cuh"
#include "CKKS/Rescale.cuh"
#include "Rotation.cuh"

#include <cooperative_groups.h>
#include <cuda/barrier>

// Evict-first loads on read-once copy sources (see AddSub.cu; same knob).
#ifndef FIDESLIB_PW_LDCS
#define FIDESLIB_PW_LDCS 1
#endif
#if FIDESLIB_PW_LDCS
#define FIDESLIB_PW_STREAM_LD(p) __ldcs(p)
#else
#define FIDESLIB_PW_STREAM_LD(p) (*(p))
#endif
//#include "cooperative_groups/memcpy_async.h"
namespace cg = cooperative_groups;

namespace FIDESlib {
namespace CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2, void** l3,
                                    void** l4) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l4[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        res = modadd(res, aux, primeid);
        res = modadd(res, modmult<algo>(((T*)l2[blockIdx.y])[idx], ((T*)l3[blockIdx.y])[idx], primeid), primeid);
        ((T*)l[blockIdx.y])[idx] = res;
    } else {
        using T = uint32_t;
    }
}

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0,
                                  void** bc1, void** in, void** aux) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux0 = ((T*)in[blockIdx.y])[idx];
        T res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c1[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid),
                                    C_.P[primeid], primeid, C_.P_shoup[primeid]);
        res = modadd(res, aux0, primeid);
        res = modadd(res,
                     modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc1[blockIdx.y])[idx], primeid),
                                         C_.P[primeid], primeid, C_.P_shoup[primeid]),
                     primeid);
        ((T*)c1[blockIdx.y])[idx] = res;
        aux0 = ((T*)aux[blockIdx.y])[idx];
        res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid),
                                  C_.P[primeid], primeid, C_.P_shoup[primeid]);
        res = modadd(res, aux0, primeid);

        ((T*)c0[blockIdx.y])[idx] = res;
    } else {
        using T = uint32_t;
    }
}

__global__ void mult1Add2_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
    } else {
        using T = uint32_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
    }
}

template <typename T>
__device__ __forceinline__ void addMult__(T* l, const T* l1, const T* l2, const int primeid) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    l[idx] = modadd(l[idx], modmult<algo>(l1[idx], l2[idx], primeid), primeid);
}

template <typename T>
__global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid) {
    addMult__<T>(l, l1, l2, primeid);
}

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    //    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        addMult__<uint64_t>((uint64_t*)l[blockIdx.y], (uint64_t*)l1[blockIdx.y], (uint64_t*)l2[blockIdx.y], primeid);
    } else {
        addMult__<uint32_t>((uint32_t*)l[blockIdx.y], (uint32_t*)l1[blockIdx.y], (uint32_t*)l2[blockIdx.y], primeid);
    }
}

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //    if (idx == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        ((uint64_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint64_t*)l1[blockIdx.y])[idx], ((uint64_t*)l2[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint32_t*)l1[blockIdx.y])[idx], ((uint32_t*)l2[blockIdx.y])[idx], primeid);
    }
}

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        uint64_t in = ((uint64_t*)l1[blockIdx.y])[idx];
        ((uint64_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    } else {
        uint32_t in = ((uint32_t*)l1[blockIdx.y])[idx];
        ((uint32_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    }
};

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1,
                                      const __grid_constant__ int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        uint64_t in2_0 = ((uint64_t*)c2_key_switched_0[blockIdx.y])[idx];
        uint64_t in2_1 = ((uint64_t*)c2_key_switched_1[blockIdx.y])[idx];
        uint64_t in0 = ((uint64_t*)c0_res[blockIdx.y])[idx];
        uint64_t ok = modadd(modmult<ALGO_BARRETT>(in0, in0, primeid), in2_0, primeid);
        ((uint64_t*)c0_res[blockIdx.y])[idx] = ok;
        uint64_t in1 = ((uint64_t*)c1[blockIdx.y])[idx];
        uint64_t aux = modmult<ALGO_BARRETT>(in0, in1, primeid);
        uint64_t aux2 = modadd(aux, aux, primeid);
        ok = modadd(aux2, in2_1, primeid);
        ((uint64_t*)c1[blockIdx.y])[idx] = ok;
    } else {
    }
}

// n32: this is the bootstrap's MODULUS RAISE. Guarding the body on
// `ISU64(primeid) && ISU64(0)` with no else writes NOTHING on a uniform-U32 chain
// (constants.type == 0, so ISU64 is false for every prime), and the target limbs arrive
// from RNSPoly::grow -> generate() unzeroed, so EvalMod would consume uninitialized pool
// memory and the bootstrap would decrypt to NaN.
// Handle every width combination: read at the source limb's width, switch modulus in a type
// wide enough to hold both moduli, store at the target limb's width. The mixed cases are not
// exercised by a uniform chain but must not be silent no-ops either.
__device__ __forceinline__ void broadcastLimb0Body(const void* src, void* dst, const int idx, const int primeid) {
    if (ISU64(0)) {
        uint64_t in = ((const uint64_t*)src)[idx];
        SwitchModulus(in, 0, primeid);
        if (ISU64(primeid))
            ((uint64_t*)dst)[idx] = in;
        else
            ((uint32_t*)dst)[idx] = (uint32_t)in;
    } else {
        const uint32_t in32 = ((const uint32_t*)src)[idx];
        if (ISU64(primeid)) {
            uint64_t in = in32;
            SwitchModulus(in, 0, primeid);
            ((uint64_t*)dst)[idx] = in;
        } else {
            uint32_t in = in32;
            SwitchModulus(in, 0, primeid);
            ((uint32_t*)dst)[idx] = in;
        }
    }
}

__global__ void broadcastLimb0_(void** a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y + 1;
    broadcastLimb0Body(a[0], a[primeid], idx, primeid);
}

// CENTRED multi-limb CRT lift for COEFF-encoded plaintexts (d == 2).
//
// Why this is NOT compositeModRaise_: that kernel centres each Garner TERM against its own q_k
// (SwitchModulus's halfQ = source prime >> 1), which is right for ModRaise — whose input really
// is small relative to q0 — but wrong here. The whole point of a d-limb coeff lift is that the
// encoded integer m spans the PRODUCT Q0 = q0*q1, so the AGGREGATE must be centred against
// Q0/2, not each term against its own prime. Getting this wrong is silent: the reconstruction
// returns a plausible but wrong plaintext.
//
// Exact in u64: t1 < q1 and q0 < 2^32 (NATIVEINT bound), so M = r0 + t1*q0 < Q0 <= 2^60.
// Grid: {N/threads, limbs}; src[k] is a raw snapshot of source limb k.
__global__ void coeffLiftCentered2_(void** a, void** src, const __grid_constant__ uint64_t q0,
                                    const __grid_constant__ uint64_t q1,
                                    const __grid_constant__ uint64_t q0inv_mod_q1,
                                    const __grid_constant__ uint64_t Qhalf,
                                    const uint64_t* Q0_mod_qi) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y;

    const uint64_t r0 = ISU64(0) ? ((const uint64_t*)src[0])[idx]
                                 : (uint64_t)((const uint32_t*)src[0])[idx];
    const uint64_t r1 = ISU64(1) ? ((const uint64_t*)src[1])[idx]
                                 : (uint64_t)((const uint32_t*)src[1])[idx];

    // Garner: t1 = (r1 - r0) * q0^{-1} mod q1, then M = r0 + t1*q0 in [0, Q0).
    const uint64_t r0m  = r0 % q1;
    const uint64_t diff = (r1 >= r0m) ? (r1 - r0m) : (r1 + q1 - r0m);
    const uint64_t t1   = (uint64_t)(((__uint128_t)diff * q0inv_mod_q1) % q1);
    const uint64_t M    = r0 + t1 * q0;

    const uint64_t qi = C_.primes[primeid];
    uint64_t v = M % qi;
    if (M > Qhalf)                       // negative representative: subtract Q0
        v = (v + qi - Q0_mod_qi[primeid]) % qi;

    if (ISU64(primeid)) ((uint64_t*)a[primeid])[idx] = v;
    else                ((uint32_t*)a[primeid])[idx] = (uint32_t)v;
}

// COMPOSITESCALING ModRaise (see header). Grid: {N/threads, limbs}; src[k] holds a SNAPSHOT
// of source limb k's coefficients (raw device copy, width of prime k). All arithmetic is
// width-branched per prime; the accumulator uses the TARGET prime's width.
__global__ void compositeModRaise_(void** a, void** src, const __grid_constant__ int d, const uint64_t* qhatinv,
                                   const uint64_t* qhat) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y;
    const int limbs = gridDim.y;

    if (ISU64(primeid)) {
        uint64_t acc = 0;
        for (int k = 0; k < d; ++k) {
            uint64_t x;
            if (ISU64(k)) {
                x = ((const uint64_t*)src[k])[idx];
                x = modmult<ALGO_BARRETT>(x, qhatinv[k], k);
            } else {
                uint32_t x32 = ((const uint32_t*)src[k])[idx];
                x32 = modmult<ALGO_BARRETT>(x32, (uint32_t)qhatinv[k], k);
                x = x32;
            }
            SwitchModulus(x, k, primeid);
            acc = modadd(acc, modmult<ALGO_BARRETT>(x, qhat[k * limbs + primeid], primeid), primeid);
        }
        ((uint64_t*)a[primeid])[idx] = acc;
    } else {
        uint32_t acc = 0;
        for (int k = 0; k < d; ++k) {
            uint32_t x;
            if (ISU64(k)) {
                // wide source, narrow target: switch modulus in 64-bit, then narrow
                uint64_t x64 = ((const uint64_t*)src[k])[idx];
                x64 = modmult<ALGO_BARRETT>(x64, qhatinv[k], k);
                SwitchModulus(x64, k, primeid);
                x = (uint32_t)x64;
            } else {
                x = ((const uint32_t*)src[k])[idx];
                x = modmult<ALGO_BARRETT>(x, (uint32_t)qhatinv[k], k);
                SwitchModulus(x, k, primeid);
            }
            acc = modadd(acc, modmult<ALGO_BARRETT>(x, (uint32_t)qhat[k * limbs + primeid], primeid), primeid);
        }
        ((uint32_t*)a[primeid])[idx] = acc;
    }
}

__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    broadcastLimb0Body(limb0[0], a[blockIdx.y], idx, primeid);
}

__global__ void copy_(void** a, void** b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (ISU64(blockIdx.y)) {
        ((uint64_t*)b[blockIdx.y])[idx] = ((uint64_t*)a[blockIdx.y])[idx];
    } else {
        ((uint32_t*)b[blockIdx.y])[idx] = ((uint32_t*)a[blockIdx.y])[idx];
    }
}

/* Vectorized limb copy: FOUR elements per thread instead of one.
 *
 * copy_ moves one element per thread, so its warp count tracks the ELEMENT count, not the
 * byte count. A 32-bit chain carries ~2x the limbs of a 64-bit chain at the same logQ, so
 * for identical bytes it launches ~1.83x the warps — and a pure copy has no arithmetic to
 * hide the extra issue cost. This kernel makes each thread move 16 B (uint4) or 32 B
 * (ulonglong4), so the copy is bandwidth-bound rather than issue-bound on both widths.
 *
 * Width selection is deliberately IDENTICAL to copy_ (ISU64(blockIdx.y)) so behaviour is
 * bit-for-bit unchanged. Grid must be {N/512, limbs} with 128 threads; the caller checks
 * N % 512 == 0 and falls back to copy_ otherwise (neither kernel takes a length argument,
 * so the grid must cover N exactly). Alignment holds: limb strides are N*4 / N*8 bytes,
 * both multiples of 32, on top of cudaMalloc's 256 B base alignment. */
__global__ void copy_v4_(void** a, void** b) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (ISU64(blockIdx.y)) {
        ((ulonglong4*)b[blockIdx.y])[i] = ((ulonglong4*)a[blockIdx.y])[i];
    } else {
        ((uint4*)b[blockIdx.y])[i] = ((uint4*)a[blockIdx.y])[i];
    }
}

/* TYPE-UNAWARE limb copy, parameterised by BYTES PER THREAD.
 *
 * A copy moves BYTES — it has no reason to know whether the limb holds u32 or u64 elements.
 * Dropping the width branch removes a constant-memory load (ISU64) and a branch per thread,
 * makes mixed-width limbs correct by construction, and sidesteps the latent
 * ISU64(blockIdx.y) confusion in copy_/copy_v4_ (that macro wants a PRIME ID; blockIdx.y is
 * a limb SLOT — harmless only while both chains are width-uniform).
 *
 * BYTES is the tuning axis rather than "elements": it is the only unit comparable across
 * limb widths AND portable across GPUs, so retuning for other hardware means changing one
 * number. Implemented uniformly as BYTES/16 stores of uint4, the widest portable vector
 * type — deliberately NOT special-casing particular widths to particular vector types, which
 * would bake this architecture's quirks into the byte knob.
 *
 * OBSERVATION for whoever retunes (recorded, not baked in): the vector TYPE appears to matter
 * independently of the byte count. The retired copy_v4_ moved 32 B/thread as ONE ulonglong4
 * and measured 10.54 us/call on the n64 chain, against 12.58 for two uint4s at identical
 * bytes AND identical block count. If a future architecture wants 32 B+, it is worth probing
 * a wider vector type as a SEPARATE axis rather than assuming bytes alone determine it.
 *
 * Grid must be {bytes_per_limb/(BYTES*128), nlimbs}, block 128 — the kernel carries no
 * length, so the grid has to cover the limb exactly. */
template <int BYTES>
__global__ void copy_bytes_(void** a, void** b) {
    constexpr int V = BYTES / 16;  // uint4 == 16 B
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll
    for (int q = 0; q < V; ++q)
        ((uint4*)b[blockIdx.y])[V * i + q] = FIDESLIB_PW_STREAM_LD((const uint4*)a[blockIdx.y] + V * i + q);
}

/* Cross-TU launcher. A __global__ TEMPLATE launched from a TU that only sees its declaration
 * gets a weak local stub with no device code in that TU's fatbin => 'invalid device function'.
 * Keep every instantiation here. */

// Bytes-indexed vectorizations of the last two 1-element-per-thread pointwise kernels. Same
// conventions as AddSub.cu's family (16 B/thread; the optimum is set by the grid shape,
// grid.y == nlimbs, not by the kernel's arithmetic).
template <int BYTES, ALGO algo>
__global__ void Scalar_mult_bytes_(void** a, const uint64_t* b, const __grid_constant__ int primeid_init,
                                   const uint64_t* shoup_mu) {
    constexpr int V = BYTES / 16;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (ISU64(primeid)) {
        const uint64_t s = b[primeid], mu = shoup_mu ? shoup_mu[primeid] : 0;
#pragma unroll
        for (int q = 0; q < V; ++q) {
            ulonglong2 va = ((ulonglong2*)a[blockIdx.y])[V * i + q];
            va.x = modmult<algo>((uint64_t)va.x, s, primeid, mu);
            va.y = modmult<algo>((uint64_t)va.y, s, primeid, mu);
            ((ulonglong2*)a[blockIdx.y])[V * i + q] = va;
        }
    } else {
        const uint32_t s = (uint32_t)b[primeid], mu = (uint32_t)(shoup_mu ? shoup_mu[primeid] : 0);
#pragma unroll
        for (int q = 0; q < V; ++q) {
            uint4 va = ((uint4*)a[blockIdx.y])[V * i + q];
            va.x = modmult<algo>(va.x, s, primeid, mu);
            va.y = modmult<algo>(va.y, s, primeid, mu);
            va.z = modmult<algo>(va.z, s, primeid, mu);
            va.w = modmult<algo>(va.w, s, primeid, mu);
            ((uint4*)a[blockIdx.y])[V * i + q] = va;
        }
    }
}

template <int BYTES>
__global__ void eval_linear_w_sum_bytes_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w,
                                         const __grid_constant__ int primeid_init) {
    constexpr int V = BYTES / 16;
    constexpr ALGO algo = ALGO_BARRETT;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (ISU64(primeid)) {
#pragma unroll
        for (int q = 0; q < V; ++q) {
            ulonglong2 acc;
            const ulonglong2 v0 = ((const ulonglong2*)(bs[0])[blockIdx.y])[V * idx + q];
            acc.x = modmult<algo>((uint64_t)v0.x, w[primeid], primeid);
            acc.y = modmult<algo>((uint64_t)v0.y, w[primeid], primeid);
            for (int i = 1; i < n; ++i) {
                const ulonglong2 vi = ((const ulonglong2*)(bs[i])[blockIdx.y])[V * idx + q];
                const uint64_t wi = w[i * MAXP + primeid];
                acc.x = modadd((uint64_t)acc.x, modmult<algo>((uint64_t)vi.x, wi, primeid), primeid);
                acc.y = modadd((uint64_t)acc.y, modmult<algo>((uint64_t)vi.y, wi, primeid), primeid);
            }
            ((ulonglong2*)a[blockIdx.y])[V * idx + q] = acc;
        }
    } else {
#pragma unroll
        for (int q = 0; q < V; ++q) {
            uint4 acc;
            const uint4 v0 = ((const uint4*)(bs[0])[blockIdx.y])[V * idx + q];
            const uint32_t w0 = (uint32_t)w[primeid];
            acc.x = modmult<algo>(v0.x, w0, primeid);
            acc.y = modmult<algo>(v0.y, w0, primeid);
            acc.z = modmult<algo>(v0.z, w0, primeid);
            acc.w = modmult<algo>(v0.w, w0, primeid);
            for (int i = 1; i < n; ++i) {
                const uint4 vi = ((const uint4*)(bs[i])[blockIdx.y])[V * idx + q];
                const uint32_t wi = (uint32_t)w[i * MAXP + primeid];
                acc.x = modadd(acc.x, modmult<algo>(vi.x, wi, primeid), primeid);
                acc.y = modadd(acc.y, modmult<algo>(vi.y, wi, primeid), primeid);
                acc.z = modadd(acc.z, modmult<algo>(vi.z, wi, primeid), primeid);
                acc.w = modadd(acc.w, modmult<algo>(vi.w, wi, primeid), primeid);
            }
            ((uint4*)a[blockIdx.y])[V * idx + q] = acc;
        }
    }
}

void launchScalarMultBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, const uint64_t* b, int primeid_init,
                           const uint64_t* shoup_mu, int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 32: Scalar_mult_bytes_<32, ALGO_BARRETT><<<grid, block, 0, stream>>>(a, b, primeid_init, shoup_mu); break;
        case 64: Scalar_mult_bytes_<64, ALGO_BARRETT><<<grid, block, 0, stream>>>(a, b, primeid_init, shoup_mu); break;
        default: Scalar_mult_bytes_<16, ALGO_BARRETT><<<grid, block, 0, stream>>>(a, b, primeid_init, shoup_mu); break;
    }
}

void launchEvalLinearWSumBytes(dim3 grid, dim3 block, cudaStream_t stream, int n, void** a, void*** bs, uint64_t* w,
                               int primeid_init, int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 32: eval_linear_w_sum_bytes_<32><<<grid, block, 0, stream>>>(n, a, bs, w, primeid_init); break;
        case 64: eval_linear_w_sum_bytes_<64><<<grid, block, 0, stream>>>(n, a, bs, w, primeid_init); break;
        default: eval_linear_w_sum_bytes_<16><<<grid, block, 0, stream>>>(n, a, bs, w, primeid_init); break;
    }
}

void launchCopyBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 16: copy_bytes_<16><<<grid, block, 0, stream>>>(a, b); break;
        case 32: copy_bytes_<32><<<grid, block, 0, stream>>>(a, b); break;
        case 64: copy_bytes_<64><<<grid, block, 0, stream>>>(a, b); break;
        default: throw std::runtime_error("launchCopyBytes: bytes_per_thread must be 16, 32 or 64");
    }
}

__global__ void copy1D_(void* a, void* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    ((uint64_t*)b)[idx] = ((uint64_t*)a)[idx];
}

template <ALGO algo>
__global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid_init,
                             const uint64_t* shoup_mu) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modmult<algo>(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid, shoup_mu ? shoup_mu[primeid] : 0);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modmult<algo>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid,
                                                        (uint32_t)(shoup_mu ? shoup_mu[primeid] : 0));
    }
}

__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w,
                                   const __grid_constant__ int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        uint64_t res = modmult<algo>(((uint64_t*)(bs[0])[blockIdx.y])[idx], w[primeid], primeid);
        for (int i = 1; i < n; ++i) {
            uint64_t temp = modmult<algo>(((uint64_t*)(bs[i])[blockIdx.y])[idx], w[i * MAXP + primeid], primeid);
            res = modadd(res, temp, primeid);
        }
        ((uint64_t*)a[blockIdx.y])[idx] = res;
    } else {
        uint32_t res = modmult<algo>(((uint32_t*)(bs[0])[blockIdx.y])[idx], (uint32_t)w[primeid], primeid);
        for (int i = 1; i < n; ++i) {
            uint32_t temp = modmult<algo>(((uint32_t*)(bs[i])[blockIdx.y])[idx], (uint32_t)(w[i * MAXP + primeid]), primeid);
            res = modadd(res, temp, primeid);
        }
        ((uint32_t*)a[blockIdx.y])[idx] = res;
    }
}

// KSK bit-packing: funnelshift extraction of coefficient idx from a bits-per-coefficient packed
// stream. Reads two overlapping 32-bit words; warp neighbors overlap so the extra word is
// L1-served and DRAM sees ~bits/32 of the dense stream. The producer (packKsk_) zeroes one
// guard word past the stream so the idx==N-1 speculative q[1] read is always in-bounds.
__device__ __forceinline__ uint32_t kskUnpack(const void* p, const uint32_t idx, const uint32_t bits,
                                              const uint32_t mask) {
    const uint32_t bitoff = idx * bits;
    const uint32_t* q = (const uint32_t*)p + (bitoff >> 5);
    return __funnelshift_r(q[0], q[1], bitoff & 31) & mask;
}

__global__ void expandKskA_(uint32_t* out, const KskSeedWords seed, const int digit, const uint32_t p,
                            const uint32_t n16, const int N) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        out[idx] = kskexpand::expand_coeff(seed.k, (uint32_t)digit, p, (uint32_t)idx, n16);
}

__global__ void expandKskA64_(uint64_t* out, const KskSeedWords seed, const int digit, const uint64_t p,
                              const uint32_t n8, const int N) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        out[idx] = kskexpand::expand_coeff64(seed.k, (uint32_t)digit, p, (uint32_t)idx, n8);
}

__global__ void packKsk_(uint32_t* out, const uint32_t* in, const int N, const int bits) {
    const uint32_t total = (uint32_t)(((uint64_t)N * bits + 31) >> 5);
    const uint32_t w = threadIdx.x + blockIdx.x * blockDim.x;
    if (w > total)
        return;
    if (w == total) {  // zeroed guard word for the consumer funnelshift
        out[w] = 0;
        return;
    }
    const uint32_t mask = (1u << bits) - 1u;
    const uint64_t bit0 = (uint64_t)w << 5;
    uint32_t k = (uint32_t)(bit0 / bits);
    const uint32_t off = (uint32_t)(bit0 - (uint64_t)k * bits);
    uint64_t acc = (uint64_t)(in[k] & mask) >> off;
    for (uint32_t filled = bits - off; filled < 32 && k + 1 < (uint32_t)N; filled += bits)
        acc |= (uint64_t)(in[++k] & mask) << filled;
    out[w] = (uint32_t)acc;
}

// Min-blocks tier for the packed dot-kernel arms. sm_120 has maxThreadsPerSM = 1536, so
// min-blocks 16 x 128 = 2048 is infeasible there (ptxas silently ignores it and occupancy drops);
// pin 12 on >= sm_120, keep 16 on earlier architectures.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define FIDESLIB_DOT_MINCTA_PACKED 12
#else
#define FIDESLIB_DOT_MINCTA_PACKED 16
#endif

// These dot kernels are register-occupancy-limited, not DRAM-bound: __launch_bounds__(128, 12)
// caps registers at ~42 and raises residency so the streamed KSK reads have latency cover.
// KSK_BITS is the COMPILE-TIME packed width (0 = dense): compile-time width makes the mask/shift
// immediates, and the packed arms pin min-blocks to hold the occupancy tier.
// REGEN arms regenerate the kska operand from the key's 256-bit seed (KskSeedExpand.cuh, SPEC v1)
// instead of streaming it from DRAM. Stage-A shape: one 16-word ChaCha12 block serves 16
// consecutive slots, so threads 0..num_d*8-1 each compute ONE block into dynamic smem, one sync,
// then every thread consumes its word. The rare rejection escalates per-thread with full blocks —
// the exact expand_coeff attempt sequence, so regen values are bit-identical to the expanded rows.
// u32 (type==0) chains only; host gates on ksk_seed_set + FIDESLIB_KSK_REGEN.
template <int KSK_BITS, bool REGEN>
__global__ void __launch_bounds__(128, KSK_BITS ? FIDESLIB_DOT_MINCTA_PACKED : 12)
    fusedDotKSK_2_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id,
                   int num_special, int init, KskSeedWords aseed, uint32_t n16) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int blky = blockIdx.y + init;
    //num_special = C_.K;
    int primeid;
    if (blky < num_special) {
        primeid = C_.primeid_digit_to[0][blky];
    } else {
        primeid = C_.primeid_partition[id][blky - num_special];
    }

    const int primeid_digit = C_.primeid_digit[primeid];

    int pos_dec = blky - num_special;

    extern __shared__ uint32_t ks_smem[];  // REGEN only: num_d*128 keystream words
    if constexpr (REGEN) {
        const uint32_t pval = (uint32_t)C_.primes[primeid];
        if (threadIdx.x < num_d * 8) {
            const int di = threadIdx.x >> 3, sub = threadIdx.x & 7;
            uint32_t o[16];
            kskexpand::chacha_block(aseed.k, (uint32_t)(blockIdx.x * 8 + sub), (uint32_t)di, pval, o);
#pragma unroll
            for (int w = 0; w < 16; ++w)
                ks_smem[di * 128 + sub * 16 + w] = o[w];
        }
        __syncthreads();
    }

    if (C_.type == 0) {
        // All-U32 chain fast path: the generic arm below promotes every operand to uint64_t and runs
        // 64-bit Barrett per term. 32-bit Barrett (Neal_mult_32) + u32 accumulators compute the
        // IDENTICAL canonical residues (same reduce-then-add order) in fewer instructions and
        // registers, the occupancy limiter of this kernel. Grid-uniform branch.
        uint32_t a1, a2;
        [[maybe_unused]] uint32_t pval, m_p;
        if constexpr (REGEN) {
            pval = (uint32_t)C_.primes[primeid];
            m_p = (0xFFFFFFFFu / pval) * pval;  // exact-uniform rejection threshold (spec v1)
        }
        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            const int p = decomp ? pos_dec : pos;
            const uint32_t in = ((uint32_t*)digits[i + decomp * 3 * C_.dnum][p])[idx];
            uint32_t kska, kskb;
            if constexpr (REGEN) {
                uint32_t v = ks_smem[i * 128 + threadIdx.x];
                if (v >= m_p) {  // rare escalation: expand_coeff's attempts t=1..15, verbatim
                    uint32_t o[16];
                    for (uint32_t t = 1; t < (uint32_t)kskexpand::kTMax; ++t) {
                        kskexpand::chacha_block(aseed.k, (uint32_t)(idx >> 4) + t * n16, (uint32_t)i, pval, o);
                        v = o[idx & 15];
                        if (v < m_p)
                            break;
                    }
                }
                kska = v % pval;
            } else if constexpr (KSK_BITS) {
                kska = kskUnpack(digits[C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                 (1u << KSK_BITS) - 1u);
            } else {
                kska = ((uint32_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
            }
            if constexpr (KSK_BITS) {
                kskb = kskUnpack(digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                 (1u << KSK_BITS) - 1u);
            } else {
                kskb = ((uint32_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
            }
            const uint32_t m1 = modmult<ALGO_BARRETT>(in, kska, primeid);
            const uint32_t m2 = modmult<ALGO_BARRETT>(in, kskb, primeid);
            if (i == 0) {
                a1 = m1;
                a2 = m2;
            } else {
                a1 = modadd(a1, m1, primeid);
                a2 = modadd(a2, m2, primeid);
            }
        }
        if (primeid < C_.L) {
            ((uint32_t*)out1[pos_dec])[idx] = a1;
            ((uint32_t*)out2[pos_dec])[idx] = a2;
        } else {
            ((uint32_t*)sout1[primeid - C_.L])[idx] = a1;
            ((uint32_t*)sout2[primeid - C_.L])[idx] = a2;
        }
        return;
    }

    /*
    int i = 0;
    bool decomp = (i == primeid_digit);
    int pos = C_.pos_in_digit[i][primeid];

    uint64_t aux1, aux2;

    uint64_t in = ((uint64_t*)digits[0 + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
    aux1 = modmult<ALGO_BARRETT>(in, ((uint64_t*)digits[C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx],
                                 primeid);
    aux2 = modmult<ALGO_BARRETT>(
        in, ((uint64_t*)digits[2 * C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

    for (i = 1; i < num_d; ++i) {
        decomp = (i == primeid_digit);
        pos = C_.pos_in_digit[i][primeid];
        in = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        uint64_t add1 = modmult<ALGO_BARRETT>(
            in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        uint64_t add2 = modmult<ALGO_BARRETT>(
            in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        aux1 = modadd(aux1, add1, primeid);
        aux2 = modadd(aux2, add2, primeid);
    }
    */
    uint64_t aux1, aux2;

    for (int i = 0; i < num_d; ++i) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];

        //printf("Digit %d: in: %p\n", i, digits);
        //printf("Digit %d: in: %p, kska: %p, kskb: %p\n", i, digits[i + decomp * 3 * C_.dnum],
        //       digits[C_.dnum + i + decomp * 3 * C_.dnum], digits[2 * C_.dnum + i + decomp * 3 * C_.dnum]);

        uint64_t in;
        if (ISU64(primeid)) {
            in = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        } else {
            in = ((uint32_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }
        uint64_t add1;
        uint64_t add2;
        if (ISU64(primeid)) {
            add1 = modmult<ALGO_BARRETT>(
                in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
            add2 = modmult<ALGO_BARRETT>(
                in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        } else {
            add1 = modmult<ALGO_BARRETT>(
                in, (uint64_t)((uint32_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
            add2 = modmult<ALGO_BARRETT>(
                in, (uint64_t)((uint32_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        }

        if (i == 0) {
            aux1 = add1;
            aux2 = add2;
        } else {
            aux1 = modadd(aux1, add1, primeid);
            aux2 = modadd(aux2, add2, primeid);
        }
    }

    if (primeid < C_.L) {
        if (ISU64(primeid)) {
            ((uint64_t*)out1[pos_dec])[idx] = aux1;
            ((uint64_t*)out2[pos_dec])[idx] = aux2;
        } else {
            ((uint32_t*)out1[pos_dec])[idx] = (uint32_t)aux1;
            ((uint32_t*)out2[pos_dec])[idx] = (uint32_t)aux2;
        }
    } else {
        if (ISU64(primeid)) {
            ((uint64_t*)sout1[primeid - C_.L])[idx] = aux1;
            ((uint64_t*)sout2[primeid - C_.L])[idx] = aux2;
        } else {
            ((uint32_t*)sout1[primeid - C_.L])[idx] = (uint32_t)aux1;
            ((uint32_t*)sout2[primeid - C_.L])[idx] = (uint32_t)aux2;
        }
    }
}


// Optionally __restrict__-qualify the coop4 hoisted kernel's pointer-table params so ptxas may
// hoist the j-invariant din pointer chases out of the rotation loop at its own register budget.
// All tables address disjoint device buffers. Compile-time switch, default 0.
#ifndef FIDESLIB_PTR_RESTRICT
#define FIDESLIB_PTR_RESTRICT 0
#endif
#if FIDESLIB_PTR_RESTRICT
#define FLPT3 void* const* const* __restrict__
#define FLPT2 void* const* __restrict__
#else
#define FLPT3 void***
#define FLPT2 void**
#endif
// Launch-bounds pin for the 4-slot cooperative regen kernel. 0 = no pin (ptxas natural demand);
// a tier does not transfer across kernels or GPUs, so sweep it rather than inherit one.
#ifndef FIDESLIB_DOT_REGEN4_MINCTA
#define FIDESLIB_DOT_REGEN4_MINCTA 0
#endif
// i-pair unroll in the GSTEP-specialized LT dot (doubles the loads in flight per warp).
#ifndef FIDESLIB_LT_I2
#define FIDESLIB_LT_I2 1
#endif
// i-QUAD unroll: doubles the batch again over I2 at ~+6 registers. Compile-time switch, default 0;
// takes precedence over I2 when both are set.
#ifndef FIDESLIB_LT_I4
#define FIDESLIB_LT_I4 0
#endif
// Evict-first (__ldcs) loads on the two STREAMED-ONCE inputs — kskb (42 % L2 hit) and the
// LT plaintexts (5 % hit) — so they stop evicting the n×-reused digit reads.
#ifndef FIDESLIB_KSK_LDCS
#define FIDESLIB_KSK_LDCS 1
#endif
#if FIDESLIB_KSK_LDCS
#define FIDESLIB_STREAM_LD(p) __ldcs(p)
#else
#define FIDESLIB_STREAM_LD(p) (*(p))
#endif
// Optionally persist the digit reads in SHARED MEMORY across the hoisted4 rotation loop (din is
// j-independent, yet is re-read from global for every rotation). Per-thread slice = num_d x 16 B,
// no __syncthreads (each thread reads only its own slice); num_d > 8 falls back to streaming at
// the launcher. Compile-time switch, default 0.
#ifndef FIDESLIB_DIN_SMEM
#define FIDESLIB_DIN_SMEM 0
#endif
// Optional evict-first (__stcs) stores on the hoisted4 automorphism outputs: the outputs are read
// once later, while the digit streams are re-read n times per launch and should stay in L2.
// Compile-time switch, default 0.
#ifndef FIDESLIB_HOISTED_STCS
#define FIDESLIB_HOISTED_STCS 0
#endif
#if FIDESLIB_HOISTED_STCS
#define FIDESLIB_HOISTED_ST(p, v) __stcs((p), (v))
#else
#define FIDESLIB_HOISTED_ST(p, v) (*(p) = (v))
#endif
// LAZY u64 ACCUMULATION in the coop4 dot kernels: d*(ks mod p) == d*ks (mod p), so accumulate
// the raw product (u64)d*ks per digit and reduce ONCE after the digit loop with modreduce_lazy
// (exact for any a < 2^64). Overflow-safe: d < 2^28, ks < 2^32 => each product < 2^60, so the
// u64 accumulator holds >= 15 digits. Bit-exact by congruence.
#ifndef FIDESLIB_LAZY_DOT_ACC
#define FIDESLIB_LAZY_DOT_ACC 1
#endif
// Second tier: evict-first on the LT dot's ciphertext reads too (each read exactly once).
#ifndef FIDESLIB_LT_CTIN_LDCS
#define FIDESLIB_LT_CTIN_LDCS 1
#endif
#if FIDESLIB_LT_CTIN_LDCS
#define FIDESLIB_STREAM_LD2(p) __ldcs(p)
#else
#define FIDESLIB_STREAM_LD2(p) (*(p))
#endif

/* Exact v % p from the SAME reciprocal the spec's rejection threshold already needs — the
 * naive `v % p` on a runtime divisor is a ~25-instruction sequence, which at 16 coefficients
 * per (rotation, digit) would rival the ChaCha block itself. m = floor(2^32/p) gives
 * q_hat in {floor(v/p)-1, floor(v/p)} => one conditional subtract. Bit-identical remainder. */
__device__ __forceinline__ uint32_t modByRecip(const uint32_t v, const uint32_t p, const uint32_t m) {
    const uint32_t r = v - __umulhi(v, m) * p;
    return r >= p ? r - p : r;
}

// Helpers for the 4-slot cooperative regen kernels (fusedDotKSKRegen4_ below and
// hoistedRotateDotKSKRegen4_ further down — see the latter's header comment for the design).
__device__ __forceinline__ uint32_t sel4(const uint32_t k, const uint32_t v0, const uint32_t v1, const uint32_t v2,
                                         const uint32_t v3) {
    uint32_t r = v0;
    r = (k == 1) ? v1 : r;
    r = (k == 2) ? v2 : r;
    r = (k == 3) ? v3 : r;
    return r;
}

/* One cooperative ChaCha12 block across a 4-lane group (width-4 shuffles, segment-relative
 * lane ids). Lane k holds column k = words {k, k+4, k+8, k+12}; outputs are the
 * feedforwarded column, still in COLUMN order. */
__device__ __forceinline__ void chachaCoop4(const uint32_t gmask, const uint32_t k, const uint32_t key0,
                                            const uint32_t key4, const uint32_t ctr, const uint32_t digit,
                                            const uint32_t modulus, uint32_t& oa, uint32_t& ob, uint32_t& oc,
                                            uint32_t& od) {
    const uint32_t A = sel4(k, 0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u);
    const uint32_t D = sel4(k, ctr, digit, modulus, kskexpand::kDomainSep);
    uint32_t a = A, b = key0, c = key4, d = D;
#pragma unroll
    for (int r = 0; r < kskexpand::kRounds; r += 2) {
        a += b; d ^= a; d = __funnelshift_l(d, d, 16);
        c += d; b ^= c; b = __funnelshift_l(b, b, 12);
        a += b; d ^= a; d = __funnelshift_l(d, d, 8);
        c += d; b ^= c; b = __funnelshift_l(b, b, 7);
        b = __shfl_sync(gmask, b, (k + 1) & 3u, 4);
        c = __shfl_sync(gmask, c, (k + 2) & 3u, 4);
        d = __shfl_sync(gmask, d, (k + 3) & 3u, 4);
        a += b; d ^= a; d = __funnelshift_l(d, d, 16);
        c += d; b ^= c; b = __funnelshift_l(b, b, 12);
        a += b; d ^= a; d = __funnelshift_l(d, d, 8);
        c += d; b ^= c; b = __funnelshift_l(b, b, 7);
        b = __shfl_sync(gmask, b, (k + 3) & 3u, 4);
        c = __shfl_sync(gmask, c, (k + 2) & 3u, 4);
        d = __shfl_sync(gmask, d, (k + 1) & 3u, 4);
    }
    oa = a + A;
    ob = b + key0;
    oc = c + key4;
    od = d + D;
}


// 4-slot cooperative port of fusedDotKSKRegen_ — same shape as hoistedRotateDotKSKRegen4_
// below (see its header comment for the design; helpers sel4/chachaCoop4 are defined there,
// above the launcher). No rotation loop, no automorphism: stores stay coalesced uint4.
template <int KSK_BITS>
__global__ void fusedDotKSKRegen4_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d,
                                   int id, int num_special, int init, KskSeedWords aseed, const uint32_t n16) {
    constexpr int SLOTS = 4;
    const uint32_t gtid = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);
    const int base = (int)(gtid * SLOTS);
    const uint32_t b0 = gtid >> 2;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane & 3u;
    const uint32_t gmask = 0xFu << (lane & ~3u);
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];
    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    const uint32_t pval = (uint32_t)C_.primes[primeid];
    const uint32_t recip = 0xFFFFFFFFu / pval;
    const uint32_t m_p = recip * pval;
    const uint32_t key0 = sel4(k, aseed.k[0], aseed.k[1], aseed.k[2], aseed.k[3]);
    const uint32_t key4 = sel4(k, aseed.k[4], aseed.k[5], aseed.k[6], aseed.k[7]);

#if FIDESLIB_LAZY_DOT_ACC
    uint64_t a1[SLOTS], a2[SLOTS];
#else
    uint32_t a1[SLOTS], a2[SLOTS];
#endif
#pragma unroll
    for (int w = 0; w < SLOTS; ++w) {
        a1[w] = 0;
        a2[w] = 0;
    }

    for (int i = 0; i < num_d; ++i) {
        const bool decomp = (i == primeid_digit);
        const int pos = C_.pos_in_digit[i][primeid];
        const int p = decomp ? pos_dec : pos;
        const uint32_t* inp = (const uint32_t*)digits[i + decomp * 3 * C_.dnum][p] + base;
        const void* kskbp = digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][p];

        const uint4 iv = FIDESLIB_STREAM_LD((const uint4*)inp);  // single-use in the fused (n=1) kernel
        uint32_t kb[5];
        if constexpr (KSK_BITS == 28) {
            const uint32_t bit0 = (uint32_t)base * KSK_BITS;
#pragma unroll
            for (int t = 0; t < 5; ++t)
                kb[t] = FIDESLIB_STREAM_LD((const uint32_t*)kskbp + (bit0 >> 5) + t);
        }

        uint32_t ca, cb, cc, cd;
        chachaCoop4(gmask, k, key0, key4, b0, (uint32_t)i, pval, ca, cb, cc, cd);
        uint32_t need = ((ca >= m_p) ? 1u : 0u) | ((cb >= m_p) ? 2u : 0u) | ((cc >= m_p) ? 4u : 0u) |
                        ((cd >= m_p) ? 8u : 0u);
        uint32_t gneed = need;
        gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
        gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
        if (gneed) {
            for (uint32_t t = 1; t < (uint32_t)kskexpand::kTMax; ++t) {
                uint32_t ea, eb, ec, ed;
                chachaCoop4(gmask, k, key0, key4, b0 + t * n16, (uint32_t)i, pval, ea, eb, ec, ed);
                if (need & 1u) {
                    ca = ea;
                    if (ea < m_p)
                        need &= ~1u;
                }
                if (need & 2u) {
                    cb = eb;
                    if (eb < m_p)
                        need &= ~2u;
                }
                if (need & 4u) {
                    cc = ec;
                    if (ec < m_p)
                        need &= ~4u;
                }
                if (need & 8u) {
                    cd = ed;
                    if (ed < m_p)
                        need &= ~8u;
                }
                gneed = need;
                gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
                gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
                if (!gneed)
                    break;
            }
        }

        uint32_t ks[SLOTS] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint32_t pub = sel4((k - (uint32_t)s) & 3u, ca, cb, cc, cd);
            const uint32_t rcv = __shfl_sync(gmask, pub, (k + (uint32_t)s) & 3u, 4);
            const uint32_t r = (k + (uint32_t)s) & 3u;
            ks[0] = (r == 0) ? rcv : ks[0];
            ks[1] = (r == 1) ? rcv : ks[1];
            ks[2] = (r == 2) ? rcv : ks[2];
            ks[3] = (r == 3) ? rcv : ks[3];
        }

        const uint32_t in4[4] = {iv.x, iv.y, iv.z, iv.w};
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
#if !FIDESLIB_LAZY_DOT_ACC
            const uint32_t kska = modByRecip(ks[w], pval, recip);
#endif
            uint32_t kskb;
            if constexpr (KSK_BITS == 28) {
                if (gtid & 1u) {
                    const uint32_t lo = 16u + (uint32_t)(w * KSK_BITS);
                    kskb = __funnelshift_r(kb[lo >> 5], kb[(lo >> 5) + 1], lo & 31) & ((1u << KSK_BITS) - 1u);
                } else {
                    const uint32_t lo = (uint32_t)(w * KSK_BITS);
                    kskb = __funnelshift_r(kb[lo >> 5], kb[(lo >> 5) + 1], lo & 31) & ((1u << KSK_BITS) - 1u);
                }
            } else if constexpr (KSK_BITS)
                kskb = kskUnpack(kskbp, (uint32_t)(base + w), KSK_BITS, (1u << KSK_BITS) - 1u);
            else
                kskb = ((const uint32_t*)kskbp)[base + w];
#if FIDESLIB_LAZY_DOT_ACC
            // Wide-prime guard: the raw keystream word is ~2^32 regardless of p, so in*ks is 2^62
            // at a 30-bit prime and dnum=7 of them wrap u64. Reducing ks first keeps every product
            // < 2^60 and the lazy sum < 2^63. Narrow primes keep the reduce-free fast path.
            a1[w] += (uint64_t)in4[w] *
                     (C_.prime_bits[primeid] > 28 ? modByRecip(ks[w], pval, recip) : ks[w]);
            a2[w] += (uint64_t)in4[w] * kskb;
#else
            a1[w] = modadd(a1[w], modmult<ALGO_BARRETT>(in4[w], kska, primeid), primeid);
            a2[w] = modadd(a2[w], modmult<ALGO_BARRETT>(in4[w], kskb, primeid), primeid);
#endif
        }
    }

    uint32_t* o1 = (primeid < C_.L) ? (uint32_t*)out1[pos_dec] : (uint32_t*)sout1[primeid - C_.L];
    uint32_t* o2 = (primeid < C_.L) ? (uint32_t*)out2[pos_dec] : (uint32_t*)sout2[primeid - C_.L];
#if FIDESLIB_LAZY_DOT_ACC
    *(uint4*)(o1 + base) = make_uint4(modreduce_lazy(a1[0], primeid), modreduce_lazy(a1[1], primeid),
                                      modreduce_lazy(a1[2], primeid), modreduce_lazy(a1[3], primeid));
    *(uint4*)(o2 + base) = make_uint4(modreduce_lazy(a2[0], primeid), modreduce_lazy(a2[1], primeid),
                                      modreduce_lazy(a2[2], primeid), modreduce_lazy(a2[3], primeid));
#else
    *(uint4*)(o1 + base) = make_uint4(a1[0], a1[1], a1[2], a1[3]);
    *(uint4*)(o2 + base) = make_uint4(a2[0], a2[1], a2[2], a2[3]);
#endif
}

// Same-TU launcher (see the .cuh note: cross-TU template-kernel launches hit
// 'invalid device function' — the launch must live in the defining TU). Supported packed
// widths are the instantiated set {27, 28}; kskPackBitsPolicy only arms those.
// ===================== u64 (KSKB, SPEC v2) regen kernels =====================
// The NATIVE_SIZE=64 twins of the register-shape regen arms. One KSKB ChaCha block serves
// 8 coefficients (2 keystream words each), so SLOTS = 8 and each thread owns the 8
// consecutive coefficients its block generates — the same no-barrier amortization as the
// u32 SLOTS=16 kernels. u64 chains never pack (kskPackBitsPolicy is u32-only), so there is
// no BITS template; kskb streams dense u64. Reduction of the accepted 64-bit word pair is
// a plain `% p` (no 32-bit recip trick at this width; Barrett handles the dot products).
// Launch-bounds min-blocks tier for the u64 (KSKB) regen dot kernels: 16 accumulator pairs plus a
// 16-word keystream are live by construction, so capping registers buys rematerialization, not warps.
#ifndef FIDESLIB_DOT_REGEN_MINCTA
#define FIDESLIB_DOT_REGEN_MINCTA 3
#endif

__global__ void __launch_bounds__(128, FIDESLIB_DOT_REGEN_MINCTA)
    fusedDotKSKRegen64_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id,
                        int num_special, int init, KskSeedWords aseed, const uint32_t n8) {
    constexpr int SLOTS = 8;
    const uint32_t b0 = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);  // == slot>>3
    const int base = (int)(b0 << 3);
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];
    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    const uint64_t pval = C_.primes[primeid];
    const uint64_t m_p = (0xFFFFFFFFFFFFFFFFull / pval) * pval;
    const uint32_t w13 = ((uint32_t)(pval >> 32) << 4);  // digit tag ORed in per digit below
    const uint32_t p_lo = (uint32_t)pval;

    uint64_t a1[SLOTS], a2[SLOTS];
#pragma unroll
    for (int w = 0; w < SLOTS; ++w) {
        a1[w] = 0;
        a2[w] = 0;
    }

    for (int i = 0; i < num_d; ++i) {
        const bool decomp = (i == primeid_digit);
        const int pos = C_.pos_in_digit[i][primeid];
        const int pp = decomp ? pos_dec : pos;
        const uint64_t* inp = (const uint64_t*)digits[i + decomp * 3 * C_.dnum][pp] + base;
        const uint64_t* kskbp = (const uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][pp] + base;

        uint32_t ks[16];
        kskexpand::chacha_block_tail(aseed.k, b0, w13 | ((uint32_t)i & 0xFu), p_lo, kskexpand::kDomainSep64, ks);
        uint64_t v[SLOTS];
        uint32_t need = 0;
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
            v[w] = (uint64_t)ks[2 * w] | ((uint64_t)ks[2 * w + 1] << 32);
            need |= (v[w] >= m_p) ? (1u << w) : 0u;
        }
        if (need) {
            for (uint32_t tt = 1; tt < (uint32_t)kskexpand::kTMax; ++tt) {
                uint32_t es[16];
                kskexpand::chacha_block_tail(aseed.k, b0 + tt * n8, w13 | ((uint32_t)i & 0xFu), p_lo,
                                             kskexpand::kDomainSep64, es);
#pragma unroll
                for (int w = 0; w < SLOTS; ++w)
                    if (need & (1u << w)) {
                        v[w] = (uint64_t)es[2 * w] | ((uint64_t)es[2 * w + 1] << 32);
                        if (v[w] < m_p)
                            need &= ~(1u << w);
                    }
                if (!need)
                    break;
            }
        }
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
            const uint64_t kska = v[w] % pval;
            const uint64_t din = inp[w];
            a1[w] = modadd(a1[w], modmult<ALGO_BARRETT>(din, kska, primeid), primeid);
            a2[w] = modadd(a2[w], modmult<ALGO_BARRETT>(din, kskbp[w], primeid), primeid);
        }
    }

#pragma unroll
    for (int w = 0; w < SLOTS; ++w) {
        if (primeid < C_.L) {
            ((uint64_t*)out1[pos_dec])[base + w] = a1[w];
            ((uint64_t*)out2[pos_dec])[base + w] = a2[w];
        } else {
            ((uint64_t*)sout1[primeid - C_.L])[base + w] = a1[w];
            ((uint64_t*)sout2[primeid - C_.L])[base + w] = a2[w];
        }
    }
}

__global__ void __launch_bounds__(128, FIDESLIB_DOT_REGEN_MINCTA)
    hoistedRotateDotKSKRegen64_(void*** din1, void** c0, void*** out1, void*** sout1, void*** out2, void*** sout2,
                                const int n, const int* indexes, void*** digits, int num_d, int id, int num_special,
                                int init, void** sc0, bool c0_modup, const uint32_t* __restrict__ seeds,
                                const uint32_t n8) {
    constexpr int SLOTS = 8;
    const uint32_t b0 = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);
    const int base = (int)(b0 << 3);
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];
    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    const uint64_t pval = C_.primes[primeid];
    const uint64_t m_p = (0xFFFFFFFFFFFFFFFFull / pval) * pval;
    const uint32_t whi = ((uint32_t)(pval >> 32) << 4);
    const uint32_t p_lo = (uint32_t)pval;

    // c0 folded into aux2's initial value, exactly like the u32 regen kernel / the dense
    // generic arm (shoup by P only on the non-modup mainline path).
    const uint64_t* c0p = nullptr;
    if (c0_modup || primeid < C_.L)
        c0p = primeid < C_.L ? (const uint64_t*)c0[pos_dec] : (const uint64_t*)sc0[primeid - C_.L];
    const bool c0_shoup = (!c0_modup && primeid < C_.L);

    for (int j = 0; j < n; ++j) {
        const int offset = j * 3 * 2 * C_.dnum;
        const uint32_t* key = seeds + j * 8;

        uint64_t aux1[SLOTS], aux2[SLOTS];
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
            aux1[w] = 0;
            uint64_t in2 = c0p ? c0p[base + w] : 0ull;
            if (c0p && c0_shoup)
                in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
            aux2[w] = in2;
        }

        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            const int pp = decomp ? pos_dec : pos;
            const uint64_t* dinp = (const uint64_t*)din1[i + decomp * 3 * C_.dnum][pp] + base;
            const uint64_t* kskbp =
                (const uint64_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][pp] + base;

            uint32_t ks[16];
            kskexpand::chacha_block_tail(key, b0, whi | ((uint32_t)i & 0xFu), p_lo, kskexpand::kDomainSep64, ks);
            uint64_t v[SLOTS];
            uint32_t need = 0;
#pragma unroll
            for (int w = 0; w < SLOTS; ++w) {
                v[w] = (uint64_t)ks[2 * w] | ((uint64_t)ks[2 * w + 1] << 32);
                need |= (v[w] >= m_p) ? (1u << w) : 0u;
            }
            if (need) {
                for (uint32_t tt = 1; tt < (uint32_t)kskexpand::kTMax; ++tt) {
                    uint32_t es[16];
                    kskexpand::chacha_block_tail(key, b0 + tt * n8, whi | ((uint32_t)i & 0xFu), p_lo,
                                                 kskexpand::kDomainSep64, es);
#pragma unroll
                    for (int w = 0; w < SLOTS; ++w)
                        if (need & (1u << w)) {
                            v[w] = (uint64_t)es[2 * w] | ((uint64_t)es[2 * w + 1] << 32);
                            if (v[w] < m_p)
                                need &= ~(1u << w);
                        }
                    if (!need)
                        break;
                }
            }
#pragma unroll
            for (int w = 0; w < SLOTS; ++w) {
                const uint64_t kska = v[w] % pval;
                const uint64_t din = dinp[w];
                aux1[w] = modadd(aux1[w], modmult<ALGO_BARRETT>(din, kska, primeid), primeid);
                aux2[w] = modadd(aux2[w], modmult<ALGO_BARRETT>(din, kskbp[w], primeid), primeid);
            }
        }

        const int rot_index = indexes[j];
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
            const uint32_t out_idx = automorph_slot(C_.logN, rot_index, (uint32_t)(base + w));
            if (primeid < C_.L) {
                ((uint64_t*)out1[j][pos_dec])[out_idx] = aux1[w];
                ((uint64_t*)out2[j][pos_dec])[out_idx] = aux2[w];
            } else {
                ((uint64_t*)sout1[j][primeid - C_.L])[out_idx] = aux1[w];
                ((uint64_t*)sout2[j][primeid - C_.L])[out_idx] = aux2[w];
            }
        }
    }
}

void launchFusedDotKSK_2(dim3 grid, dim3 block, cudaStream_t stream, void** out1, void** sout1, void** out2,
                         void** sout2, void*** digits, int num_d, int id, int num_special, int init,
                         int ksk_pack_bits, const uint32_t* a_seed, uint32_t n16, int regen_shape,
                         int chain_type) {
    KskSeedWords sw{};
    if (a_seed)
        for (int i = 0; i < 8; ++i)
            sw.k[i] = a_seed[i];
    if (a_seed == nullptr)
        regen_shape = 0;
    if (regen_shape != 0 && chain_type != 0) {
        // u64 (KSKB) chain: the single register-shape arm; one thread per 8 coefficients.
        if (grid.x % 8u != 0u)
            throw std::runtime_error("launchFusedDotKSK_2: u64 regen needs grid.x % 8 == 0");
        grid.x /= 8u;
        fusedDotKSKRegen64_<<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id, num_special,
                                                        init, sw, n16 * 2u /* == N>>3 */);
        return;
    }
    // Shape 1 (register regen) is the 4-slot cooperative kernel: each thread owns 4 coefficients, so
    // grid.x shrinks by 4 — done here rather than at the call sites, which all pass the N/block.x grid.
    const bool coop4 = regen_shape == 1;
    if (coop4) {
        if (grid.x % 16u != 0u)
            throw std::runtime_error("launchFusedDotKSK_2: regen shape 1 needs grid.x % 16 == 0");
        grid.x /= 4u;
    }
    const size_t smem = regen_shape == 2 ? (size_t)num_d * 128 * sizeof(uint32_t) : 0;  // stage-A tile only
/* Macro, not a helper template, for one reason: BITS must reach the kernel as a COMPILE-TIME
 * template argument (a runtime-width variant costs registers and occupancy), so the arms have
 * to be selected by a switch over instantiations. The macro keeps the argument list — 15+
 * parameters — written once per launcher instead of once per (width x arm) pair. */
#define FIDESLIB_FUSED_DOT_ARM(BITS)                                                                             \
    if (coop4)                                                                                                   \
        fusedDotKSKRegen4_<BITS><<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id,         \
                                                             num_special, init, sw, n16);                        \
    else if (regen_shape == 2)                                                                                   \
        fusedDotKSK_2_<BITS, true><<<grid, block, smem, stream>>>(out1, sout1, out2, sout2, digits, num_d, id,    \
                                                                  num_special, init, sw, n16);                   \
    else                                                                                                         \
        fusedDotKSK_2_<BITS, false><<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id,      \
                                                                num_special, init, sw, n16);
    switch (ksk_pack_bits) {
        case 0:
            FIDESLIB_FUSED_DOT_ARM(0)
            break;
        case 27:
            FIDESLIB_FUSED_DOT_ARM(27)
            break;
        case 28:
            FIDESLIB_FUSED_DOT_ARM(28)
            break;
        default:
            throw std::runtime_error("launchFusedDotKSK_2: unsupported ksk_pack_bits " +
                                     std::to_string(ksk_pack_bits));
    }
#undef FIDESLIB_FUSED_DOT_ARM
}

constexpr bool PRINT = false;

// Same register-occupancy and compile-time KSK_BITS treatment as fusedDotKSK_2_ above.
template <int KSK_BITS>
__global__ void __launch_bounds__(128, KSK_BITS ? FIDESLIB_DOT_MINCTA_PACKED : 12)
    hoistedRotateDotKSK_2_(void*** din1, void** c0, void*** out1, void*** sout1, void*** out2,
                           void*** sout2, const int n, const int* indexes, void*** digits, int num_d,
                           int id, int num_special, int init, void** sc0, bool c0_modup) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    extern __shared__ char buffer[];

    if (C_.type == 0) {
        // u32 fast path — see fusedDotKSK_2_ above. The shared digit
        // cache is reinterpreted as u32 (uses half the allocation; layout self-consistent
        // within this arm); residues and store order identical to the generic arm => bit-exact.
        uint32_t* in1s = ((uint32_t*)buffer) + num_d * threadIdx.x;
        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            in1s[i] = ((uint32_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }
        uint32_t in2 = 0;
        if (c0_modup || primeid < C_.L) {
            in2 = primeid < C_.L ? ((uint32_t*)c0[pos_dec])[idx] : ((uint32_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>(in2, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
        }
        for (int j = 0; j < n; ++j) {
            const int offset = j * 3 * 2 * C_.dnum;
            uint32_t aux1, aux2;
            for (int i = 0; i < num_d; ++i) {
                const bool decomp = (i == primeid_digit);
                const int pos = C_.pos_in_digit[i][primeid];
                const int p = decomp ? pos_dec : pos;
                uint32_t kska, kskb;
                if constexpr (KSK_BITS) {
                    kska = kskUnpack(digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                     (1u << KSK_BITS) - 1u);
                    kskb = kskUnpack(digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                     (1u << KSK_BITS) - 1u);
                } else {
                    kska = ((uint32_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
                    kskb = ((uint32_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
                }
                const uint32_t add1 = modmult<ALGO_BARRETT>(in1s[i], kska, primeid);
                const uint32_t add2 = modmult<ALGO_BARRETT>(in1s[i], kskb, primeid);
                if (i == 0) {
                    aux1 = add1;
                    aux2 = (c0_modup || primeid < C_.L) ? modadd(in2, add2, primeid) : add2;
                } else {
                    aux1 = modadd(aux1, add1, primeid);
                    aux2 = modadd(aux2, add2, primeid);
                }
            }
            const uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
            if (primeid < C_.L) {
                ((uint32_t*)out1[j][pos_dec])[out_idx] = aux1;
                ((uint32_t*)out2[j][pos_dec])[out_idx] = aux2;
            } else {
                ((uint32_t*)sout1[j][primeid - C_.L])[out_idx] = aux1;
                ((uint32_t*)sout2[j][primeid - C_.L])[out_idx] = aux2;
            }
        }
        return;
    }

    uint64_t* in1 = ((uint64_t*)buffer) + num_d * threadIdx.x;

    for (int i = 0; i < num_d; ++i) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];
        if (ISU64(primeid)) {
            in1[i] = ((uint64_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        } else {
            in1[i] = ((uint32_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i]);
    }

    uint64_t in2 = 0;

    if (c0_modup || primeid < C_.L) {
        if (ISU64(primeid)) {
            in2 = primeid < C_.L ? ((uint64_t*)c0[pos_dec])[idx] : ((uint64_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
        } else {
            in2 = primeid < C_.L ? ((uint32_t*)c0[pos_dec])[idx] : ((uint32_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>((uint32_t)in2, (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        }
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {
        uint64_t aux1, aux2;
        int offset = j * 3 * 2 * C_.dnum;

        for (int i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            uint64_t kska, kskb;
            if (ISU64(primeid)) {
                kska = ((uint64_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
                kskb =
                    ((uint64_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
            } else {
                kska = ((uint32_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
                kskb =
                    ((uint32_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
            }
            uint64_t add1 = modmult<ALGO_BARRETT>(in1[i], kska, primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("kska %d : %lu\n", i, kska);

            if (PRINT && idx == 0 && blky == 6)
                printf("add1 %d : %lu\n", i, add1);
            uint64_t add2 = modmult<ALGO_BARRETT>(in1[i], kskb, primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("kskb %d : %lu\n", i, kskb);
            if (PRINT && idx == 0 && blky == 6)
                printf("add2 %d : %lu\n", i, add2);

            if (i == 0) {
                aux1 = add1;
                if (c0_modup || primeid < C_.L) {
                    aux2 = modadd(in2, add2, primeid);
                } else {
                    aux2 = add2;
                }
            } else {
                aux1 = modadd(aux1, add1, primeid);
                aux2 = modadd(aux2, add2, primeid);
            }
            if (PRINT && idx == 0 && blky == 6)
                printf("aux1 %d : %lu\n", i, aux1);
            if (PRINT && idx == 0 && blky == 6)
                printf("aux2 %d : %lu\n", i, aux2);
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux1);
        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux2);

        uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
        //uint32_t out_idx = idx;
        if (primeid < C_.L) {
            if (ISU64(primeid)) {
                ((uint64_t*)out1[j][pos_dec])[out_idx] = aux1;
                ((uint64_t*)out2[j][pos_dec])[out_idx] = aux2;
            } else {
                ((uint32_t*)out1[j][pos_dec])[out_idx] = (uint32_t)aux1;
                ((uint32_t*)out2[j][pos_dec])[out_idx] = (uint32_t)aux2;
            }
        } else {
            if (ISU64(primeid)) {
                ((uint64_t*)sout1[j][primeid - C_.L])[out_idx] = aux1;
                ((uint64_t*)sout2[j][primeid - C_.L])[out_idx] = aux2;
            } else {
                ((uint32_t*)sout1[j][primeid - C_.L])[out_idx] = (uint32_t)aux1;
                ((uint32_t*)sout2[j][primeid - C_.L])[out_idx] = (uint32_t)aux2;
            }
        }
    }
}


// ============================ 4-slot cooperative regen ============================
// A one-ChaCha-block-per-thread regen kernel (SLOTS = 16) is forced to ~168 registers and is
// latency-bound. Here each thread owns FOUR consecutive slots and a 4-lane group computes each
// ChaCha block cooperatively — column-per-lane, the diagonal rounds via 3 __shfl rotations in/out
// (bit-exact against the spec reference). Per-thread live state collapses (8 accumulator regs
// instead of 32, 4 keystream words instead of 16), buying the resident warps that hide the
// load latency at ~1.4x the ChaCha ALU per slot.
// Escalation and modByRecip are word-order-independent, so the keystream stays in COLUMN
// order (lane k holds words {k,k+4,k+8,k+12}) through the whole escalation loop and is
// lane-transposed to slot order exactly once per accepted block.
// All shuffles use the 4-lane group mask; group lanes never diverge from each other (the
// escalation predicate is group-ORed first), which is what makes the masks legal.

template <int KSK_BITS>
__global__ void
#if FIDESLIB_DOT_REGEN4_MINCTA
    __launch_bounds__(128, FIDESLIB_DOT_REGEN4_MINCTA)
#endif
        hoistedRotateDotKSKRegen4_(FLPT3 din1, FLPT2 c0, FLPT3 out1, FLPT3 sout1, FLPT3 out2, FLPT3 sout2,
                                   const int n, const int* indexes, FLPT3 digits, int num_d, int id,
                                   int num_special, int init, FLPT2 sc0, bool c0_modup,
                                   const uint32_t* __restrict__ seeds, const uint32_t n16) {
    constexpr int SLOTS = 4;
    const uint32_t gtid = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);
    const int base = (int)(gtid * SLOTS);      // first of this thread's 4 consecutive slots
    const uint32_t b0 = gtid >> 2;             // the GROUP's ChaCha block (== slot >> 4, group-uniform)
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane & 3u;              // column index within the 4-lane group
    const uint32_t gmask = 0xFu << (lane & ~3u);
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];
    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    const uint32_t pval = (uint32_t)C_.primes[primeid];
    const uint32_t recip = 0xFFFFFFFFu / pval;
    const uint32_t m_p = recip * pval;

    const uint32_t* c0p = nullptr;
    if (c0_modup || primeid < C_.L)
        c0p = primeid < C_.L ? (const uint32_t*)c0[pos_dec] : (const uint32_t*)sc0[primeid - C_.L];
    const bool c0_shoup = (!c0_modup && primeid < C_.L);

#if FIDESLIB_DIN_SMEM
    // One global read of this thread's digit slice, reused for every rotation j.
    extern __shared__ uint4 sdin[];  // [num_d][blockDim.x]
    for (int i = 0; i < num_d; ++i) {
        const bool decomp = (i == primeid_digit);
        const int pos = C_.pos_in_digit[i][primeid];
        const int p = decomp ? pos_dec : pos;
        sdin[i * blockDim.x + threadIdx.x] = *(const uint4*)((const uint32_t*)din1[i + decomp * 3 * C_.dnum][p] + base);
    }
#endif

    for (int j = 0; j < n; ++j) {
        const int offset = j * 3 * 2 * C_.dnum;
        const uint32_t* key = seeds + j * 8;
        const uint32_t key0 = key[k];
        const uint32_t key4 = key[4 + k];

#if FIDESLIB_LAZY_DOT_ACC
        uint64_t aux1[SLOTS], aux2[SLOTS];
#else
        uint32_t aux1[SLOTS], aux2[SLOTS];
#endif
        if (c0p) {
            const uint4 cv = *(const uint4*)(c0p + base);
            const uint32_t c4[4] = {cv.x, cv.y, cv.z, cv.w};
#pragma unroll
            for (int w = 0; w < SLOTS; ++w) {
                aux1[w] = 0;
                aux2[w] = c0_shoup ? modmult<ALGO_SHOUP>(c4[w], (uint32_t)C_.P[primeid], primeid,
                                                        (uint32_t)C_.P_shoup[primeid])
                                   : c4[w];
            }
        } else {
#pragma unroll
            for (int w = 0; w < SLOTS; ++w) {
                aux1[w] = 0;
                aux2[w] = 0;
            }
        }

        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            const int p = decomp ? pos_dec : pos;
            const void* kskbp = digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][p];

            // Issue both input streams before the ChaCha so the regen ALU hides their latency
            // (the kbstage2 lesson, kept by construction here). A software pipeline of the
            // NEXT digit's loads was BUILT and lost +0.51 ms (0/3): +13 regs -> 9 -> 7
            // blocks/SM, and occupancy is the binding resource at this shape.
#if FIDESLIB_DIN_SMEM
            const uint4 dv = sdin[i * blockDim.x + threadIdx.x];
#else
            const uint4 dv = *(const uint4*)((const uint32_t*)din1[i + decomp * 3 * C_.dnum][p] + base);
#endif
            uint32_t kb[5];
            if constexpr (KSK_BITS == 28) {
                const uint32_t bit0 = (uint32_t)base * KSK_BITS;  // == 112*gtid; &31 is 0 or 16
#pragma unroll
                for (int t = 0; t < 5; ++t)
                    kb[t] = FIDESLIB_STREAM_LD((const uint32_t*)kskbp + (bit0 >> 5) + t);
            }

            // --- cooperative regen, COLUMN order end to end (transpose once at the end) ---
            uint32_t ca, cb, cc, cd;
            chachaCoop4(gmask, k, key0, key4, b0, (uint32_t)i, pval, ca, cb, cc, cd);
            uint32_t need = ((ca >= m_p) ? 1u : 0u) | ((cb >= m_p) ? 2u : 0u) | ((cc >= m_p) ? 4u : 0u) |
                            ((cd >= m_p) ? 8u : 0u);
            uint32_t gneed = need;
            gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
            gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
            if (gneed) {
                for (uint32_t t = 1; t < (uint32_t)kskexpand::kTMax; ++t) {
                    uint32_t ea, eb, ec, ed;
                    chachaCoop4(gmask, k, key0, key4, b0 + t * n16, (uint32_t)i, pval, ea, eb, ec, ed);
                    if (need & 1u) {
                        ca = ea;
                        if (ea < m_p)
                            need &= ~1u;
                    }
                    if (need & 2u) {
                        cb = eb;
                        if (eb < m_p)
                            need &= ~2u;
                    }
                    if (need & 4u) {
                        cc = ec;
                        if (ec < m_p)
                            need &= ~4u;
                    }
                    if (need & 8u) {
                        cd = ed;
                        if (ed < m_p)
                            need &= ~8u;
                    }
                    gneed = need;
                    gneed |= __shfl_xor_sync(gmask, gneed, 1, 4);
                    gneed |= __shfl_xor_sync(gmask, gneed, 2, 4);
                    if (!gneed)
                        break;
                }
            }

            // --- lane transpose to slot order: ks[r] = word 4k+r = lane r's column word k.
            // Step s: publish column word (k−s)&3, read from group lane (k+s)&3, store ks[(k+s)&3].
            uint32_t ks[SLOTS] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t pub = sel4((k - (uint32_t)s) & 3u, ca, cb, cc, cd);
                const uint32_t rcv = __shfl_sync(gmask, pub, (k + (uint32_t)s) & 3u, 4);
                const uint32_t r = (k + (uint32_t)s) & 3u;
                ks[0] = (r == 0) ? rcv : ks[0];
                ks[1] = (r == 1) ? rcv : ks[1];
                ks[2] = (r == 2) ? rcv : ks[2];
                ks[3] = (r == 3) ? rcv : ks[3];
            }

            // --- the dot on this thread's 4 slots ---
            const uint32_t d4[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
            for (int w = 0; w < SLOTS; ++w) {
#if !FIDESLIB_LAZY_DOT_ACC
                const uint32_t kska = modByRecip(ks[w], pval, recip);
#endif
                uint32_t kskb;
                if constexpr (KSK_BITS == 28) {
                    // bit0&31 is 16 for odd gtid, 0 for even — branch so lo stays compile-time.
                    if (gtid & 1u) {
                        const uint32_t lo = 16u + (uint32_t)(w * KSK_BITS);
                        kskb = __funnelshift_r(kb[lo >> 5], kb[(lo >> 5) + 1], lo & 31) & ((1u << KSK_BITS) - 1u);
                    } else {
                        const uint32_t lo = (uint32_t)(w * KSK_BITS);
                        kskb = __funnelshift_r(kb[lo >> 5], kb[(lo >> 5) + 1], lo & 31) & ((1u << KSK_BITS) - 1u);
                    }
                } else if constexpr (KSK_BITS)
                    kskb = kskUnpack(kskbp, (uint32_t)(base + w), KSK_BITS, (1u << KSK_BITS) - 1u);
                else
                    kskb = ((const uint32_t*)kskbp)[base + w];
#if FIDESLIB_LAZY_DOT_ACC
                // Wide-prime guard: reduce the ~2^32 keystream word before the lazy accumulate on
                // >28-bit primes — see fusedDotKSKRegen4_.
                aux1[w] += (uint64_t)d4[w] *
                           (C_.prime_bits[primeid] > 28 ? modByRecip(ks[w], pval, recip) : ks[w]);
                aux2[w] += (uint64_t)d4[w] * kskb;
#else
                aux1[w] = modadd(aux1[w], modmult<ALGO_BARRETT>(d4[w], kska, primeid), primeid);
                aux2[w] = modadd(aux2[w], modmult<ALGO_BARRETT>(d4[w], kskb, primeid), primeid);
#endif
            }
        }

        const int rot_index = indexes[j];
#pragma unroll
        for (int w = 0; w < SLOTS; ++w) {
            const uint32_t out_idx = automorph_slot(C_.logN, rot_index, (uint32_t)(base + w));
#if FIDESLIB_LAZY_DOT_ACC
            const uint32_t r1 = modreduce_lazy(aux1[w], primeid);
            const uint32_t r2 = modreduce_lazy(aux2[w], primeid);
#else
            const uint32_t r1 = aux1[w];
            const uint32_t r2 = aux2[w];
#endif
            if (primeid < C_.L) {
                FIDESLIB_HOISTED_ST((uint32_t*)out1[j][pos_dec] + out_idx, r1);
                FIDESLIB_HOISTED_ST((uint32_t*)out2[j][pos_dec] + out_idx, r2);
            } else {
                FIDESLIB_HOISTED_ST((uint32_t*)sout1[j][primeid - C_.L] + out_idx, r1);
                FIDESLIB_HOISTED_ST((uint32_t*)sout2[j][primeid - C_.L] + out_idx, r2);
            }
        }
    }
}


void launchHoistedRotateDotKSK_2(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream, void*** din1, void** c0,
                                 void*** out1, void*** sout1, void*** out2, void*** sout2, int n, const int* indexes,
                                 void*** digits, int num_d, int id, int num_special, int init, void** sc0,
                                 bool c0_modup, int ksk_pack_bits, const uint32_t* seeds, uint32_t n16,
                                 int chain_type) {
    if (seeds && chain_type != 0) {
        // u64 (KSKB) chain: caller passes the /16 grid it uses for the u32 register shape;
        // the u64 shape owns 8 coefficients per thread, so scale grid.x back up by 2.
        dim3 grid64 = grid;
        grid64.x *= 2u;
        hoistedRotateDotKSKRegen64_<<<grid64, block, 0, stream>>>(din1, c0, out1, sout1, out2, sout2, n, indexes,
                                                                  digits, num_d, id, num_special, init, sc0,
                                                                  c0_modup, seeds, n16 * 2u /* == N>>3 */);
        return;
    }
/* Macro, not a helper template, for one reason: BITS must reach the kernel as a COMPILE-TIME
 * template argument (a runtime-width variant costs registers and occupancy), so the arms have
 * to be selected by a switch over instantiations. The macro keeps the argument list — 15+
 * parameters — written once per launcher instead of once per (width x arm) pair. */
    // Regen arm = the cooperative 4-slot kernel (grid.x grows 4x — the caller passes the /16 grid).
    const bool coop4 = seeds != nullptr;
    dim3 grid4 = grid;
    if (coop4)
        grid4.x *= 4u;
#if FIDESLIB_DIN_SMEM
    const size_t din_smem = coop4 ? (size_t)num_d * block.x * sizeof(uint4) : 0;
#else
    const size_t din_smem = 0;
#endif
#define FIDESLIB_HOISTED_DOT_ARM(BITS)                                                                             \
    if (coop4)                                                                                                     \
        hoistedRotateDotKSKRegen4_<BITS><<<grid4, block, din_smem, stream>>>(din1, c0, out1, sout1, out2, sout2, n, \
                                                                      indexes, digits, num_d, id, num_special,      \
                                                                      init, sc0, c0_modup, seeds, n16);            \
    else                                                                                                           \
        hoistedRotateDotKSK_2_<BITS><<<grid, block, shmem, stream>>>(din1, c0, out1, sout1, out2, sout2, n, indexes, \
                                                                     digits, num_d, id, num_special, init, sc0,     \
                                                                     c0_modup);
    switch (ksk_pack_bits) {
        case 0:
            FIDESLIB_HOISTED_DOT_ARM(0)
            break;
        case 27:
            FIDESLIB_HOISTED_DOT_ARM(27)
            break;
        case 28:
            FIDESLIB_HOISTED_DOT_ARM(28)
            break;
        default:
            throw std::runtime_error("launchHoistedRotateDotKSK_2: unsupported ksk_pack_bits " +
                                     std::to_string(ksk_pack_bits));
    }
#undef FIDESLIB_HOISTED_DOT_ARM
}

__global__ void hoistedRotateDotKSKBatched___(void*** c1, void*** din1, void*** c0, void*** sc0, void*** out1,
                                              void*** sout1, void*** out2, void*** sout2, const int n,
                                              const int* indexes, void*** digits, int num_d, int id, int num_special,
                                              int init_, bool c0_modup) {
    // cg::thread_block tb = cg::this_thread_block();
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init_;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;
    //__shared__ cuda::barrier<cuda::thread_scope_block> bar;
    extern __shared__ char buffer[];

    // Initialize barrier (single thread)
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //    init(&bar, blockDim.x * blockDim.y * blockDim.z);
    //}
    //__syncthreads();

    int stride_in = blockDim.x * blockDim.z;
    int stride_ksk = blockDim.x * blockDim.y;
    // uint64_t* in_base = ((uint64_t*)buffer);
    // uint64_t* ksk_base = ((uint64_t*)buffer) + num_d * stride_in;
    uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
    uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.y * blockDim.x + threadIdx.x);

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("in1: %p, ksk: %p\n", in1, ksk);

    //int n_elem = blockDim.x;
    /*
    for (int z = 0; z < blockDim.z; z++) {
        for (uint32_t i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            cuda::memcpy_async(
                tb, in_base + n_elem * z + stride_in * i,
                ((uint64_t*)(decomp ? c1[z] : din1[num_d * z + i])[decomp ? pos_dec : pos]) + blockIdx.x * blockDim.x,
                cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
        }
    }
*/
    /*
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (uint32_t i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            cuda::memcpy_async(
                in_base + n_elem * threadIdx.z + stride_in * i,
                ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos]) +
                    blockIdx.x * blockDim.x,
                cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
        }
    }
*/

    for (uint32_t i = threadIdx.y; i < num_d; i += blockDim.y) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];

        in1[i * stride_in] =
            ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i * stride_in]);
    }
    __syncthreads();
    uint64_t in2 = 0;
    if ((c0_modup && threadIdx.y == 1) || primeid < C_.L) {
        in2 = threadIdx.y == 1 ? ((primeid < C_.L) ? ((uint64_t*)c0[threadIdx.z][pos_dec])[idx]
                                                   : ((uint64_t*)sc0[threadIdx.z][primeid - C_.L])[idx])
                               : in1[stride_in * primeid_digit];
        if (!c0_modup || threadIdx.y == 0)
            in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {

        if (j > 0)
            __syncthreads();

        if (digits[(2 * j + threadIdx.y) * (num_d + 1)]) {
            uint64_t aux;
            for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
                bool decomp = (i == primeid_digit);
                int pos = C_.pos_in_digit[i][primeid];
                /*
            if (threadIdx.x == 0) {


                cuda::memcpy_async(
                    ksk_base + n_elem * threadIdx.y + stride_ksk * i,
                    ((uint64_t*)
                         digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos]) +
                        blockIdx.x * blockDim.x,
                    cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
            }*/
                uint64_t** ksk_from = (uint64_t**)digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)];

                ksk[i * stride_ksk] = ksk_from[decomp ? pos_dec : pos][idx];

                if (PRINT && idx == 0 && blky == 6)
                    printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
            }
            //bar.arrive_and_wait();
            __syncthreads();

            for (int i = 0; i < num_d; ++i) {
                uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

                if (PRINT && idx == 0 && blky == 6)
                    printf("add %d : %lu\n", i, add);
                if (i == 0) {
                    if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
                        aux = modadd(in2, add, primeid);
                    } else {
                        aux = add;
                    }
                } else {
                    aux = modadd(aux, add, primeid);
                }
                if (PRINT && idx == 0 && blky == 6)
                    printf("aux %d : %lu\n", i, aux);
            }

            if (PRINT && idx == 0 && blky == 6)
                printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

            uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
            //uint32_t out_idx = idx;
            uint64_t* out = (primeid < C_.L)
                                ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                    : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                    : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

            out[out_idx] = aux;
        } else {

            uint64_t* out = (primeid < C_.L)
                                ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                    : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                    : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

            if (PRINT && idx == 0)
                printf("y: %d Primeid_digit: %d, in2: %lu %lu\n", blky, primeid_digit, in2,
                       in1[stride_in * (primeid_digit + (primeid_digit == -1))]);

            out[idx] = in2;
        }
    }
}

/*
__global__ void hoistedRotateDotKSKBatched___(void*** c1, void*** din1, void*** c0, void*** sc0, void*** out1,
                                              void*** sout1, void*** out2, void*** sout2, const int n,
                                              const int* indexes, void*** digits, int num_d, int id, int num_special,
                                              int init, bool c0_modup) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    extern __shared__ char buffer[];

    int stride_in = blockDim.x * blockDim.z;
    int stride_ksk = blockDim.x * blockDim.y;
    uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
    uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.x * blockDim.y + threadIdx.y);

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("in1: %p, ksk: %p\n", in1, ksk);

    for (int i = threadIdx.y; i < num_d; i += blockDim.y) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];
        in1[i * stride_in] =
            ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i * stride_in]);
    }

    uint64_t in2 = 0;
    if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
        in2 = ((uint64_t*)(primeid < C_.L ? c0 : sc0)[threadIdx.z][pos_dec])[idx];
        if (!c0_modup)
            in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {
        uint64_t aux;

        if (j > 0)
            __syncthreads();
        for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            ksk[i * stride_ksk] =
                ((uint64_t*)
                     digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos])[idx];

            if (PRINT && idx == 0 && blky == 6)
                printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
        }
        __syncthreads();
        for (int i = 0; i < num_d; ++i) {
            uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("add %d : %lu\n", i, add);
            if (i == 0) {
                if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
                    aux = modadd(in2, add, primeid);
                } else {
                    aux = add;
                }
            } else {
                aux = modadd(aux, add, primeid);
            }
            if (PRINT && idx == 0 && blky == 6)
                printf("aux %d : %lu\n", i, aux);
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

        uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
        //uint32_t out_idx = idx;
        uint64_t* out = (primeid < C_.L) ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                             : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                         : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                             : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

        out[out_idx] = aux;
    }
}
*/

// n32: the gStep > 8 sibling of dotProductLtBatchedPt3___, used by the non-batched linear
// transform (LimbPartition::dotProductPt) and by CoeffsToSlots' wide steps -- at logN=16 a
// single bootstrap exercises BOTH kernels. It was uint64_t-hardcoded with no width branch,
// so on U32 limbs it read/wrote at twice the element size. Same treatment: templated body,
// width branch at the top. modmult<algo> and modadd both have uint32_t overloads.
template <typename T>
__device__ __forceinline__ void dotProductPtBody(void** c0, void** c1, void*** data, const size_t ptroffset,
                                                 const int n, const int idx, const int primeid) {
    constexpr ALGO algo = ALGO_BARRETT;

    T out0, out1;
    T in = ((T*)data[n * 2][ptroffset + blockIdx.y])[idx];
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT:, b: %d, g:, in pt: %lu \n", 0, (unsigned long)in);
    out0 = modmult<algo>(in, ((T*)data[0][ptroffset + blockIdx.y])[idx], primeid);

    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: %d, b: %d, in c0: %lu \n", -1, 0, (unsigned long)((T*)data[0][ptroffset + blockIdx.y])[idx]);

    out1 = modmult<algo>(in, ((T*)data[n][ptroffset + blockIdx.y])[idx], primeid);

    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: %d, b: %d, in c1: %lu \n", -1, 0, (unsigned long)((T*)data[n][ptroffset + blockIdx.y])[idx]);

    for (int i = 1; i < n; ++i) {
        in = ((T*)data[n * 2 + i][ptroffset + blockIdx.y])[idx];

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT:, b: %d, g:, in pt: %lu \n", i, (unsigned long)in);

        T aux0 = modmult<algo>(in, ((T*)data[i][ptroffset + blockIdx.y])[idx], primeid);

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT: %d, b: %d, in c0: %lu \n", -1, i, (unsigned long)((T*)data[i][ptroffset + blockIdx.y])[idx]);

        T aux1 = modmult<algo>(in, ((T*)data[n + i][ptroffset + blockIdx.y])[idx], primeid);

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT: %d, b: %d, in c1: %lu \n", -1, i,
                   (unsigned long)((T*)data[n + i][ptroffset + blockIdx.y])[idx]);
        out0 = modadd(out0, aux0, primeid);
        out1 = modadd(out1, aux1, primeid);
    }
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: , g: , res: %lu \n", (unsigned long)out0);
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: , g: , res: %lu \n", (unsigned long)out1);
    ((T*)c0[ptroffset + blockIdx.y])[idx] = out0;
    ((T*)c1[ptroffset + blockIdx.y])[idx] = out1;
}

__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit,
                              const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];

    if (ISU64(primeid))
        dotProductPtBody<uint64_t>(c0, c1, data, ptroffset, n, idx, primeid);
    else
        dotProductPtBody<uint32_t>(c0, c1, data, ptroffset, n, idx, primeid);
}

/*
__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int& b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;

    __uint128_t* acc_this_thread = (__uint128_t*)(((uint64_t*)buffer) + in_stride * batch) + block_id;
    uint64_t* pt = ((uint64_t*)buffer) + (batch + 2) * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    const bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (!im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i = 0; i < batch; ++i) {
                if (pt_partition != nullptr) {
                    acc_this_thread[0] = (__uint128_t)in[in_stride * i] * pt[0];
                } else {
                    acc_this_thread[0] = 0;
                }

                if constexpr (0) {
                    __syncthreads();
                    if (threadIdx.z == 0) {
                        __uint128_t res = acc_this_thread[0];
                        for (int i = 1; i < blockDim.z; ++i) {
                            res = res + acc_this_thread[i * blockDim.x * blockDim.y];
                        }

                        ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                            modreduce<ALGO_NATIVE>(res, primeid);
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                    __syncthreads();
                } else {
                    const int r_init = 1 << (32 - __clz(b - 1) - 1);
                    int r = r_init;
                    if (r > 0) {
                        __syncthreads();
                        if (threadIdx.z + r < b) {
                            acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
                        }
                    }

                    r >>= 1;
                    for (; r > 0; r >>= 1) {
                        __syncthreads();
                        if (threadIdx.z < r) {
                            acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
                        }
                    }
                    if (threadIdx.z == 0) {

                        ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                            modreduce<ALGO_NATIVE>(acc_this_thread[0], primeid);
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                }
            }
        }
    }
}*/

__global__ void dotProductLtBatchedPt2___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
    //int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    //uint64_t* in = ((uint64_t*)buffer) + block_id;
    //uint64_t in[6];

    //uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = ((uint64_t*)buffer) + block_id;
    //uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    //int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    uint64_t pt;

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {

        for (int i = 0; i < bStep; ++i) {
            uint64_t in = ((uint64_t*)inputs[k * bStep + i][blockIdx.y])[idx];

            for (int j = 0; j < gStep; ++j) {
                void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

                uint64_t mult = 0;
                if (pt_partition != nullptr) {
                    //pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    mult = modmult<ALGO_BARRETT>(in, ((uint64_t*)pt_partition[blockIdx.y])[idx], primeid);
                }
                if (i == 0)
                    acc_this_thread[j * in_stride] = mult;
                else
                    acc_this_thread[j * in_stride] = modadd(acc_this_thread[j * in_stride], mult, primeid);
                if (i == bStep - 1)
                    ((uint64_t*)outputs[k * gStep + j][blockIdx.y])[idx] = acc_this_thread[j * in_stride];
            }
        }
    }
}

// n32: T = limb width, ACC = accumulator wide enough for bStep * (T*T).
// U64: 60-bit primes -> 120-bit products -> __uint128_t accumulator (as before).
// U32: 28-bit primes -> 56-bit products -> a uint64_t accumulator holds bStep up to 2^8=256
// terms without overflow, which covers every baby-step count this kernel is launched with.
// The host sizes the shared buffer with sizeof(__uint128_t) (LimbPartitionBatch.cu), so the
// U32 arm's uint64_t accumulator simply under-uses it — over-allocation is the safe direction.
template <typename T, typename ACC>
#ifndef FIDESLIB_LT_CTIN_ABLATE
#define FIDESLIB_LT_CTIN_ABLATE 0
#endif

__device__ __forceinline__ void dotProductLtBatchedPt3Body(void*** c0_out, void*** c1_out, void*** c0_in,
                                                           void*** c1_in, void*** pts, const int bStep,
                                                           const int gStep, const int n, const int idx,
                                                           const int primeid, char* buffer) {
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);

    ACC* acc_this_thread = ((ACC*)buffer) + block_id;

    // Wide-prime guard: see dotProductLtBatchedPt3BodyG.
    const bool wideAcc = sizeof(T) == 4 && C_.prime_bits[primeid] > 28;
    const bool im_c0 = threadIdx.y == 0;
    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < bStep; ++i) {
            // DIAGNOSTIC (FIDESLIB_LT_CTIN_ABLATE, default 0; produces incorrect results by design):
            // delete the CIPHERTEXT input stream, keep every plaintext load and all arithmetic, to
            // bound the cost of that read. The value is idx-dependent so nothing gets folded away.
#if FIDESLIB_LT_CTIN_ABLATE
            const T in = (T)(idx + i);
#else
            const T in = ((T*)inputs[k * bStep + i][blockIdx.y])[idx];
#endif

            for (int j = 0; j < gStep; ++j) {
                void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

                ACC mult = 0;
                if (pt_partition != nullptr) {
                    mult = (ACC)in * (ACC)((T*)pt_partition[blockIdx.y])[idx];
                }
                if (i == 0)
                    acc_this_thread[j * in_stride] = mult;
                else
                    acc_this_thread[j * in_stride] = acc_this_thread[j * in_stride] + mult;
                if (wideAcc)
                    acc_this_thread[j * in_stride] =
                        (ACC)modreduce_lazy((uint64_t)acc_this_thread[j * in_stride], primeid);

                if (i == bStep - 1) {
                    // modreduce(__uint128_t)->uint64_t and modreduce(uint64_t)->uint32_t both exist
                    const T res = modreduce<ALGO_NATIVE>(acc_this_thread[j * in_stride], primeid);
                    ((T*)outputs[k * gStep + j][blockIdx.y])[idx] = res;
                }
            }
        }
    }
}

// This is the live CoeffsToSlots / SlotsToCoeffs / linear-transform dot product
// (LinearTransform.cu -> RNSPoly::LTdotProductPtBatch -> LimbPartitionBatch.cu). It must branch on
// limb width: a uint64_t-only body against U32 limbs fuses coefficients 2*idx and 2*idx+1.
/* GSTEP-specialized variant: the generic body's runtime j-loop serializes the gStep plaintext
 * loads behind one another (cold-DRAM latency). Compile-time GSTEP (1) unrolls the pt loads so all
 * of an i-iteration's loads are in flight together, (2) moves the accumulators from shared memory
 * into registers, and (3) double-buffers the ciphertext read one i ahead. */
template <typename T, typename ACC, int GSTEP>
__device__ __forceinline__ void dotProductLtBatchedPt3BodyG(void*** c0_out, void*** c1_out, void*** c0_in,
                                                            void*** c1_in, void*** pts, const int bStep, const int n,
                                                            const int idx, const int primeid) {
    // Wide-prime guard: the u64 accumulator's "bStep up to 2^8" bound assumes 28-bit primes
    // (2^56 products). A 30-bit limb's products are 2^60, so bStep >= 16 wraps u64 silently.
    // Reduce per chunk on wide primes; the branch is uniform per primeid.
    const bool wideAcc = sizeof(T) == 4 && C_.prime_bits[primeid] > 28;
    const bool im_c0 = threadIdx.y == 0;
    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        ACC acc[GSTEP];
#pragma unroll
        for (int j = 0; j < GSTEP; ++j)
            acc[j] = 0;
        int i = 0;
#if FIDESLIB_LT_I4
        // i-quad unroll: four ciphertext reads and four plaintext batches issue together.
        for (; i + 3 < bStep; i += 4) {
            const T in0 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i][blockIdx.y] + idx);
            const T in1 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i + 1][blockIdx.y] + idx);
            const T in2 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i + 2][blockIdx.y] + idx);
            const T in3 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i + 3][blockIdx.y] + idx);
#pragma unroll
            for (int j = 0; j < GSTEP; ++j) {
                void** p0 = pts[k * bStep * GSTEP + j * bStep + i];
                void** p1 = pts[k * bStep * GSTEP + j * bStep + i + 1];
                void** p2 = pts[k * bStep * GSTEP + j * bStep + i + 2];
                void** p3 = pts[k * bStep * GSTEP + j * bStep + i + 3];
                ACC m0 = (p0 != nullptr) ? (ACC)in0 * (ACC)FIDESLIB_STREAM_LD((T*)p0[blockIdx.y] + idx) : (ACC)0;
                ACC m1 = (p1 != nullptr) ? (ACC)in1 * (ACC)FIDESLIB_STREAM_LD((T*)p1[blockIdx.y] + idx) : (ACC)0;
                ACC m2 = (p2 != nullptr) ? (ACC)in2 * (ACC)FIDESLIB_STREAM_LD((T*)p2[blockIdx.y] + idx) : (ACC)0;
                ACC m3 = (p3 != nullptr) ? (ACC)in3 * (ACC)FIDESLIB_STREAM_LD((T*)p3[blockIdx.y] + idx) : (ACC)0;
                acc[j] = acc[j] + m0 + m1 + m2 + m3;
                // wide primes: 4 products = 2^62 on top of a <2^31 residue — reduce per quad.
                if (wideAcc)
                    acc[j] = (ACC)modreduce_lazy((uint64_t)acc[j], primeid);
            }
        }
#endif
#if FIDESLIB_LT_I2
        // i-pair unroll: both ciphertext reads and both plaintext batches issue together.
        for (; i + 1 < bStep; i += 2) {
            const T in0 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i][blockIdx.y] + idx);
            const T in1 = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i + 1][blockIdx.y] + idx);
#pragma unroll
            for (int j = 0; j < GSTEP; ++j) {
                void** p0 = pts[k * bStep * GSTEP + j * bStep + i];
                void** p1 = pts[k * bStep * GSTEP + j * bStep + i + 1];
                ACC m0 = (p0 != nullptr) ? (ACC)in0 * (ACC)FIDESLIB_STREAM_LD((T*)p0[blockIdx.y] + idx) : (ACC)0;
                ACC m1 = (p1 != nullptr) ? (ACC)in1 * (ACC)FIDESLIB_STREAM_LD((T*)p1[blockIdx.y] + idx) : (ACC)0;
                acc[j] = acc[j] + m0 + m1;
                if (wideAcc)
                    acc[j] = (ACC)modreduce_lazy((uint64_t)acc[j], primeid);
            }
        }
#endif
        for (; i < bStep; ++i) {
            const T in = FIDESLIB_STREAM_LD2((T*)inputs[k * bStep + i][blockIdx.y] + idx);
#pragma unroll
            for (int j = 0; j < GSTEP; ++j) {
                void** pt_partition = pts[k * bStep * GSTEP + j * bStep + i];
                ACC mult = 0;
                if (pt_partition != nullptr)
                    mult = (ACC)in * (ACC)FIDESLIB_STREAM_LD((T*)pt_partition[blockIdx.y] + idx);
                acc[j] = acc[j] + mult;
                if (wideAcc)
                    acc[j] = (ACC)modreduce_lazy((uint64_t)acc[j], primeid);
            }
        }
#pragma unroll
        for (int j = 0; j < GSTEP; ++j)
            ((T*)outputs[k * GSTEP + j][blockIdx.y])[idx] = modreduce<ALGO_NATIVE>(acc[j], primeid);
    }
}

__global__ void dotProductLtBatchedPt3___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];

    extern __shared__ char buffer[];

    if (ISU64(primeid))
        dotProductLtBatchedPt3Body<uint64_t, __uint128_t>(c0_out, c1_out, c0_in, c1_in, pts, bStep, gStep, n, idx,
                                                          primeid, buffer);
    else if (gStep == 2)
        dotProductLtBatchedPt3BodyG<uint32_t, uint64_t, 2>(c0_out, c1_out, c0_in, c1_in, pts, bStep, n, idx, primeid);
    else if (gStep == 4)
        dotProductLtBatchedPt3BodyG<uint32_t, uint64_t, 4>(c0_out, c1_out, c0_in, c1_in, pts, bStep, n, idx, primeid);
    else if (gStep == 1)
        dotProductLtBatchedPt3BodyG<uint32_t, uint64_t, 1>(c0_out, c1_out, c0_in, c1_in, pts, bStep, n, idx, primeid);
    else
        dotProductLtBatchedPt3Body<uint32_t, uint64_t>(c0_out, c1_out, c0_in, c1_in, pts, bStep, gStep, n, idx,
                                                       primeid, buffer);
}

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;
    //uint64_t in[6];

    uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = acc + block_id;
    uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
        printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[/*in_stride * */ i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[/*in_stride * */ i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i_0 = 0; i_0 < batch; i_0 += 2) {

                if (pt_partition != nullptr) {
                    acc_this_thread[0] = modmult<ALGO_BARRETT>(in[in_stride * i_0], pt[0], primeid);
                    if (i_0 + 1 < batch)
                        acc_this_thread[in_stride] = modmult<ALGO_BARRETT>(in[in_stride * (i_0 + 1)], pt[0], primeid);
                } else {
                    acc_this_thread[0] = 0;
                    acc_this_thread[in_stride] = 0;
                }

                {
                    int r = r_init;
                    if (r > 0) {
                        __syncthreads();
                        if (threadIdx.z + r < b) {
                            acc_this_thread[0] =
                                modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
                        }
                        if (threadIdx.z >= r) {
                            acc_this_thread[in_stride] =
                                modadd(acc_this_thread[in_stride],
                                       acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
                        }
                    }

                    r >>= 1;
                    for (; r > 0; r >>= 1) {
                        __syncthreads();
                        if (threadIdx.z < r) {
                            acc_this_thread[0] =
                                modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
                        }
                        if (threadIdx.z >= b - r) {
                            acc_this_thread[in_stride] =
                                modadd(acc_this_thread[in_stride],
                                       acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
                        }
                    }
                    if (threadIdx.z == 0) {
                        ((uint64_t*)outputs[k * batch * gStep + i_0 * gStep + j][blockIdx.y])[idx] = acc_this_thread[0];
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                    if (threadIdx.z == b - 1) {
                        if (i_0 + 1 < batch)
                            ((uint64_t*)outputs[k * batch * gStep + (i_0 + 1) * gStep + j][blockIdx.y])[idx] =
                                acc_this_thread[in_stride];
                    }
                }
            }
        }
    }
}

/*

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;

    uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = acc + block_id;
    uint64_t* pt = acc + in_stride * batch + (threadIdx.x + blockDim.x * threadIdx.z);

    int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
        printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i = 0; i < batch; ++i) {
                if (pt_partition != nullptr) {
                    acc_this_thread[in_stride * i] = modmult<ALGO_BARRETT>(in[in_stride * i], pt[0], primeid);
                } else {
                    acc_this_thread[in_stride * i] = 0;
                }
            }

            int r = r_init;
            if (r > 0) {
                __syncthreads();
                if (threadIdx.z + r < b) {
                    for (int i = 0; i < batch; ++i) {
                        acc_this_thread[in_stride * i] =
                            modadd(acc_this_thread[in_stride * i],
                                   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
                    }
                }
            }

            r >>= 1;
            for (; r > 0; r >>= 1) {
                __syncthreads();
                if (threadIdx.z < r) {
                    for (int i = 0; i < batch; ++i) {
                        acc_this_thread[in_stride * i] =
                            modadd(acc_this_thread[in_stride * i],
                                   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
                    }
                }
            }
            if (threadIdx.z == 0) {

                for (int i = 0; i < batch; ++i) {
                    ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                        acc_this_thread[in_stride * i];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[in_stride * i]);
                }
            }
        }
    }
}

 */
__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        uint64_t in = modmult<ALGO_SHOUP>(((uint64_t*)b[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
        ((uint64_t*)a[blockIdx.y])[idx] = modadd(in, ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        uint32_t in = modmult<ALGO_SHOUP>(((uint32_t*)b[blockIdx.y])[idx], (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        ((uint32_t*)a[blockIdx.y])[idx] = modadd(in, ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void scaleByP_(void** a, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        uint64_t in = modmult<ALGO_SHOUP>(((uint64_t*)a[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
        ((uint64_t*)a[blockIdx.y])[idx] = in;
    } else {
        uint32_t in = modmult<ALGO_SHOUP>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        ((uint32_t*)a[blockIdx.y])[idx] = in;
    }
}

__global__ void add_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void add_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void sub_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void add_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid],
                                                   primeid, C_.P_shoup[primeid]);
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx],
                                                   (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
            }
        }
    }
}

__global__ void sub_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid],
                                                   primeid, C_.P_shoup[primeid]);
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx],
                                                   (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
            }
        }
    }
}

__global__ void copy_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    }
}

__global__ void copy_reuse_negative_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = load == 0 ? 0 : C_.primes[primeid] - load;
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = load == 0 ? 0 : C_.primes[primeid] - load;
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    }
}

__global__ void add_scalar_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)b)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)b)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void mult_scalar_reuse_b___(void*** a, void*** b, void*** b_shoup, const int primeid_init, const int n,
                                       const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)b)[i * MAXP + primeid];
            uint64_t load_shoup = ((uint64_t*)b_shoup)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)b)[i * MAXP + primeid];
            uint32_t load_shoup = ((uint32_t*)b_shoup)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
            }
        }
    }
}

__global__ void sub_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void mult_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_BARRETT>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_BARRETT>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void binomialMultAccum_(const __grid_constant__ int primeid_init, void** acc0, void** acc1, void** acc2,
                                   void*** a0, void*** a1, void*** b0, void*** b1, const __grid_constant__ int n) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T s0 = ((T*)(acc0[blockIdx.y]))[idx];
        T s1 = ((T*)(acc1[blockIdx.y]))[idx];
        T s2 = 0;
        for (int j = 0; j < n; ++j) {
            const T a0v = ((T*)(((void**)a0[j])[blockIdx.y]))[idx];
            const T a1v = ((T*)(((void**)a1[j])[blockIdx.y]))[idx];
            const T b0v = ((T*)(((void**)b0[j])[blockIdx.y]))[idx];
            const T b1v = ((T*)(((void**)b1[j])[blockIdx.y]))[idx];
            s0 = modadd(s0, modmult<ALGO_BARRETT>(a0v, b0v, primeid), primeid);
            s1 = modadd(s1,
                        modadd(modmult<ALGO_BARRETT>(a0v, b1v, primeid),
                               modmult<ALGO_BARRETT>(a1v, b0v, primeid), primeid),
                        primeid);
            const T d2 = modmult<ALGO_BARRETT>(a1v, b1v, primeid);
            s2 = (j == 0) ? d2 : modadd(s2, d2, primeid);
        }
        ((T*)(acc0[blockIdx.y]))[idx] = s0;
        ((T*)(acc1[blockIdx.y]))[idx] = s1;
        ((T*)(acc2[blockIdx.y]))[idx] = s2;
    } else {
        using T = uint32_t;
        T s0 = ((T*)(acc0[blockIdx.y]))[idx];
        T s1 = ((T*)(acc1[blockIdx.y]))[idx];
        T s2 = 0;
        for (int j = 0; j < n; ++j) {
            const T a0v = ((T*)(((void**)a0[j])[blockIdx.y]))[idx];
            const T a1v = ((T*)(((void**)a1[j])[blockIdx.y]))[idx];
            const T b0v = ((T*)(((void**)b0[j])[blockIdx.y]))[idx];
            const T b1v = ((T*)(((void**)b1[j])[blockIdx.y]))[idx];
            s0 = modadd(s0, modmult<ALGO_BARRETT>(a0v, b0v, primeid), primeid);
            s1 = modadd(s1,
                        modadd(modmult<ALGO_BARRETT>(a0v, b1v, primeid),
                               modmult<ALGO_BARRETT>(a1v, b0v, primeid), primeid),
                        primeid);
            const T d2 = modmult<ALGO_BARRETT>(a1v, b1v, primeid);
            s2 = (j == 0) ? d2 : modadd(s2, d2, primeid);
        }
        ((T*)(acc0[blockIdx.y]))[idx] = s0;
        ((T*)(acc1[blockIdx.y]))[idx] = s1;
        ((T*)(acc2[blockIdx.y]))[idx] = s2;
    }
}

__global__ void binomialMult_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0,
                              void** d1) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = aux1;

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = aux1;

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialMultExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2,
                                    void** d0, void** d1) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialSquare_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialSquareExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, C_.P[primeid], primeid, C_.P_shoup[primeid]);

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] =
            modmult<ALGO_SHOUP>(modadd(aux1, aux1, primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] =
            modmult<ALGO_SHOUP>(modadd(aux1, aux1, primeid), (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialDotProdBatched___(const __grid_constant__ int primeid_init, void*** c0, void*** c1, void*** d0,
                                          void*** d1, void*** c0_out, void*** c1_out, void*** c2_out, int its, int n) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            T d0in = ((T*)(d0[i][blockIdx.y]))[idx];
            T d1in = ((T*)(d1[i][blockIdx.y]))[idx];
            T acc0, acc1, acc2;
            for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
                T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
                T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

                T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

                T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid),
                                primeid);

                T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

                if (j == 0) {
                    acc0 = aux0;
                    acc1 = aux1;
                    acc2 = aux2;
                } else {
                    acc0 = modadd(acc0, aux0, primeid);
                    acc1 = modadd(acc1, aux1, primeid);
                    acc2 = modadd(acc2, aux2, primeid);
                }
            }

            ((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
            ((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
            ((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
        }
    } else {
        using T = uint32_t;
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            T d0in = ((T*)(d0[i][blockIdx.y]))[idx];
            T d1in = ((T*)(d1[i][blockIdx.y]))[idx];
            T acc0, acc1, acc2;
            for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
                T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
                T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

                T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

                T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid),
                                primeid);

                T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

                if (j == 0) {
                    acc0 = aux0;
                    acc1 = aux1;
                    acc2 = aux2;
                } else {
                    acc0 = modadd(acc0, aux0, primeid);
                    acc1 = modadd(acc1, aux1, primeid);
                    acc2 = modadd(acc2, aux2, primeid);
                }
            }

            ((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
            ((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
            ((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
        }
    }
}

}  // namespace CKKS
}  // namespace FIDESlib

#define YY(algo)                                                 \
    template __global__ void FIDESlib::CKKS::Scalar_mult_<algo>( \
        void** a, const uint64_t* b, const __grid_constant__ int primeid_init, const uint64_t* shoup_mu);
#include "ntt_types.inc"
#undef YY

template __global__ void FIDESlib::CKKS::addMult_<uint64_t>(uint64_t* l, const uint64_t* l1, const uint64_t* l2,
                                                            const __grid_constant__ int primeid);

template __global__ void FIDESlib::CKKS::addMult_<uint32_t>(uint32_t* l, const uint32_t* l1, const uint32_t* l2,
                                                            const __grid_constant__ int primeid);
