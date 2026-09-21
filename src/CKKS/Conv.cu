//
// Created by carlosad on 4/04/24.
//
#include "AddSub.cuh"
#include "CKKS/Conv.cuh"
#include "ModMult.cuh"

#include <cuda_runtime.h>

// Evict-first loads on the base-conversion input limbs (read exactly once each).
#ifndef FIDESLIB_BC_LDCS
#define FIDESLIB_BC_LDCS 0
#endif
#if FIDESLIB_BC_LDCS
#define FIDESLIB_BC_STREAM_LD(p) __ldcs(p)
#else
#define FIDESLIB_BC_STREAM_LD(p) (*(p))
#endif

/* LAZY REDUCTION in the u32 BConv arms — accumulate the raw 32x32->64
 * products and reduce once (modreduce_lazy) instead of a Shoup multiply + modadd per term.
 * Default ON; build with -DFIDESLIB_LAZY_BCONV=0 to restore the eager reference arm, which is
 * how the wall A/B is run (there is no env knob — the arms are compile-time). */
#ifndef FIDESLIB_LAZY_BCONV
#define FIDESLIB_LAZY_BCONV 1
#endif

namespace FIDESlib::CKKS {

/* The BConv staging buffer is uint64_t, but on an all-u32 chain every slot holds a residue < 2^28:
 * a stride-1 uint64_t shared access is an inherent 2-way bank conflict, so alias the same allocation
 * at 32-bit width for type==0 (the host sizes shared_bytes with sizeof(uint64_t); the u32 view under-uses it). */
__device__ __forceinline__ void stBuff(const bool u32buf, uint64_t* b64, uint32_t* b32, const int pos,
                                       const uint64_t v) {
    if (u32buf)
        b32[pos] = (uint32_t)v;
    else
        b64[pos] = v;
}


template <typename T>
__global__ void conv1_(T* a, const T q_hat_inv, const int primeid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    a[idx] = modmult(a[idx], q_hat_inv, primeid);
}

template __global__ void conv1_(uint32_t* a, const uint32_t q_hat_inv, const int primeid);

template __global__ void conv1_(uint64_t* a, const uint64_t q_hat_inv, const int primeid);

constexpr bool USING_CONSTANTS_TABLE = 0;

template <ALGO algo>
__global__ void ModDown2(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b,
                         const __grid_constant__ int primeid_init, const Global::Globals* Globals) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ char shared_mem[];

    uint64_t* buff = &((uint64_t*)shared_mem)[0];
    uint32_t* buff32 = (uint32_t*)shared_mem;  // type==0 alias, half width
    const bool u32buf = (C_.type == 0);       // grid-uniform

    for (int i = threadIdx.y; i < C_.K; i += blockDim.y) {
        int primeid = i + C_.L;
        if constexpr (USING_CONSTANTS_TABLE) {  // using constants table
            constexpr ALGO algo_ = algo == ALGO_SHOUP ? ALGO_BARRETT : algo;
            if (ISU64(primeid)) {
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo_>(FIDESLIB_BC_STREAM_LD((const uint64_t*)(b[i]) + idx), TABLE64(C_.L, C_.L + i), C_.L + i));
            } else {
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo_>(FIDESLIB_BC_STREAM_LD((const uint32_t*)(b[i]) + idx), (uint32_t)TABLE32(C_.L, C_.L + i), C_.L + i));
            }
        } else {
            if constexpr (algo != 3) {
                if (ISU64(primeid)) {
                    stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo>(FIDESLIB_BC_STREAM_LD((const uint64_t*)(b[i]) + idx), G_->ModDown_pre_scale[primeid], primeid));
                } else {
                    stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo>((uint64_t)FIDESLIB_BC_STREAM_LD((const uint32_t*)(b[i]) + idx), G_->ModDown_pre_scale[primeid], primeid));
                }
            } else {
                if (ISU64(primeid)) {
                    stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo>(FIDESLIB_BC_STREAM_LD((const uint64_t*)(b[i]) + idx), G_->ModDown_pre_scale[primeid],
                                                               primeid, G_->ModDown_pre_scale_shoup[primeid]));
                } else {
                    // U32 primes carry 2^32-scaled Shoup constants: the multiply must run in the
                    // 32-bit overload. Promoting to uint64_t pairs Shoup_mult_64 with a 2^32-scaled
                    // psi, whose __umul64hi quotient is always 0 -> buff holds the UNREDUCED product
                    // (~2^56). Still congruent mod p (looks in-range after the final modreduce), but
                    // base conversion needs the exact representative -> k*p excess -> the converted
                    // limbs go mutually CRT-inconsistent.
                    stBuff(u32buf, buff, buff32, tid + blockDim.x * i,
                    modmult<algo>(FIDESLIB_BC_STREAM_LD((const uint32_t*)(b[i]) + idx), (uint32_t)G_->ModDown_pre_scale[primeid], primeid,
                                      (uint32_t)G_->ModDown_pre_scale_shoup[primeid]));
                }
            }
            /*
                if (idx == 0) {
                    printf("Pre Scale from primeid:%d: %lu ", primeid,
                           G_::ModDown_pre_scale[primeid]);
                    for (int i_ = 0; i_ < 2; ++i_) {
                        printf("%lu ", buff[tid + i_ + blockDim.x * i]);
                    }
                    printf("\n");
                }
*/
        }
    }
    __syncthreads();

    for (int j = threadIdx.y; j < n; j += blockDim.y) {

        //if (idx == 0) printf("Matrix to %d: ", j);
        int primeid = C_.primeid_flattened[primeid_init + j];
        if constexpr (1) {
            if (ISU64(primeid)) {
                // WIDTH-NEUTRAL: the same per-term Shoup replacement for
                // U64 output primes — valid on ANY chain (uniform-64 or mixed): buff entries
                // are canonical u64 residues (< their source prime), Shoup_mult_64 accepts an
                // arbitrary 64-bit multiplicand, and the *_shoup companion matrix carries the
                // width-aware 2^64-scaled constant keyed on this output prime. Replaces the
                // ~200-instruction emulated __uint128_t % p per (coefficient, output limb)
                // below; identical canonical residue by construction (reduce-then-add vs
                // sum-then-reduce), so bit-exact vs CPU OpenFHE.
                uint64_t res = 0;
                for (int i = 0; i < C_.K; ++i) {
                    const int m = MODDOWN_MATRIX(i, primeid);
                    res = modadd(res,
                                 modmult<ALGO_SHOUP>(buff[i * blockDim.x + tid], G_->ModDown_matrix[m], primeid,
                                                     G_->ModDown_matrix_shoup[m]),
                                 primeid);
                }
                ((uint64_t*)a[j])[idx] = res;
                continue;
            }
            if (C_.type == 0) {
                // All-U32 chain fast path. The generic arm below accumulates in __uint128_t and
                // finishes with an emulated `res % p`. On an all-U32 chain every buff entry and matrix
                // value is a residue < 2^28, so raw 32x32->64 products are accumulated and reduced ONCE
                // (modreduce_lazy): the IDENTICAL canonical residue, bit-exact. Matrix reads go through
                // the u32 shadow copies (filled iff type==0). Mixed and U64 chains keep the generic arm;
                // the branch is grid-uniform. Bound: C_.K special primes x 2^56, far short of u64.
#if FIDESLIB_LAZY_BCONV
                uint64_t acc = 0;
                for (int i = 0; i < C_.K; ++i) {
                    const int m = MODDOWN_MATRIX(i, primeid);
                    acc += (uint64_t)buff32[i * blockDim.x + tid] * (uint64_t)G_->ModDown_matrix32[m];
                }
                ((uint32_t*)a[j])[idx] = modreduce_lazy(acc, primeid);
#else  // reference EAGER form — kept as the A/B arm and the bit-exactness reference
                uint32_t res = 0;
                for (int i = 0; i < C_.K; ++i) {
                    const int m = MODDOWN_MATRIX(i, primeid);
                    res = modadd(res,
                                 modmult<ALGO_SHOUP>(buff32[i * blockDim.x + tid],
                                                     G_->ModDown_matrix32[m], primeid,
                                                     G_->ModDown_matrix_shoup32[m]),
                                 primeid);
                }
                ((uint32_t*)a[j])[idx] = res;
#endif
                continue;
            }
            __uint128_t res = 0;
            for (int i = 0; i < C_.K; ++i) {
                res = res + (__uint128_t)buff[i * blockDim.x + tid] * G_->ModDown_matrix[MODDOWN_MATRIX(i, primeid)];
            }

            // TODO use better reduction
            if (!ISU64(primeid)) {
                ((uint32_t*)a[j])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid);
            } else {
                ((uint64_t*)a[j])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid);
            }
        } else {
            uint64_t res = 0;
            for (int i = 0; i < C_.K; ++i) {
                if constexpr (USING_CONSTANTS_TABLE) {  // using constants table
                    constexpr ALGO algo_ = algo == ALGO_SHOUP ? ALGO_BARRETT : algo;
                    uint64_t aux = modmult<algo_>(buff[i * blockDim.x + tid], (uint64_t)TABLE64(C_.L + i, j), j);
                    // res = modadd(res, aux, j);
                } else {
                    if constexpr (algo != 3) {

                        uint64_t aux =
                            modmult<algo>(buff[i * blockDim.x + tid], G_->ModDown_matrix[MODDOWN_MATRIX(i, j)], j);

                        res = modadd(res, aux, j);

                    } else {
                        uint64_t aux =
                            modmult<algo>(buff[i * blockDim.x + tid], G_->ModDown_matrix[MODDOWN_MATRIX(i, j)], j,
                                          G_->ModDown_matrix_shoup[MODDOWN_MATRIX(i, j)]);
                        res = modadd(res, aux, j);
                    }
                }

                if (idx == 0)
                    printf("%lu ", G_->ModDown_matrix[MODDOWN_MATRIX(i, j)]);
            }

            if (!ISU64(j)) {
                ((uint32_t*)a[j])[idx] = (uint32_t)res;
            } else {
                ((uint64_t*)a[j])[idx] = (uint64_t)res;
            }
        }
        /*
            if (idx == 0)
                printf("\n");
*/
    }
}

#define YY(algo)                                                                                             \
    template __global__ void ModDown2<algo>(void** __restrict__ a, const __grid_constant__ int n,            \
                                            void** __restrict__ b, const __grid_constant__ int primeid_init, \
                                            const Global::Globals* Globals);

#include "ntt_types.inc"

#undef YY

template <ALGO algo>
__global__ void DecompAndModUpConv(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b,
                                   const int __grid_constant__ d, const Global::Globals* Globals) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ char shared_mem[];
    uint64_t* buff = ((uint64_t*)shared_mem);
    uint32_t* buff32 = (uint32_t*)shared_mem;  // type==0 alias, half width
    const bool u32buf = (C_.type == 0);       // grid-uniform
    /*
        if (threadIdx.y == 0 && idx == 0) {
            for (int j = d; j < d + 1; ++j) {
                for (int k = 0; k < 64; ++k) {
                    printf("%d %d:", j, k);
                    for (int l = 0; l < 64; ++l) {
                        printf("%lu ", G_::DecompAndModUp_pre_scale[MODUPIDX_SCALE(j, k, l)]);
                    }
                    printf("\n");
                }
            }

            for (int i = 0; i < 64; ++i) {
                for (int j = d; j < d + 1; ++j) {
                    for (int k = 0; k < 64; ++k) {
                        printf("%d %d %d:", i, j, k);
                        for (int l = 0; l < 64; ++l) {
                            printf("%lu ", G_::DecompAndModUp_matrix[MODUPIDX_MATRIX(i, j, k, l)]);
                        }
                        printf("\n");
                    }
                }
            }

        }
*/

    int n_d_n = C_.num_primeid_digit_from[d][n - 1];
    // assert(n_d_n != 0);
    for (int i_ = threadIdx.y; i_ < n_d_n; i_ += blockDim.y) {
        const int primeid = C_.primeid_digit_from[d][i_];
        const int pos = i_;  //C_.pos_in_digit[d][primeid];
        assert(a[i_] != nullptr);
        if constexpr (algo != 3) {
            if (ISU64(primeid)) {
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i_,
                    modmult<algo>(((uint64_t*)(a[pos]))[idx],
                                  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid));
            } else {
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i_,
                    modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx],
                                  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid));
            }
        } else {
            if (ISU64(primeid)) {
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i_,
                    modmult<algo>(
                    ((uint64_t*)(a[pos]))[idx], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
                    primeid, G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]));
            } else {
                // See ModDown2: 2^32-scaled Shoup constants require the 32-bit multiply; the
                // uint64_t promotion left buff unreduced (~2^56) and poisoned the digit base
                // conversion with k*p multiples.
                stBuff(u32buf, buff, buff32, tid + blockDim.x * i_,
                    modmult<algo>(((uint32_t*)a[pos])[idx],
                                  (uint32_t)G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid,
                                  (uint32_t)G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]));
            }
        }
        /*
            if (i_ == 0 && idx == 0) {
                printf("Pre Scale from d=%d, n_d_n-1=%d, i_:%d, primeid=%d, index:%d: %lu ", d, n_d_n - 1, i_, primeid,
                       MODUPIDX_SCALE(d, n_d_n - 1, primeid),
                       G_::DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
                for (int i = 0; i < 8; ++i) {
                    printf("%lu ", buff[tid + i + blockDim.x * i_]);
                }
                printf("\n");
            }
*/
    }

    __syncthreads();

    //assert(C_.num_primeid_digit_to[d][n - 1] != 0);
    for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
        //if (j_ == 0 && idx == 0) printf("Matrix to %d: ", j_);
        const int primeid_j = C_.primeid_digit_to[d][j_];
        if (primeid_j < n || primeid_j >= C_.L) {

            if constexpr (1) {
                if (ISU64(primeid_j)) {
                    // WIDTH-NEUTRAL u64 fast path — same rationale as ModDown2 above: per-term
                    // Shoup vs the emulated u128 % p, valid on any chain, bit-exact.
                    uint64_t res64 = 0;
                    for (int i_ = 0; i_ < n_d_n; ++i_) {
                        const int primeid = C_.primeid_digit_from[d][i_];
                        const int m = MODUPIDX_MATRIX(n - 1, d, primeid, primeid_j);
                        res64 = modadd(res64,
                                       modmult<ALGO_SHOUP>(buff[i_ * blockDim.x + tid],
                                                           G_->DecompAndModUp_matrix[m], primeid_j,
                                                           G_->DecompAndModUp_matrix_shoup[m]),
                                       primeid_j);
                    }
                    assert(b[j_] != nullptr);
                    ((uint64_t*)b[j_])[idx] = res64;
                    continue;
                }
                if (C_.type == 0) {
                    // All-U32 chain fast path — same rationale as ModDown2 above: u32 shadow matrices,
                    // raw 32x32->64 products accumulated (one IMAD.WIDE.U32 per term) and reduced ONCE
                    // with modreduce_lazy. BIT-EXACT: sum(b_i*m_i) mod p == sum(b_i*m_i mod p) mod p and
                    // the sum is exact: products < 2^56, so n_d_n would have to exceed 2^8 to wrap u64.
#if FIDESLIB_LAZY_BCONV
                    uint64_t acc = 0;
                    for (int i_ = 0; i_ < n_d_n; ++i_) {
                        const int primeid = C_.primeid_digit_from[d][i_];
                        const int m = MODUPIDX_MATRIX(n - 1, d, primeid, primeid_j);
                        acc += (uint64_t)buff32[i_ * blockDim.x + tid] *
                               (uint64_t)G_->DecompAndModUp_matrix32[m];
                    }
                    assert(b[j_] != nullptr);
                    ((uint32_t*)b[j_])[idx] = modreduce_lazy(acc, primeid_j);
#else  // reference EAGER form — kept as the A/B arm and the bit-exactness reference
                    uint32_t res32 = 0;
                    for (int i_ = 0; i_ < n_d_n; ++i_) {
                        const int primeid = C_.primeid_digit_from[d][i_];
                        const int m = MODUPIDX_MATRIX(n - 1, d, primeid, primeid_j);
                        res32 = modadd(res32,
                                       modmult<ALGO_SHOUP>(buff32[i_ * blockDim.x + tid],
                                                           G_->DecompAndModUp_matrix32[m], primeid_j,
                                                           G_->DecompAndModUp_matrix_shoup32[m]),
                                       primeid_j);
                    }
                    assert(b[j_] != nullptr);
                    ((uint32_t*)b[j_])[idx] = res32;
#endif
                    continue;
                }

                __uint128_t res = 0;
                for (int i_ = 0; i_ < n_d_n; ++i_) {

                    const int primeid = C_.primeid_digit_from[d][i_];

                    assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
                    res = res + (__uint128_t)buff[i_ * blockDim.x + tid] *
                                    G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];

                    if (0) {
                        uint64_t aux = G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
                        printf("(%d, %d, %d, %d, %lu)", i_, primeid, primeid_j,
                               MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j), aux);
                    }
                }

                assert(b[j_] != nullptr);
                if (!ISU64(primeid_j)) {
                    ((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid_j);
                } else {
                    ((uint64_t*)b[j_])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
                }
            } else {
                uint64_t res = 0;
                for (int i_ = 0; i_ < n_d_n; ++i_) {
                    assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);

                    if constexpr (algo != 3) {
                        uint64_t aux = modmult<algo>(
                            buff[i_ * blockDim.x + tid],
                            G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)], primeid_j);
                        res = modadd(res, aux, primeid_j);
                    } else {
                        uint64_t aux = modmult<algo>(
                            buff[i_ * blockDim.x + tid],
                            G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)], primeid_j,
                            G_->DecompAndModUp_matrix_shoup[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)]);
                        res = modadd(res, aux, primeid_j);
                    }
                    /*
                    if (j_ == 0 && idx == 0)
                        printf("%lu ", G_::DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)]);
*/
                }

                assert(b[j_] != nullptr);
                if (!ISU64(primeid_j)) {
                    ((uint32_t*)b[j_])[idx] = (uint32_t)res;
                } else {
                    ((uint64_t*)b[j_])[idx] = (uint64_t)res;
                }
            }
        }
        /*
            if (j_ == 0 && idx == 0)
                printf("\n");
        */
    }
}

#define YY(algo)                                                                                            \
    template __global__ void DecompAndModUpConv<algo>(void** __restrict__ a, const int __grid_constant__ n, \
                                                      void** __restrict__ b, const int __grid_constant__ d, \
                                                      const Global::Globals* Globals);
#include "ntt_types.inc"

#undef YY
}  // namespace FIDESlib::CKKS
