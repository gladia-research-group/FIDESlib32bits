//
// Created by carlosad on 4/04/24.
//
#include "AddSub.cuh"
#include "CKKS/Conv.cuh"
#include "ModMult.cuh"

#include <cuda_runtime.h>

namespace FIDESlib::CKKS {

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

    for (int i = threadIdx.y; i < C_.K; i += blockDim.y) {
        int primeid = i + C_.L;
        if constexpr (USING_CONSTANTS_TABLE) {  // using constants table
            constexpr ALGO algo_ = algo == ALGO_SHOUP ? ALGO_BARRETT : algo;
            if (ISU64(primeid)) {
                buff[tid + blockDim.x * i] =
                    modmult<algo_>(((uint64_t*)(b[i]))[idx], TABLE64(C_.L, C_.L + i), C_.L + i);
            } else {
                buff[tid + blockDim.x * i] =
                    modmult<algo_>(((uint32_t*)b[i])[idx], (uint32_t)TABLE32(C_.L, C_.L + i), C_.L + i);
            }
        } else {
            if constexpr (algo != 3) {
                if (ISU64(primeid)) {
                    buff[tid + blockDim.x * i] =
                        modmult<algo>(((uint64_t*)(b[i]))[idx], G_->ModDown_pre_scale[primeid], primeid);
                } else {
                    buff[tid + blockDim.x * i] =
                        modmult<algo>((uint64_t)((uint32_t*)b[i])[idx], G_->ModDown_pre_scale[primeid], primeid);
                }
            } else {
                if (ISU64(primeid)) {
                    buff[tid + blockDim.x * i] = modmult<algo>(((uint64_t*)(b[i]))[idx], G_->ModDown_pre_scale[primeid],
                                                               primeid, G_->ModDown_pre_scale_shoup[primeid]);
                } else {
                    buff[tid + blockDim.x * i] =
                        modmult<algo>((uint64_t)((uint32_t*)b[i])[idx], G_->ModDown_pre_scale[primeid], primeid,
                                      G_->ModDown_pre_scale_shoup[primeid]);
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
                buff[tid + blockDim.x * i_] =
                    modmult<algo>(((uint64_t*)(a[pos]))[idx],
                                  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
            } else {
                buff[tid + blockDim.x * i_] =
                    modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx],
                                  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
            }
        } else {
            if (ISU64(primeid)) {
                buff[tid + blockDim.x * i_] = modmult<algo>(
                    ((uint64_t*)(a[pos]))[idx], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
                    primeid, G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
            } else {
                buff[tid + blockDim.x * i_] =
                    modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx],
                                  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid,
                                  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
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
