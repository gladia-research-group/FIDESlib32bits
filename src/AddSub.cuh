//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_ADDSUB_CUH
#define FIDESLIB_ADDSUB_CUH
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"

namespace FIDESlib {

template <typename T>
__global__ void add_(T* a, const T* b, const int primeId);

template <typename T>
__global__ void sub_(T* a, const T* b, const int primeId);

/** a = a .+ b % p*/
__global__ void add_(void** a, void** b, const int primeid_init);

/** BYTES-indexed in-place add; grid {bytes_per_limb/(BYTES*128), nlimbs}, block 128. AddSub.cu. */
void launchAddBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int primeid_init,
                    int bytes_per_thread);
void launchSubBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int primeid_init,
                    int bytes_per_thread);
void launchScalarAddSubBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, const uint64_t* b,
                             int primeid_init, int bytes_per_thread, bool sub);
/** Bytes-per-thread for the vectorized pointwise family; FIDESLIB_ADD_BYTES, default 16. */
int fideslibAddBytes();

/** a = a .- b % p*/
__global__ void sub_(void** a, void** b, const int primeid_init);

/** a = b .+ c % p*/
__global__ void add_(void** a, void** b, void** c, const int primeid_init);

/** a = b .- c % p*/
__global__ void sub_(void** a, void** b, void** c, const int primeid_init);

/** a = a + b % p*/
__global__ void scalar_add_(void** a, uint64_t* b, const int primeid_init);

/** a = a - b % p*/
__global__ void scalar_sub_(void** a, uint64_t* b, const int primeid_init);

template <typename T>
__forceinline__ __device__ T modadd(const T a, const T b, const int primeId) {
    const T prime_p = C_.primes[primeId];
    // if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, prime_p);
    T tmp0 = a + b;
    return tmp0 - prime_p * (tmp0 >= prime_p);
}

template <typename T>
__forceinline__ __device__ T modsub(const T a, const T b, const int primeId) {
    const T prime_p = C_.primes[primeId];
    //   if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, prime_p);
    T tmp0 = a - b;
    return tmp0 + prime_p * (tmp0 >= prime_p);
}

}  // namespace FIDESlib
#endif  //FIDESLIB_ADDSUB_CUH
