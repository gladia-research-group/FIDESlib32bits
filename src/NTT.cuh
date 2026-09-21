//
// Created by carlosad on 4/04/24.
//

#ifndef FIDESLIB_NTT_CUH
#define FIDESLIB_NTT_CUH

#include <cinttypes>
#include "CKKS/forwardDefs.cuh"
#include "ConstantsGPU.cuh"

namespace FIDESlib {

struct FusedIterationsParams {
    struct __align__(128) AtomicCounter {
        uint32_t n = 0;
        uint32_t pad[(128 - sizeof(uint32_t)) / sizeof(uint32_t)];
    };
    AtomicCounter counters[MAXP];
    struct Conf {
        dim3 grid;
        dim3 block;
    };
    Conf first;
    Conf second;
};

/* Utility function, no real use other than testing. */
template <typename T>
__global__ void Bit_Reverse(T* dat, uint32_t N);

/* Get pointer to kernel, needed for explicit Cuda Graph construction. */
void* get_NTT_reference(bool second);

// ------------------------------------- INTT ----------------------------------------
/** Kernel fusions */
enum INTT_MODE { INTT_NONE, INTT_MULT_AND_SAVE, INTT_MULT_AND_ACC, INTT_ROTATE_AND_SAVE, INTT_SQUARE_AND_SAVE };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
__global__ void INTT_(const Global::Globals* Globals, T* __restrict__ dat, const int __grid_constant__ primeid,
                      T* __restrict__ res, const T* __restrict__ dat2 = nullptr, T* __restrict__ res0 = nullptr,
                      T* __restrict__ res1 = nullptr, const T* __restrict__ kska = nullptr,
                      const T* __restrict__ kskb = nullptr, T* __restrict__ c0 = nullptr,
                      const T* __restrict__ c0tilde = nullptr);

template <bool second, ALGO algo, INTT_MODE mode>
__global__ void INTT_(const Global::Globals* Globals, void** __restrict__ dat, const int __grid_constant__ primeid_init,
                      void** __restrict__ res, void** __restrict__ dat2 = nullptr, void** __restrict__ res0 = nullptr,
                      void** __restrict__ res1 = nullptr, void** __restrict__ kska = nullptr,
                      void** __restrict__ kskb = nullptr, void** __restrict__ c0 = nullptr,
                      void** __restrict__ c0tilde = nullptr);

// ------------------------------------- NTT ----------------------------------------
/** Kernel fusions */
// NTT_RESCALE2: fused composite DOUBLE prime drop — one wide pass instead of two
// sequential NTT_RESCALE passes. Sequential-drop semantics preserved exactly (bit-identical to
// two NTT_RESCALE passes): the second top's once-divided value w is derived per-coefficient in
// coeff domain, and per-prime constants commute with the (exact, modular) NTT. Only ALGO_SHOUP
// is instantiated (the only algo the rescale path uses). dat = limbptr + (limbsize-2), so
// dat[0] = the qb limb, dat[1] = the qa top; primeid_rescale = qa's primeid, and qb's primeid
// is derived as primeid_rescale - 1 (single-GPU composite chains have level-ordered q ids —
// asserted by the host caller, which falls back to two passes otherwise).
enum NTT_MODE { NTT_NONE, NTT_RESCALE, NTT_MULTPT, NTT_MODDOWN, NTT_KSK_DOT, NTT_KSK_DOT_ACC, NTT_RESCALE2 };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
__global__ void NTT_(const Global::Globals* Globals, T* __restrict__ dat, const int __grid_constant__ primeid,
                     T* __restrict__ res, const T* __restrict__ pt = nullptr,
                     const int __grid_constant__ primeid_rescale = -1, T* __restrict__ res2 = nullptr,
                     const T* __restrict__ kskb = nullptr);

template <bool second, ALGO algo, NTT_MODE mode>
__global__ void NTT_(const Global::Globals* Globals, void** __restrict__ dat, const int __grid_constant__ primeid_init,
                     void** __restrict__ res, void** __restrict__ pt = nullptr,
                     const int __grid_constant__ primeid_rescale = -1, void** __restrict__ res2 = nullptr,
                     void** __restrict__ kskb = nullptr);

// ------------------------------------- 1D NTT version ----------------------------------------

template <typename T, int WARP_SIZE = 32>
__global__ void NTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N,
                       const int __grid_constant__ primeid, const int __grid_constant__ logN);

template <typename T, int WARP_SIZE = 32>
__global__ void INTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N,
                        const int __grid_constant__ primeid, const T N_inv, const int __grid_constant__ logN);
}  // namespace FIDESlib

#endif  //FIDESLIB_NTT_CUH
