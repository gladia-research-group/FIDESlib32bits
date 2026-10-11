//
// FIDESLIB_BCONV2=1 (default off): a faster kernel for the key-switch ModUp base conversion (DecompAndModUpConv), same
// tables and the same exact result (lazy reductions give the canonical residue in any order). One thread per
// coefficient and 16 targets per block, 4 accumulated together; the block's matrix tile and the pre-scaled digit limbs
// sit in shared memory instead of a per-MAC global matrix load and a serial chain over 2 threads per coefficient.
// All-u32 chains only (Constants::type == 0); the callers keep the original kernel otherwise.
//
#ifndef FIDESLIB_CKKS_MODUPCONV2_CUH
#define FIDESLIB_CKKS_MODUPCONV2_CUH

#include <cuda_runtime.h>
#include "ConstantsGPU.cuh"

namespace FIDESlib::CKKS {

/** FIDESLIB_BCONV2 (default 0), settable at run time for the bit-exactness gate. */
bool modUpConv2On();
void setModUpConv2(int on);

/** Drop-in for DecompAndModUpConv<<<N/64, {64,2}, 8*maxSize*64>>>(a, n, b, d, G): hc = this GPU's host constants. */
void launchModUpConv2(void** a, int n, void** b, int d, const Global::Globals* G, const Constants& hc, int maxSize,
                      cudaStream_t s);

/** Drop-in for DecompAndModUpConvMulti (digit = blockIdx.z here): dids_host mirrors the device digit-id table. */
void launchModUpConv2Multi(void*** atabs, int n, void*** btabs, const int* dids, const int* dids_host, int ndig,
                           const Global::Globals* G, const Constants& hc, int maxSize, cudaStream_t s);

}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_MODUPCONV2_CUH
