#pragma once
// Tensor-core 256-point NTT/INTT core (traffic campaign, 2026-10-04).
//
// The two-pass 4-step kernel (NTT.cu) runs, per block, eight independent 256-point transforms on an 8-row
// shared-memory tile. That core is one fixed linear map R (256x256 over Z_p) per prime, PROBED on the device
// from the butterfly code itself (NTTtc.cu), so the result is bit-identical. Here the core is evaluated as a
// matmul on the INT8 tensor cores: residues (< 2^28) and R entries are split into four unsigned 8-bit chunks,
// mma.m16n8k32.u8 accumulates the 16 chunk products into 7 shifted s32 accumulators (exact: 8 k-steps x 32
// x 255^2 x <=4 pairs < 2^27), and the exact 64-bit sum is reduced mod p with a 64-bit Barrett step.
// n = 8 of the mma tile is exactly the 8 rows of the tile; m = 256 output elements; k = 256 inputs.
#include <cstdint>
#include <cuda_runtime.h>
#include "ConstantsGPU.cuh"

namespace FIDESlib {

struct TcTables {
    const uint8_t* R[MAXP];     // forward core: 4 planes [l][m][k], 65536 B each (nullptr = not built)
    const uint8_t* Rinv[MAXP];  // inverse core
    uint64_t mu[MAXP];          // floor(2^64 / p) for the 64-bit Barrett reduction
    const uint8_t* B[MAXP];     // v2 blobs (two-stage radix-16 core), forward; nullptr = map not separable
    const uint8_t* Binv[MAXP];  // v2 blobs, inverse
};

/// FIDESLIB_TC_NTT (env, default 0) / setTcNtt: route the u32 transform cores to the tensor-core path.
int tcNttFlag();
void setTcNtt(int v);
/// Build the per-prime core tables on `device` (NTTtc.cu) and publish them to NTT.cu's device symbols.
void buildTcNttTables(const Global::Globals* G, int device);
/// NTT.cu: publish tables / flag to the kernels' translation unit.
void setTcTables(const TcTables& t, int device);
/// Device self-test of the tensor-core core against the butterflies (NTTtc.cu); returns mismatching words.
long tcNttSelfTest(const Global::Globals* G, int primeid, int device, int version = 1);

}  // namespace FIDESlib
