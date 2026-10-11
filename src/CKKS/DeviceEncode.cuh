#ifndef FIDESLIB_CKKS_DEVICE_ENCODE_CUH
#define FIDESLIB_CKKS_DEVICE_ENCODE_CUH
// CKKS packed encoding on the GPU: OpenFHE's CKKSPackedEncoding::Encode (full slots) on the SPRU encoder kernels.
#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

/// Encode `values` (at most N/2, zero-padded) into `pt` with `limbs` q-limbs: coefficients round(v * scale), the
/// degree's extra integer factors round(scale)^(deg-1), and OpenFHE's 2^61 approx split when the coefficients exceed
/// a word. Metadata as OpenFHE reports it: NoiseFactor = scale^deg, NoiseLevel = deg, slots = N/2. No host sync:
/// the result is complete on the returned stream (the plaintext's own).
cudaStream_t encodeOnDevice(Plaintext& pt, const std::vector<std::complex<double>>& values, int limbs, double scale,
                            int deg, double* cmaxPinned = nullptr);

/// `n` rows of `width` values from host memory (dtype 0 float32, 1 float64, 2 complex64, 3 complex128; complex rows
/// interleave re/im), each encoded into pts[i] as encodeOnDevice does: one pinned staging copy and one upload for all
/// rows, widened to double on the device. No host sync; every plaintext is complete on the stream returned for it.
std::vector<cudaStream_t> encodeRowsOnDevice(const std::vector<Plaintext*>& pts, const void* rows, int dtype, size_t n,
                                             size_t width, int limbs, double scale, int deg);

}  // namespace FIDESlib::CKKS
#endif
