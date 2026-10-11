#ifndef FIDESLIB_CKKS_COEFF_ROWS_CUH
#define FIDESLIB_CKKS_COEFF_ROWS_CUH
// Plaintext-weight inner sums from coefficient rows: sum_r cts[r] * pt_r where pt_r is given by its polynomial
// coefficients divided by the scaling factor (one float row of N values per plaintext), so the encode is a rounding,
// a reduction into each limb and an NTT: no FFT, no host work. Built for streamed weight matrices (a BSGS linear's
// baby-step sums): every row is read once, the plaintexts never exist as objects.
#include <cstddef>
#include <vector>
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

/// out = sum_r cts[r] * pt_r with pt_r[k] = round(rows[r * rowStride + k] * scale) (k < N), NoiseLevel 1, at the
/// ciphertexts' level. `rows` is device memory (read in place) or pinned host memory (moved by the copy engine,
/// double-buffered so a chunk's transfer and encode overlap the previous chunk's products). Every ciphertext is
/// NoiseLevel 1 at one level; `out` holds a ciphertext at that level whose data is overwritten. Metadata as a
/// multPt / addMultPt chain sets it: NoiseLevel 2, NoiseFactor = cts[0].NoiseFactor * scale. No host sync. Rows
/// are processed FIDESLIB_COEFF_CHUNK (default 16) at a time.
void multCoeffRowsSum(Ciphertext& out, const std::vector<const Ciphertext*>& cts, const float* rows, size_t rowStride,
                      double scale);

/// Coefficients whose rounding left the int64 range since the start (clamped; nonzero means a wrong result).
unsigned long long coeffRowsOverflows();

}  // namespace FIDESlib::CKKS
#endif
