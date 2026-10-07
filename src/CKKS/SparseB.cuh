#ifndef FIDESLIB_CKKS_SPARSEB_CUH
#define FIDESLIB_CKKS_SPARSEB_CUH
// Lever B (eprint 2026/1023, Algorithms 4/5): repetition-aware depth-1 CoeffToSlot / SlotToCoeff for a sparse
// bootstrap route (n = 2*slots effective coefficients repeated r = N/n times, n > r/2). See
// BootstrapPrecomputation::SparseB for the layout; the plaintext vectors are built by the precomputation loader
// (openfhe-interface/RawCiphertext.cu, FIDESLIB_BTS_SPARSE_B).
#include <vector>
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {
class BootstrapPrecomputation;

/// CtS on the folded raised ciphertext (standard order): BSGS over the s diagonals, block partial sum over r/4 blocks
/// of s, + conj. Leaves the ciphertext at NoiseLevel 2 (one level consumed after the rescale).
void SparseBCoeffsToSlots(Ciphertext& ctxt, int slots, BootstrapPrecomputation& pre);
/// StC on the EvalMod output: BSGS over the s diagonals, block partial sum over r/2 blocks of n, + rot(n/2).
void SparseBSlotsToCoeffs(Ciphertext& ctxt, int slots, BootstrapPrecomputation& pre);
/// The K = 24 EvalMod series of this layout (degree 15, r = 5 double angles, same depth as the shipped K = 16 one).
const std::vector<double>& sparseBChebyshevK24();
/// Unit gate: run only the CtS (stc = false) or StC (true) of the route on `ct` (fetches the precomputation itself).
/// Returns {n, r, s} of the route, or n = 0 when the route has no lever-B precomputation.
void sparseBUnit(Ciphertext& ct, int slots, bool stc, int& n, int& r, int& s);
}  // namespace FIDESlib::CKKS
#endif
