#ifndef FIDESLIB_CKKS_OMR_CUH
#define FIDESLIB_CKKS_OMR_CUH
#include <ostream>
namespace FIDESlib::CKKS {
class Ciphertext;
// Lever D price gate (eprint 2025/1298 OverModRaise1, SOTA_2026-10-05.md): wall of the shipped hoisted BSGS CtS stage 0
// (31 diagonals, 16 baby steps) against a PtMult-first radix-2 stage 0 (3 diagonals: 3 plaintext products + 3 rescales of
// the over-raised ciphertext + 2 un-hoisted rotations + 2 adds), emulated on a ciphertext raised to the top level with the
// shipped kernels; also the wall of one 31-diagonal later stage, to price the denser 1/5/5/4 split. Values are not checked.
// Returns 0, or -1 when the route has no multi-stage CtS.
long omrStage0Bench(Ciphertext& any, int slots, int iters, std::ostream& os);
}
#endif
