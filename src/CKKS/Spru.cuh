#ifndef FIDESLIB_CKKS_SPRU_CUH
#define FIDESLIB_CKKS_SPRU_CUH
// SPRU bootstrapping (Coron & Koestler, arXiv 2607.27401, "Low-Latency Bootstrapping for CKKS using Roots of Unity"),
// Section 5 with n = 2 components: the single complex slot of the s = 1 route (plaintext m0 + m1 X^{N/2}).
// The input is switched at the composite bottom to a binary block key s' (h blocks of B = N/h, one 1 per block,
// s'_0 = 1); each component m_a = <s', C_a> mod q0 is evaluated in the exponent:
//   exp(2 pi i m_a / q0) = prod_b sum_j s'_{bB+j} exp(2 pi i C_{bB+j,a} / q0)
// with the key bits packed in 2n = 4 ciphertexts (cs_u), the candidate roots of unity in 4 plaintexts (E_u), the
// block sums by a trace (Accumulate over stride h n), the product over the h blocks by log2(h) rotate-multiply steps,
// the imaginary part by a conjugation, and the two components recombined into z = m0 + i m1 in every slot.
#include <memory>
#include <ostream>
#include <vector>
#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {

struct SpruKey;

/// Build the SPRU material for the s = 1 route: the block key, the dense -> block switching key, the 4 bootstrapping
/// key ciphertexts at the top level, and the rotation keys (trace, product, recombination). `h` = blocks (power of 2).
std::shared_ptr<SpruKey> spruSetup(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                   const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, Context& GPUcc, int h);

struct SpruTimes {
    double adjust_ms = 0, switch_ms = 0, host_ms = 0, encode_ms = 0, extmult_ms = 0, trace_ms = 0, product_ms = 0,
           finish_ms = 0, total_ms = 0;
};

/// Bootstrap `ct` (a ciphertext whose slots all hold the same complex value) in place. `correction`: the route's
/// correction factor minus deg (as in Bootstrap()). `times`: optional per-phase wall (device syncs between phases).
void spruBootstrap(Ciphertext& ct, SpruKey& key, uint32_t correction, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                   SpruTimes* times = nullptr);

}  // namespace FIDESlib::CKKS
#endif
