#ifndef FIDESLIB_CKKS_CCMM_CUH
#define FIDESLIB_CKKS_CCMM_CUH
// Batch ciphertext-ciphertext matrix products on SinC-encoded inputs (Cheon-Kang-Lee, ePrint 2025/1957, Alg. 2-4)
#include <cstdint>
#include <memory>
#include <vector>

#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

struct CcmmKeys;

// primes {q0..q3, P}; NTT tables for n = 2^16, 2^13, 64; keys in coefficient form [digit][prime][ab][n]
struct CcmmKeySource {
    uint32_t p[5];
    std::vector<uint32_t> psi[3][5], ipsi[3][5];
    uint32_t ninv[3][5];
    std::vector<uint32_t> fwd, back, autos, rlk;
};

std::shared_ptr<CcmmKeys> ccmmUploadKeys(const CcmmKeySource& src);

// 16 4-limb ciphertexts per matrix; ccmmMul: outs = M M' (mpTransposed: insMp hold M'^T); times in ms (5 phases)
void ccmmTranspose(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, CcmmKeys& keys,
                   double* times = nullptr);
void ccmmMul(const std::vector<Ciphertext*>& insM, const std::vector<Ciphertext*>& insMp,
             const std::vector<Ciphertext*>& outs, CcmmKeys& keys, double* times = nullptr, bool mpTransposed = false);

}  // namespace FIDESlib::CKKS
#endif
