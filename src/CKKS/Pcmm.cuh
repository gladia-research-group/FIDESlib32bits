#ifndef FIDESLIB_CKKS_PCMM_CUH
#define FIDESLIB_CKKS_PCMM_CUH
// Plaintext-ciphertext matrix products at the StC-first split point of a bootstrap (BCHPS, ePrint 2024/1284;
// Sylph, arXiv 2601.18511): ring switch to 2^12, MLWE pieces of 2t tokens, exact int8 GEMM mod q, pack back.
#include <cstdint>
#include <memory>
#include <vector>

#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

struct PcmmKeys;

// Primes {q0, q1, P}, the four split moduli, negacyclic NTT tables (psi^brev, psi^-brev, n^-1) for n = 2^16 and 2^12,
// and the keys in coefficient form: fwd / back [2][3][2][2^16], pack [r][2][3][2][2^12] (r = 2048 / t).
struct PcmmKeySource {
    uint32_t q[2], P;
    uint32_t split[4];
    std::vector<uint32_t> psi16[3], ipsi16[3], psi12[3], ipsi12[3];
    uint32_t ninv16[3], ninv12[3];
    std::vector<uint32_t> fwd, back, pack;
};

std::shared_ptr<PcmmKeys> pcmmUploadKeys(const PcmmKeySource& src);

// Channels per ciphertext (16 r).
int pcmmChannels(const PcmmKeys& keys);

// In place on one split ciphertext; Wp: nch x nch int16 in the split point's channel order. times: 6 step timings (ms).
void pcmmAtSplit(Ciphertext& ct, PcmmKeys& keys, const int16_t* Wp, double* times = nullptr);

// ins -> outs (nch-channel blocks; outs may alias ins), W: (nch n_out) x (nch n_in) int16.
void pcmmMulti(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, PcmmKeys& keys,
               const int16_t* W, double* times = nullptr);

// The same with W as nW balanced base-256 int8 digit planes (W = sum_u Wd[u] 256^u).
void pcmmMultiPlanes(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, PcmmKeys& keys,
                     const int8_t* Wd, int nW, double* times = nullptr);

}  // namespace FIDESlib::CKKS
#endif
