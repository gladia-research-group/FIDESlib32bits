#pragma once
// Aggregated key switching for the first CoeffsToSlots stage (eprint 2025/1403, built here on top of
// FIDESLIB_BTS_SHIFT, 2026-10-04).
//
// Right after ModRaise the ciphertext (c0, c1) has small integer coefficients (|c| <= q0/2), so the first CtS
// matrix-vector product can be done with GHS (single-digit) key switching at NO precision cost: c1 is extended to
// the special primes EXACTLY (Garner lift, CKKS/SmallInt.cuh) instead of by base conversion, and every diagonal i
// gets its own key K_i = Enc_s(round(P * m_i * s^{(rot_i)} / D)) — the plaintext diagonal m_i AND the rescale by
// D = sf(L) baked in. Stage 0 is then: one lift, r permute-and-dot passes summed in Q+P (one fused kernel, the
// keys streamed once, the permuted c1 gathered from L2), ONE ModDown, plus sum_i m_i rot_i(c0) rescaled exactly
// while keeping the level. The output is a canonical level-L ciphertext: with the shift this stage consumes no
// level at all (two composite levels fewer than the shipped flow), and its key stream is 1/dnum of the hoisted
// baby steps'. Keys are generated on the GPU from the secret key (LoadAksKeys), setup only.
#include <cstdint>
#include <memory>
#include <vector>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Plaintext.cuh"

namespace FIDESlib::CKKS {
class RNSPoly;
class Ciphertext;

struct AksStage {
    int r = 0;
    std::vector<int> rot, kinv;  // rotation per diagonal (normalised) and 5^rot mod 2N (gather index)
    long double D = 1;           // effective divisor baked into the keys: sf_pt / (c 2^t)
    int t = 0;                   // the output carries 2^t in its scale (stage-1 plaintexts carry 2^-t)
    std::vector<std::unique_ptr<RNSPoly>> a, b;  // GHS keys over Q+P, NTT form, u32 limbs
    std::vector<std::unique_ptr<Plaintext>> m;   // diagonals at the stage level, un-pre-rotated
    void** keyTab = nullptr;  // device: [i][a|b][limb], limb over Q then P
    void** ptTab = nullptr;   // device: [i][limb over Q]
    int* kinvDev = nullptr;
    ~AksStage();
};

/// Build the stage-0 AKS keys for every bootstrap precomputation of the context (needs FIDESLIB_BTS_SHIFT >= 1).
void GenerateAksStage0(Context& cc_, BootstrapPrecomputation& pre, const std::vector<std::vector<uint64_t>>& skLimbs,
                       const std::vector<uint64_t>& skModuli, const std::vector<std::vector<uint64_t>>& sparseLimbs,
                       uint64_t seed);
/// A one-key AksStage (identity rotation, no plaintext) from host polynomials over Q+P in NTT form:
/// a[limb][coef], b[limb][coef], limbs ordered Q_0..Q_L then the special primes.
std::shared_ptr<AksStage> MakeGhsKeyStage(Context& cc_, const std::vector<std::vector<uint64_t>>& a,
                                          const std::vector<std::vector<uint64_t>>& b,
                                          const std::vector<uint64_t>& moduli);
/// Key-switch a SMALL-coefficient ciphertext (|coef| < q0, e.g. right after ModRaise) with a GHS key: exact lift of
/// c1 to Q+P (k = 3 limbs), one dot, one ModDown. Noise = ModDown rounding + c1·e/P (negligible for small c1).
void ghsSwitchSmall(Ciphertext& ct, AksStage& key);
/// Same switch when ct.c1 is ALREADY extended to Q+P (NTT form, isModUp): one dot, one ModDown per output.
void ghsSwitchExt(Ciphertext& ct, AksStage& key);
/// Replace LinearTransform for CtS stage 0; ct must be the raised ciphertext (level L, NoiseLevel 1).
void LinearTransformAKS(Ciphertext& ct, BootstrapPrecomputation::LTstep& step, AksStage& aks);
}  // namespace FIDESlib::CKKS
