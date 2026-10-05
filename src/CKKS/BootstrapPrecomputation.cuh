//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include <memory>
#include <vector>
#include "Plaintext.cuh"

namespace FIDESlib::CKKS {

class KeySwitchingKey;
struct AksStage;

class BootstrapPrecomputation {
   public:
    struct {
        int slots = -1;
        int bStep = -1;
        std::vector<Plaintext> A;
        std::vector<Plaintext> invA;
    } LT;

    struct LTstep {
        int slots = -1;
        int bStep = -1;
        int gStep = -1;
        std::vector<Plaintext> A;
        std::vector<int> rotIn;
        std::vector<int> rotOut;
    };

    std::vector<LTstep> StC;
    std::vector<LTstep> CtS;
    int accumulate_bStep = 4;
    uint32_t correctionFactor;
    bool sparse_encaps{false};
    std::weak_ptr<ContextData> sparse_context;
    // COMPOSITESCALING (d>1): the sparse-encapsulation raise cannot run in the single-tower
    // helper context — the composite bottom spans d limbs, and the helper's digit tables are
    // only filled for k < L(=1), so the M-4 keyswitch reads unfilled num_primeid_digit_to
    // entries and launches with a garbage grid ("invalid configuration argument",
    // LimbPartitionMGPU modup NTTs). Mirror the OpenFHE-side bootstrap patch instead:
    // STANDARD hybrid keyswitch in the MAIN context for both directions. The keys live here
    // rather than in the (ctxA,ctxB)-keyed secret-switching registry because both directions
    // are (main,main) with the same keyID there — the second emplace is silently dropped and
    // the M-4 fetch would return the M-2 key.
    std::unique_ptr<KeySwitchingKey> sparse_atob;  // dense -> sparse (M-4), applied at the composite bottom
    std::unique_ptr<KeySwitchingKey> sparse_btoa;  // sparse -> dense (M-2), applied at the raised level
    std::shared_ptr<AksStage> aks0;  // aggregated key switching for CtS stage 0 (CKKS/AksKeys.cuh), optional
    std::shared_ptr<AksStage> ghs_btoa;  // GHS form of sparse_btoa (digit keys summed) for the small raised ciphertext
    // FIDESLIB_BTS_SHIFT: the post-raise EvalMod constant (pre / (k N) * btsPreScale) folded into the stage-0
    // plaintexts; 0 = not folded. Bootstrap skips its multScalar + rescale when its own constant matches.
    double cts0_const = 0;
    // FIDESLIB_BTS_SHIFT_T: the exact scaling is by cts0_const * 2^cts0_t (scale bookkeeping carries the 2^t; the
    // stage-0 plaintexts carry 2^-t) so the encapsulation switch's absolute noise (~2^9/coef) is 2^t below the message
    int cts0_t = 0;
    // diagnostics (BTS_TRACE_DIR): the un-shifted plaintext stages, for a within-run reference bootstrap
    std::vector<LTstep> CtS_orig, StC_orig;
};

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
