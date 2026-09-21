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
};

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
