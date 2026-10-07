//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <vector>
#include "Plaintext.cuh"

namespace FIDESlib::CKKS {

class KeySwitchingKey;
/// Plaintext index mask the batched LT dot applies to its plaintext reads (CKKS/LimbPartitionBatch.cu); set per
/// CtS/StC stage by LtPtMaskScope, all ones everywhere else.
extern uint32_t g_ltPtMask;
/// FIDESLIB_LT_FUSED_RESCALE: whether the current LT stage's end may fold the following rescale into its ModDown
/// (false for the bootstrap's LAST stage, whose deg-2 output is the planner's landing convention).
extern bool g_ltFuseAllowed;
struct LtFuseScope {
    bool old;
    explicit LtFuseScope(bool a) : old(g_ltFuseAllowed) { g_ltFuseAllowed = a; }
    ~LtFuseScope() { g_ltFuseAllowed = old; }
};
struct LtPtMaskScope {
    uint32_t old;
    // FIDESLIB_LT_COMPACT_BYPASS=1 (read per scope): full reads even when the stage has a mask (in-process gate)
    explicit LtPtMaskScope(uint32_t m) : old(g_ltPtMask) {
        const char* e = std::getenv("FIDESLIB_LT_COMPACT_BYPASS");
        g_ltPtMask = (e && *e && *e != '0') ? 0xFFFFFFFFu : m;
    }
    ~LtPtMaskScope() { g_ltPtMask = old; }
};
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
        // FIDESLIB_LT_COMPACT: the stage's diagonals are periodic / block-constant in their stored (NTT) layout, so the
        // LT dot reads element (idx & ptMask) — the distinct values sit in a few cache lines (L2) instead of streaming
        // the full limb from DRAM. All ones = full read. Data is unchanged, so any path ignoring the mask stays exact.
        uint32_t ptMask = 0xFFFFFFFFu;
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
    // FIDESLIB_BTS_RAISE_DROP: the raise targets the modulus this many composite levels below the top (the route's
    // CtS/StC plaintexts are re-levelled to match): the whole bootstrap runs on fewer limbs and lands lower.
    int raise_drop = 0;
    // diagnostics (BTS_TRACE_DIR): the un-shifted plaintext stages, for a within-run reference bootstrap
    std::vector<LTstep> CtS_orig, StC_orig;
    // FIDESLIB_BTS_STC_FIRST (lever A, SOTA_2026-10-05: Lattigo "DecodeThenModUp" / slim order). SlotsToCoeffs runs on
    // the INPUT ciphertext, before the raise, so its stages see <= stc_first_entry + 1 limbs instead of the ~20-28 of
    // the post-EvalMod ciphertext, and the output of EvalMod is the bootstrap output (landing moves up by the StC
    // depth; the StC levels are paid on the input side). StC_first / LT_first hold the shipped StC plaintexts
    // (scaleDec included) re-levelled to the entry side: stage s at limb index stc_first_entry - s*d.
    std::vector<LTstep> StC_first;
    std::vector<Plaintext> LT_first;  // single-LT route: LT.invA re-levelled
    int stc_first_entry = -1;         // limb index the input is dropped to before the first stage; -1 = path off
    // The shipped StC plaintexts carry scaleDec = q0/sf0 ~ 2^deg, which the standard order applies AFTER EvalMod; fed
    // in before it, EvalMod's input would be 2^deg too large. The last StC_first stage carries 2^-deg instead and the
    // bootstrap's integer post-factor becomes 2^(correction + deg) = 2^CF.
    int stc_first_deg = 0;
    // sparse routes: EvalMod leaves the two real halves (a - b ; a + b) of z = a + ib in the 2*slots view (the shipped
    // StC_sparse is diag(U0, i U0) and the fold adds the two blocks); this mask (1+i ; 1-i) followed by
    // rotate(slots) + add turns them into (z ; z). Encoded at the top level, re-levelled to the landing on first use.
    std::unique_ptr<Plaintext> stc_first_mask;
    std::map<int, Plaintext> stc_first_mask_at;  // the mask re-levelled (exactly, on the GPU) to each landing seen
    // FIDESLIB_BTS_STC_FIRST=2: ModRaise's FLEXIBLEAUTO adjust (2^-correction, the scale re-nominalization) is folded
    // into the LAST StC_first stage, which then sits at limb index 2d-1 and ModRaise takes its `prescaled` path: the
    // entry needs one composite level less (dense 8 limbs, single-LT 4). The factor depends on the correction factor
    // of the call and on the input's true scale after the drop, so the last stage is built lazily per (CF, scale).
    int stc_first_mode = 0;
    std::map<std::pair<int, long long>, std::vector<LTstep>> stc_first_last;     // multi-stage routes
    std::map<std::pair<int, long long>, std::vector<Plaintext>> lt_first_last;   // single-LT route

    // FIDESLIB_BTS_SPARSE_B (lever B, eprint 2026/1023 Algorithms 4/5, the n > r/2 case): repetition-aware depth-1
    // CoeffToSlot / SlotToCoeff for a sparse route with n = 2*slots effective coefficients repeated r = N/n times.
    // CtS = sum_{i<s} p_i (.) rot_i(ct) (hoisted BSGS over s = 2n/r diagonals), block partial sum over r/4 blocks of
    // s, + conj; StC = sum_i q_i (.) rot_i(ct), block partial sum over r/2 blocks of n, + rot(n/2). Each is ONE
    // level instead of the shipped 4 (CtS) and 3 (StC). The auxiliary slots of this layout need K = 24 for EvalMod
    // (its own Chebyshev series, same depth). Plaintexts are host-encoded at the top level and re-levelled + lifted
    // to the special primes on first use at each level.
    struct SparseB {
        int n = 0, r = 0, s = 0, bStep = 0, gStep = 0;
        std::vector<Plaintext> P, Q;  // BSGS pre-rotated diagonals, top level, Q limbs only
        std::map<int, std::vector<Plaintext>> P_at, Q_at;
        double bootK = 24.0;
        // FIDESLIB_BTS_SHIFT: the CtS diagonals carry 2^-t in value AND in their scale bookkeeping (as the shipped stage 0),
        // undoing the 2^t the exact post-raise scaling puts on both of the ciphertext's
        double ctsNF = 1.0;
        std::vector<double> cheb;
        int daIts = 5;
    };
    std::unique_ptr<SparseB> sparseB;
};

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
