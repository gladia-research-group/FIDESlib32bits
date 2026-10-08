#ifndef FIDESLIB_CKKS_BOOTSTRAPSTAGES_CUH
#define FIDESLIB_CKKS_BOOTSTRAPSTAGES_CUH
// The GPU bootstrap as a sequence of stages, for an external (Python) orchestration. The stages reproduce the
// production path of Bootstrap() (Bootstrap.cu: bootstrapImpl) call for call, without its diagnostic branches
// (stage stashes, in-model A/B, within-run references), so a caller running
//   begin -> [stc_first_input] -> mod_raise -> fold -> cts_stage(0..n_cts-1) -> eval_mod ->
//   ( stc_first_output | stc_enter -> stc_stage(0..n_stc-1) -> finish )
// gets a bit-identical result. g_bootstrapOverride, when set, replaces Bootstrap() itself.
#include <cstdint>
#include <functional>
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

struct BtsState {
    int slots = 0, oldSlots = 0;
    bool prescaled = false, isLT = false, mixedChain = false, sparseEncaps = false, sparseB = false;
    bool stcFirst = false, stcFolded = false, exactOnly = false, exactConst = false, aksOn = false, shiftedFlow = false;
    bool real = false;  // real payload on the dense route (ContextData::btsRealPayload)
    uint32_t correction = 0;
    uint64_t scalar = 1, corFactor = 1;
    double constantEvalMult = 0, baked = 0;
    int nCtS = 0, nStC = 0;
    // FIDESLIB_EVALROUND (Kim et al., ePrint 2022/1256): EvalMod returns the integer part K*x - EvalMod(x), StC maps it to
    // q0*I, and the output is gamma*raised - StC(...): CtS rounding error enters both K*x and EvalMod(x) and cancels.
    bool evalRound = false;
    double erGamma = 1.0;
    std::shared_ptr<Ciphertext> erRaised;
};

BtsState btsBegin(Ciphertext& ctxt, int slots, bool prescaled, bool allowStcFirst = true);
void btsStcFirstInput(Ciphertext& ctxt, const BtsState& st);
void btsModRaise(Ciphertext& ctxt, BtsState& st);
void btsFold(Ciphertext& ctxt, const BtsState& st);
void btsCtSStage(Ciphertext& ctxt, const BtsState& st, int k);
void btsEvalMod(Ciphertext& ctxt, const BtsState& st);
void btsStcFirstOutput(Ciphertext& ctxt, const BtsState& st);
void btsStCEnter(Ciphertext& ctxt, const BtsState& st);
void btsStCStage(Ciphertext& ctxt, const BtsState& st, int k);
void btsFinish(Ciphertext& ctxt, const BtsState& st);
/// The stage sequence above, in C++ (reference for the external orchestration).
void BootstrapStaged(Ciphertext& ctxt, int slots, bool prescaled);

/// When set, Bootstrap(ctxt, slots, prescaled) calls this instead of its own body (not re-entered from inside).
extern std::function<void(Ciphertext&, int, bool)> g_bootstrapOverride;

}  // namespace FIDESlib::CKKS
#endif
