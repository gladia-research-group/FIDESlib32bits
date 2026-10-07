// Lever D (OverModRaise1) stage-0 price gate. See Omr.cuh.
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Omr.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/Bootstrap.cuh"  // last: it pulls pke/openfhe.h, whose debug.h defines a `duration` macro

using namespace FIDESlib::CKKS;

namespace {
template <class F>
double timeMs(F&& f, int iters) {
    f();  // warm
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}
}  // namespace

long FIDESlib::CKKS::omrStage0Bench(Ciphertext& any, int slots, int iters, std::ostream& os) {
    ContextData& cc = any.cc;
    Context& cc_ = any.cc_;
    auto& pre = cc.GetBootPrecomputation(slots);
    if (pre.CtS.empty())
        return -1;
    const int d = cc.compositeDegree();
    // a canonical raised ciphertext at the top level (values arbitrary), as ModRaise leaves it
    Ciphertext raised(cc_);
    raised.copy(any);
    if (raised.NoiseLevel == 2)
        raised.rescale();
    const uint32_t correction = pre.correctionFactor;
    ModRaise(raised, slots, correction, false, pre.sparse_encaps, 0.0, false);
    // the shipped order: multScalar + rescale, then stage 0 at L - d
    Ciphertext std52(cc_);
    std52.copy(raised);
    std52.multScalar(1.0 / (cc.GetBootK() * cc.N), false);
    std52.rescale();
    const int stage0L = pre.CtS.at(0).A.at(0).c0.getLevel();
    if (std52.getLevel() > stage0L)
        std52.dropToLevel(stage0L);
    os << "[omr] raised limbs " << raised.getLevel() + 1 << ", shipped stage 0 at limbs " << stage0L + 1 << ", diagonals "
       << pre.CtS.at(0).A.size() << " bStep " << pre.CtS.at(0).bStep << "\n";

    // A: shipped stage 0 (hoisted BSGS)
    std::vector<BootstrapPrecomputation::LTstep> one;
    one.push_back(std::move(pre.CtS.at(0)));
    const double tA = timeMs([&] { Ciphertext w(cc_); w.copy(std52); EvalLTStages(w, one, "omr-"); }, iters);
    const double tCopy = timeMs([&] { Ciphertext w(cc_); w.copy(std52); }, iters);
    // a 31-diagonal later stage (stage 1), for the 1/5/5/4 split price
    std::vector<BootstrapPrecomputation::LTstep> st1;
    st1.push_back(std::move(pre.CtS.at(1)));
    Ciphertext in1(cc_);
    in1.copy(std52);
    in1.dropToLevel(st1.at(0).A.at(0).c0.getLevel());
    const double tS1 = timeMs([&] { Ciphertext w(cc_); w.copy(in1); EvalLTStages(w, st1, "omr1-"); }, iters);
    pre.CtS.at(0) = std::move(one.at(0));
    pre.CtS.at(1) = std::move(st1.at(0));

    // B: OMR1 radix-2 PtMult-first stage 0 on the over-raised ciphertext: 3 x (PtMult at L, rescale to L - d) +
    // 2 un-hoisted rotations at L - d + 2 adds. Plaintexts: three stage-0 diagonals re-levelled to L (Q limbs only).
    std::vector<Plaintext> p3;
    for (int k = 0; k < 3; ++k) {
        Plaintext np = relevelPlaintext(cc_, cc, pre.CtS.at(0).A.at(k), (raised.getLevel() - stage0L) / d, 1.0);
        np.c0.freeSpecialLimbs();
        p3.push_back(std::move(np));
    }
    cudaDeviceSynchronize();
    const auto& st0 = pre.CtS.at(0);
    const int rotB = st0.rotIn.size() > 1 ? st0.rotIn[1] - st0.rotIn[0] : 1;  // a baby-step key
    const int rotG = st0.rotOut.size() > 1 ? st0.rotOut[1] : rotB;             // a giant-step key
    const double tB = timeMs(
        [&] {
            Ciphertext acc(cc_), tmp(cc_);
            acc.multPt(raised, p3.at(0), false);
            acc.rescale();
            for (int k = 1; k < 3; ++k) {
                tmp.multPt(raised, p3.at(k), false);
                tmp.rescale();
                tmp.rotate(k == 1 ? rotB : rotG, true);  // indices with full-band bootstrap keys (the model keys are banded)
                acc.add(tmp);
            }
        },
        iters);
    os << "[omr] shipped stage 0: " << tA - tCopy << " ms (copy " << tCopy << " subtracted); stage 1 (31 diag): " << tS1 - tCopy
       << " ms; OMR1 radix-2 PtMult-first stage 0: " << tB << " ms  => stage-0 delta " << (tB - (tA - tCopy)) << " ms\n";
    return 0;
}
