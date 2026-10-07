//
// Created by carlosad on 4/12/24.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include <cmath>
#include <iostream>
#include "CKKS/Bootstrap.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/AksKeys.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/SparseB.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

// Stage-divergence harness (default off): when a caller installs a stash vector, every
// btsStageProbe checkpoint (pre-CtS / post-CtS / pre-StC / post-StC / end) also deposits a
// full ciphertext clone the caller can download+decrypt offline. Zero cost when null.
std::vector<std::pair<std::string, std::shared_ptr<FIDESlib::CKKS::Ciphertext>>>*
    FIDESlib::CKKS::g_btsStageStash = nullptr;


// Effective correction factor for this bootstrap call: the ContextData override (armed by
// the wrapper's CorrectionScope) wins over the per-slots precomputation value.
static uint32_t effCorrectionFactor(FIDESlib::CKKS::ContextData& cc, int slots) {
    return cc.correctionFactorOverride >= 0
               ? (uint32_t)cc.correctionFactorOverride
               : cc.GetBootPrecomputation(slots).correctionFactor;
}

static void btsStageProbe(const char* stage, FIDESlib::CKKS::Ciphertext& ctxt) {
    if (FIDESlib::CKKS::g_btsStageStash) {
        cudaDeviceSynchronize();
        auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
        c->copy(ctxt);
        FIDESlib::CKKS::g_btsStageStash->emplace_back(stage, std::move(c));
    }
}

void FIDESlib::CKKS::BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // deg = q0_bits - scale_bits must not exceed the correction factor, or the uint32
    // subtraction below underflows and corFactor = 1 << garbage silently poisons the bootstrap.
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L - cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop) / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////

    {

        ModRaise(ctxt, slots, correction, prescaled);

        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        double constantEvalMult = pre * (1.0 / (k * cc.N));
        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        constantEvalMult *= cc.getBtsPreScale();

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    btsStageProbe("pre-CtS", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();

        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    uint64_t corFactor = (uint64_t)1 << std::llround(correction);

    btsStageProbe("pre-StC", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (corFactor != 1)
        multIntScalar(ctxt, corFactor);
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
}

namespace FIDESlib::CKKS {

static void bootstrapImpl(Ciphertext& ctxt, const int slots, const bool prescaled, const bool allowStcFirst);

std::function<void(Ciphertext&, int, bool)> g_bootstrapOverride;

void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    static thread_local bool inOverride = false;
    if (g_bootstrapOverride && !inOverride) {
        inOverride = true;
        try {
            g_bootstrapOverride(ctxt, slots, prescaled);
        } catch (...) {
            inOverride = false;
            throw;
        }
        inOverride = false;
        return;
    }
    bootstrapImpl(ctxt, slots, prescaled, true);
}

// FIDESLIB_BTS_STC_FIRST_AB=<dir>: in-model A/B of the StC-first order — the first few bootstraps that take it also run the
// shipped order on a copy of the SAME input; input and both outputs are exact-decrypted (FIDESLIB_DIAG_SK=1) into <dir>
// for the offline comparison (logs/bts_traffic/stcfirst_ab.py).
static void bootstrapImpl(Ciphertext& ctxt, const int slots, const bool prescaled, const bool allowStcFirst) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    assert(slots >= ctxt.slots);
    int old_slots = ctxt.slots;

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // deg = q0_bits - scale_bits must not exceed the correction factor, or the uint32
    // subtraction below underflows and corFactor = 1 << garbage silently poisons the bootstrap.
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L - cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop) / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;
    bool shiftedFlow = false;  // FIDESLIB_BTS_SHIFT path taken (exact scaling, +d level)

    // FIDESLIB_BTS_STC_FIRST (lever A, slim order): SlotsToCoeffs on the INPUT ciphertext, at <= entry + 1 limbs,
    // before the raise. The raised polynomial then carries (Re z ; Im z) as coefficients and EvalMod's output is the
    // bootstrap output. The StC levels are paid from the input (entry = 2d-1 limbs for ModRaise's adjust+rescale plus
    // d per StC stage), the landing moves up by the same amount. Sparse routes: the shipped StC_sparse is
    // diag(U0, i U0); the fold turns its output into (1+Y^s) Q(Y) / 2, i.e. the halves (a-b ; a+b)/2 come out of
    // EvalMod, and the mask (1-i ; 1+i) + rotate(slots) recombines z = a + ib (see BootstrapPrecomputation.cuh).
    // lever B route (standard order); FIDESLIB_BTS_SPARSE_B_RUN=0 keeps the precomputation but runs the shipped path
    // (calibration: the same context, both flows)
    const bool sparseB = cc.GetBootPrecomputation(slots).sparseB != nullptr &&
                         [] { const char* e = std::getenv("FIDESLIB_BTS_SPARSE_B_RUN"); return !(e && std::atoi(e) == 0); }();
    const bool stcFirst = cc.GetBootPrecomputation(slots).stc_first_mode > 0 && !sparseB && allowStcFirst;
    static int abCount = 0;
    std::unique_ptr<Ciphertext> abRef;
    std::string abDir;
    if (stcFirst) {
        if (const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_AB"); e && *e && abCount < 8) {
            abDir = e;
            const std::string base = abDir + "/" + std::to_string(abCount) + "-s" + std::to_string(slots);
            cudaDeviceSynchronize();
            exactDecryptDump(ctxt, (base + "-in.ct").c_str());
            abRef = std::make_unique<Ciphertext>(cc_);
            abRef->copy(ctxt);
            bootstrapImpl(*abRef, slots, prescaled, false);  // the shipped order on the same input
            cudaDeviceSynchronize();
            exactDecryptDump(*abRef, (base + "-std.ct").c_str());
            std::cerr << "[stc_first_ab] #" << abCount << " slots=" << slots << " in level " << ctxt.getLevel() << " deg "
                      << ctxt.NoiseLevel << " NF " << ctxt.NoiseFactor << " cf " << effCorrectionFactor(cc, slots)
                      << " -> std out level " << abRef->getLevel() << " deg " << abRef->NoiseLevel << "\n";
        }
    }
    const bool stcFolded = cc.GetBootPrecomputation(slots).stc_first_mode == 2;  // last stage carries the adjust
    if (stcFirst) {
        auto& pre = cc.GetBootPrecomputation(slots);
        if (ctxt.NoiseLevel == 2)
            ctxt.rescale();
        const int entry = pre.stc_first_entry;
        if (ctxt.getLevel() < entry)
            throw std::runtime_error("[stc_first] bootstrap input at limb index " + std::to_string(ctxt.getLevel()) +
                                     " is below the StC-first entry level " + std::to_string(entry) +
                                     " (the input must keep " + std::to_string(entry + 1) + " limbs)");
        if (ctxt.getLevel() > entry)
            ctxt.dropToLevel(entry);
        ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;  // the 2*slots view the sparse StC stages are indexed in
        btsStageProbe("pre-StC1", ctxt);
        if (!stcFolded) {
            if (isLT)
                EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, pre.LT_first);
            else
                EvalLTStages(ctxt, pre.StC_first, "StC1-stage-");
        } else {
            // The last stage's plaintexts carry 2^-deg (scaleDec compensation), 2^-correction and the FLEXIBLEAUTO
            // re-nominalization, so that after multPt + ModRaise's prescaled rescale the integers are v*targetSF*2^-c:
            // factor = 2^-CF * (targetSF / sf(d-1)) * (sf(entry) / NF_in)   [NF_in = true scale after the drop; the
            // stages before the last multiply NF by sf(l)/sf(entry), their rescales telescope]
            const int d = cc.compositeDegree();
            const int lastL = 2 * d - 1;
            const double nfIn = ctxt.NoiseFactor;
            const double targetSF = cc.sfAtLimb(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) * d - d * pre.raise_drop);
            const int cf = (int)effCorrectionFactor(cc, slots);
            // The input-scale ratio MUST ride the plaintext (before the raise): folding it into the exact post-raise
            // scaling also scales the q0*I term, which EvalMod needs as integer multiples (chain96/97: garbage).
            const double ratio = cc.sfAtLimb(entry) / nfIn;
            const double factor = std::ldexp(1.0, -cf) * (targetSF / cc.sfAtLimb(d - 1)) * ratio;
            const std::pair<int, long long> key{cf, std::llround(std::log2(nfIn) * 1e6)};  // nominal scales per level
            // The cached last stages must keep their special limbs (the Q-only variant lost 15 bits, chain93); the
            // cache is bounded (each entry ~0.2 GB; a plan's distinct (CF, input level) pairs are a few tens).
            constexpr size_t kLastStageCache = 24;
            if (isLT) {
                auto it = pre.lt_first_last.find(key);
                if (it == pre.lt_first_last.end()) {
                    if (pre.lt_first_last.size() >= kLastStageCache) {
                        cudaDeviceSynchronize();  // the evicted plaintexts may still be read by the previous bootstrap's kernels
                        pre.lt_first_last.erase(pre.lt_first_last.begin());
                    }
                    std::vector<Plaintext> v;
                    v.reserve(pre.LT.invA.size());
                    for (auto& pt : pre.LT.invA)
                        v.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, factor));
                    cudaDeviceSynchronize();
                    it = pre.lt_first_last.emplace(key, std::move(v)).first;
                }
                EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, it->second);
            } else {
                if (!pre.StC_first.empty())
                    EvalLTStages(ctxt, pre.StC_first, "StC1-stage-");
                auto it = pre.stc_first_last.find(key);
                if (it == pre.stc_first_last.end()) {
                    if (pre.stc_first_last.size() >= kLastStageCache) {
                        cudaDeviceSynchronize();  // the evicted plaintexts may still be read by the previous bootstrap's kernels
                        pre.stc_first_last.erase(pre.stc_first_last.begin());
                    }
                    auto& st = pre.StC.back();
                    BootstrapPrecomputation::LTstep o;
                    o.slots = st.slots; o.bStep = st.bStep; o.gStep = st.gStep; o.rotIn = st.rotIn; o.rotOut = st.rotOut;
                    o.A.reserve(st.A.size());
                    for (auto& pt : st.A)
                        o.A.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, factor));
                    std::vector<BootstrapPrecomputation::LTstep> v;
                    v.push_back(std::move(o));
                    cudaDeviceSynchronize();
                    it = pre.stc_first_last.emplace(key, std::move(v)).first;
                }
                EvalLTStages(ctxt, it->second, "StC1-last-");
            }
            // ModRaise's prescaled path expects the deg-2 ciphertext at limb index 2d-1
            if (ctxt.getLevel() != lastL || ctxt.NoiseLevel != 2)
                throw std::runtime_error("[stc_first] folded last stage left the ciphertext at limb index " +
                                         std::to_string(ctxt.getLevel()) + " deg " + std::to_string(ctxt.NoiseLevel));
        }
        btsStageProbe("post-StC1", ctxt);
        if (const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_DIAG"); e && std::atoi(e) > 0) {
            static int once = 0;
            if (once++ < 2)
                std::cerr << "[stc_first] diag: entry limb " << entry << " sf(entry) " << cc.sfAtLimb(entry) << " sf(L) "
                          << cc.sfAtLimb(cc.L) << " post-StC level " << ctxt.getLevel() << " NoiseLevel " << ctxt.NoiseLevel
                          << " NF " << ctxt.NoiseFactor << " sf(level) " << cc.sfAtLimb(ctxt.getLevel())
                          << " sf(level-d) " << cc.sfAtLimb(ctxt.getLevel() - cc.compositeDegree()) << " q0 "
                          << qDouble << " deg " << deg << " corr " << correction << "\n";
        }
    }

    {
        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();
        double constantEvalMult = pre * (1.0 / (k * cc.N));
        constantEvalMult *= cc.getBtsPreScale();
        // FIDESLIB_BTS_SHIFT: apply it exactly inside the raise (level kept) when the precomputation was built for it
        const double baked = cc.GetBootPrecomputation(slots).cts0_const;
        // diagnostic: FIDESLIB_BTS_EXACT_ONLY=1 = exact scaling inside the raise, then an exact LevelReduce to the
        // standard level, with the standard (unshifted) plaintexts — isolates the scaling step from the +d flow
        const bool exactOnly = [] { const char* e = std::getenv("FIDESLIB_BTS_EXACT_ONLY"); return e && std::atoi(e) > 0; }();
        const bool exactConst = exactOnly || (baked != 0 && std::fabs(baked / constantEvalMult - 1.0) < 1e-9);
        if (baked != 0 && !exactConst)
            std::cerr << "[bts_shift] WARNING: baked constant " << baked << " != runtime " << constantEvalMult
                      << " (or sparse slots): falling back to multScalar, the level gain is lost\n";
        const bool aksOn = exactConst && baked != 0 && cc.GetBootPrecomputation(slots).aks0 != nullptr;
        ModRaise(ctxt, slots, correction, prescaled || stcFolded, sparse_encaps, exactConst ? constantEvalMult : 0.0, aksOn);
        shiftedFlow = exactConst && baked != 0 && !aksOn;
        if (exactOnly && baked == 0 && exactConst)
            ctxt.dropToLevel(ctxt.getLevel() - cc.compositeDegree());  // standard plaintexts expect L - d
        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)

        // TO-DO: The 1/32 scale will be pre-applied with OpenFHE v1.4, so remove it from here

        /*
        if (sparse_encaps) {
            constantEvalMult = pre * (1.0 / (k * cc.N) / 32);
        }
        */

        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        if (!exactConst)
            ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    btsStageProbe("pre-CtS", ctxt);
    if (g_btsStageStash && shiftedFlow && !cc.GetBootPrecomputation(slots).CtS_orig.empty() &&
        cc.GetBootPrecomputation(slots).cts0_t == 0) {
        // within-run reference: the same raised ciphertext through the STANDARD flow (level dropped exactly by d,
        // original plaintexts swapped in); stages probed as R-*
        auto& pre = cc.GetBootPrecomputation(slots);
        Ciphertext ref(cc_);
        ref.copy(ctxt);
        ref.dropToLevel(ref.getLevel() - cc.compositeDegree());
        std::swap(pre.CtS, pre.CtS_orig);
        std::swap(pre.StC, pre.StC_orig);
        btsStageProbe("R-pre-CtS", ref);
        EvalCoeffsToSlots(ref, slots, false);
        btsStageProbe("R-post-CtS", ref);
        {
            Ciphertext raux(cc_);
            raux.conjugate(ref);
            Ciphertext encI(cc_);
            encI.sub(ref, raux);
            ref.add(raux);
            encI.multMonomial(3 * 2 * cc.N / 4);
            approxModReduction(ref, encI, cc.GetEvalKey(ref.keyID), scalar);
        }
        if (ref.NoiseLevel == 2)
            ref.rescale();
        btsStageProbe("R-post-EvalMod", ref);
        EvalCoeffsToSlots(ref, slots, true);
        btsStageProbe("R-post-StC", ref);
        if (const uint64_t cf = (uint64_t)1 << std::llround(correction); cf != 1)
            multIntScalar(ref, cf);
        btsStageProbe("R-end", ref);
        std::swap(pre.CtS, pre.CtS_orig);
        std::swap(pre.StC, pre.StC_orig);
    }
    if (sparseB) {
        if (g_btsStageStash) {  // reference: the shipped sparse CtS (+ conj) and EvalMod on the SAME folded ciphertext
            Ciphertext ref(cc_), raux(cc_);
            ref.copy(ctxt);
            ref.slots = 2 * slots;
            EvalCoeffsToSlots(ref, slots, false);
            raux.conjugate(ref);
            ref.add(raux);
            btsStageProbe("post-CtS-std", ref);
            approxModReductionSparse(ref, scalar);
            if (ref.NoiseLevel == 2)
                ref.rescale();
            btsStageProbe("post-EvalMod-std", ref);
        }
        SparseBCoeffsToSlots(ctxt, slots, cc.GetBootPrecomputation(slots));
    } else if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else if (sparseB) {
        auto& sb = *cc.GetBootPrecomputation(slots).sparseB;
        approxModReductionSparse(ctxt, scalar, sb.cheb, sb.daIts);  // the conj-add is part of SparseBCoeffsToSlots
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, scalar);
    }

    // StC-first dense output stays a lazy deg-2 (the landing convention of every other route: the planner models a
    // deg-2 landing realized to deg-1 one level lower; a deg-1 output here broke the plan's degree pins)
    if (ctxt.NoiseLevel == 2 && !(stcFirst && cc.N / 2 == slots)) {
        ctxt.rescale();
    }

    btsStageProbe("post-EvalMod", ctxt);
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);

    if (stcFirst) {  // slim order: EvalMod's output is the result; the post-factor is 2^(correction + deg) = 2^CF
        const uint64_t cfAll = corFactor << cc.GetBootPrecomputation(slots).stc_first_deg;
        if (cfAll != 1)
            multIntScalar(ctxt, cfAll);
        if (cc.N / 2 != slots) {
            auto& pre = cc.GetBootPrecomputation(slots);
            const int lv = ctxt.getLevel();
            auto it = pre.stc_first_mask_at.find(lv);
            if (it == pre.stc_first_mask_at.end()) {
                const int sh = (lv - pre.stc_first_mask->c0.getLevel()) / cc.compositeDegree();
                it = pre.stc_first_mask_at.emplace(lv, relevelPlaintext(cc_, cc, *pre.stc_first_mask, sh, 1.0)).first;
            }
            ctxt.multPt(it->second, false);
            aux.rotate(ctxt, slots);
            ctxt.add(aux);
        }
        if (mixedChain && ctxt.NoiseLevel == 2)
            ctxt.rescale();
        btsStageProbe("end", ctxt);
        if (abRef) {
            cudaDeviceSynchronize();
            exactDecryptDump(ctxt, (abDir + "/" + std::to_string(abCount) + "-s" + std::to_string(slots) + "-slim.ct").c_str());
            std::cerr << "[stc_first_ab] #" << abCount << " slim out level " << ctxt.getLevel() << " deg " << ctxt.NoiseLevel
                      << " NF " << ctxt.NoiseFactor << "\n";
            ++abCount;
        }
        ctxt.slots = old_slots;
        return;
    }

    // A ciphertext above the StC plaintexts' level (FIDESLIB_BTS_SHIFT with FIDESLIB_BTS_SHIFT_STC=0) is dropped to
    // it first: an exact LevelReduce, instead of the plaintext-above-ciphertext adjust path.
    if (!isLT && !sparseB && !cc.GetBootPrecomputation(slots).StC.empty()) {
        const int stcL = cc.GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
        if (ctxt.getLevel() > stcL) {
            if (ctxt.NoiseLevel == 2)
                ctxt.rescale();
            if (ctxt.getLevel() > stcL)
                ctxt.dropToLevel(stcL);
        }
    }
    btsStageProbe("pre-StC", ctxt);
    if (sparseB) {
        SparseBSlotsToCoeffs(ctxt, slots, cc.GetBootPrecomputation(slots));  // includes the final rot(n/2) + add
    } else if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots && !sparseB) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (corFactor != 1)
        multIntScalar(ctxt, corFactor);
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }

    ctxt.slots = old_slots;
}
}  // namespace FIDESlib::CKKS

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
    ContextData& cc = *cc_;
    SetCurrentContext(cc_);
    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
    #if NATIVEINT != 128
        if (deg > static_cast<int32_t>(m_correctionFactor)) {
            OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                          std::to_string(m_correctionFactor) + "].");
        }
    #endif
        */
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;

    double res = 0.0;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        const int d_ = cc.compositeDegree();
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        double targetSF = cc.sfAtLimb(cc.L - lvl * d_);
        // composite: the pre-raise ciphertext sits at 2 LEVELS = 2d limbs; its scale lives at
        // limb 2d-1 and the adjust's rescale drops the top d primes (their product).
        double sourceSF = cc.sfAtLimb(2 * d_ - 1);  // ciphertext->GetScalingFactor();
        uint32_t numTowers = 2 * d_;                // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = cc.modReduceProduct(2 * d_ - 1);
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        res = adjustmentFactor;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        res = std::pow((double)2.0, (double)-1.0 * (double)correction);
    }

    return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled,
                              const bool sparse_encaps, const double exactScale, const bool aksStage0) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    ContextData& cc = ctxt.cc;
    btsStageProbe("MR-entry", ctxt);
    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    if (!prescaled) {
        assert(ctxt.getLevel() + 1 - cc.compositeDegree() * ctxt.NoiseLevel >= 1);
    } else {  // deg-1 at the bottom, or deg-2 one composite level above it
        assert(ctxt.getLevel() + 1 == cc.compositeDegree() * ctxt.NoiseLevel);
    }
    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial 2 ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        std::cout << correction << std::endl;
        std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }

    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        // the raise target's scale (FIDESLIB_BTS_RAISE_DROP: the target is raise_drop composite levels below the top)
        double targetSF = cc.sfAtLimb(cc.L - lvl * cc.compositeDegree() -
                                      cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);
        double sourceSF = ctxt.NoiseFactor;        // ciphertext->GetScalingFactor();
        uint32_t numTowers = ctxt.getLevel() + 1;  // ciphertext->GetElements()[0].GetNumOfElements();
        // composite: the adjust's rescale drops the top d primes — divide by their product
        double modToDrop = cc.modReduceProduct(ctxt.getLevel());
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        // multScalar encodes the scalar at the level's NOMINAL scale sf(level); the input's true scale is sourceSF.
        // integers v*sourceSF -> v*sourceSF*adj*sf(level)/modToDrop = v*targetSF*2^-c for ANY sourceSF (the original
        // (targetSF/sourceSF)*(modToDrop/sourceSF) is the sourceSF == sf(level) case; a ciphertext level-reduced
        // from a higher level, e.g. the StC-first input, is not nominal).
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / cc.sfAtLimb(ctxt.getLevel()));
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;

        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(adjustmentFactor);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            //cc->EvalMultInPlace(ciphertext, adjustmentFactor);
            ctxt.rescale();
            ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
        ctxt.NoiseFactor = targetSF;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.rescale();
            ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
    }

    btsStageProbe("MR-bottom", ctxt);
    if (sparse_encaps) {
        if (cc.compositeDegree() > 1) {
            // COMPOSITESCALING: M-4 as a STANDARD hybrid keyswitch in the MAIN context at the
            // composite bottom — the single-tower helper context cannot host a d-limb ct
            // (its digit tables stop at k=0; see BootstrapPrecomputation::sparse_atob).
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_atob);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;
            auto sparse_context_use = sparse_context.lock();
            Ciphertext sparse_ctxt(sparse_context_use);
            auto& atob = CKKS::GetSecretSwitchingKey(ctxt.cc_, sparse_context_use, ctxt.keyID);

            sparse_ctxt.reinterpretContext(ctxt);
            sparse_ctxt.keySwitch(atob);
            ctxt.reinterpretContext(sparse_ctxt);
        }
    }
    btsStageProbe("MR-atob", ctxt);

    //   std::cout << "Boot start " << std::endl;
    // auto ctxtDCRT = raised->GetElements();
    if constexpr (PRINT) {
        std::cout << "Adjustment 1: ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    ctxt.c0.INTT(cc.batch, true);

    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //   std::cout << "Grow" << std::endl;
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) -
                  cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);  // FIDESLIB_BTS_RAISE_DROP
    //   std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c0.compositeModRaise();
    else
        ctxt.c0.broadcastLimb0();
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c0.NTT(cc.batch, true);
    // std::cout << cc.batch << std::endl;
    if constexpr (PRINT) {
        std::cout << "ModRaise ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.INTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1 ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //  std::cout << "Grow" << std::endl;
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) -
                  cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);  // FIDESLIB_BTS_RAISE_DROP
    //  std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        std::cout << "Adjustment c1  ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c1.compositeModRaise();
    else
        ctxt.c1.broadcastLimb0();
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.NTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    btsStageProbe("MR-raised", ctxt);
    // FIDESLIB_BTS_SHIFT: the EvalMod constant as an exact scaling of the small-coefficient ciphertext, before the
    // encapsulation switch makes the coefficients uniform (CKKS/SmallInt.cuh). Replaces multScalar + rescale.
    bool fusedSwitched = false;
    if (exactScale > 0 && !aksStage0) {  // lever 1b: the aggregated stage 0 scales, switches and transforms at once
        const int tt = cc.GetBootPrecomputation(slots).cts0_t;
        const double tf = std::ldexp(1.0, tt);
        const double exactScaleT = exactScale * tf;  // the 2^t rides the scale bookkeeping (stage-0 plaintexts carry 2^-t)
        if (g_btsStageStash && sparse_encaps && cc.compositeDegree() > 1) {
            // trace (small-valued => faithful decode): the shipped path and the two scaled-then-switched variants
            auto& pre = cc.GetBootPrecomputation(slots);
            {
                Ciphertext w(ctxt.cc_);
                w.copy(ctxt);
                w.keySwitch(*pre.sparse_btoa);
                w.multScalar(exactScale, false);
                w.rescale();
                btsStageProbe("X-std", w);
            }
            Ciphertext sc(ctxt.cc_);
            sc.copy(ctxt);
            smallIntScalarMultiply(cc, sc, exactScaleT);
            sc.NoiseFactor *= tf;
            if (const char* td = std::getenv("BTS_TRACE_DIR")) {  // raw polynomials before/after the exact scaling
                exactCtPolyDump(ctxt, (std::string(td) + "/P-raw").c_str());
                exactCtPolyDump(sc, (std::string(td) + "/P-scaled").c_str());
            }
            btsStageProbe("X-scaled", sc);
            {
                Ciphertext h(ctxt.cc_);
                h.copy(sc);
                h.keySwitch(*pre.sparse_btoa);
                btsStageProbe("X-hyb", h);
            }
            if (pre.ghs_btoa) {
                Ciphertext g(ctxt.cc_);
                g.copy(sc);
                ghsSwitchSmall(g, *pre.ghs_btoa);
                btsStageProbe("X-ghs", g);
            }
        }
        // fused: c1 is scaled straight into Q+P (one reconstruction, one NTT pass) and switched with the one-digit
        // GHS key — no second lift, no 6-digit hybrid ModUp. FIDESLIB_BTS_SHIFT_FUSED=0 restores the two-step path.
        const bool fused = sparse_encaps && cc.compositeDegree() > 1 && cc.GetBootPrecomputation(slots).ghs_btoa &&
                           [] { const char* e = std::getenv("FIDESLIB_BTS_SHIFT_FUSED"); return e && std::atoi(e) > 0; }();  // measured +0.14 ms vs the hybrid two-step: default off
        if (fused) {
            const long double D = 1.0L / (long double)exactScaleT;
            smallIntDivideKeepLevel(cc, ctxt.c0, 3, D, false);
            smallIntDivideKeepLevel(cc, ctxt.c1, 3, D, true);  // Q + specials, NTT form, isModUp
            ctxt.NoiseFactor *= tf;
            ghsSwitchExt(ctxt, *cc.GetBootPrecomputation(slots).ghs_btoa);
        } else {
            smallIntScalarMultiply(cc, ctxt, exactScaleT);
            ctxt.NoiseFactor *= tf;
        }
        fusedSwitched = fused;
    }
    if (sparse_encaps && !aksStage0 && !fusedSwitched) {
        if (cc.compositeDegree() > 1) {
            if (exactScale > 0 && cc.GetBootPrecomputation(slots).ghs_btoa &&
                [] { const char* e = std::getenv("FIDESLIB_BTS_SHIFT_GHS"); return e && std::atoi(e) > 0; }()) {
                // default: the library hybrid switch (−0.67 ms vs GHS at t=12; FIDESLIB_BTS_SHIFT_GHS=1 opts in)
                // The hybrid switch's noise would land undivided on the already-scaled (2^22 smaller) signal; the
                // GHS switch on the small ciphertext only adds the ModDown rounding (CKKS/AksKeys.cuh).
                ghsSwitchSmall(ctxt, *cc.GetBootPrecomputation(slots).ghs_btoa);
            } else
            // COMPOSITESCALING: M-2 back to the dense key, MAIN-context standard hybrid key.
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_btoa);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;

            auto sparse_context_use = sparse_context.lock();

            auto& btoa = CKKS::GetSecretSwitchingKey(sparse_context_use, ctxt.cc_, ctxt.keyID);

            ctxt.keySwitch(btoa);
        }
    }

    btsStageProbe("MR-btoa", ctxt);
    ctxt.slots = cc.N / 2;
}

// Kept for API compatibility: the cached-bootstrap-graph machinery was removed, so this is a no-op.
int FIDESlib::CKKS::BootstrapPrecapture(FIDESlib::CKKS::Context& /*cc*/) {
    return 0;
}

