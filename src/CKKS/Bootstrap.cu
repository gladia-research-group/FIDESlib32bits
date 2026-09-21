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
#include "CKKS/Bootstrap.cuh"
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
        pre    = cc.sfAtLimb(cc.L) / qDouble;
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

void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
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
        pre    = cc.sfAtLimb(cc.L) / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;

    {
        ModRaise(ctxt, slots, correction, prescaled, sparse_encaps);
        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        // TO-DO: The 1/32 scale will be pre-applied with OpenFHE v1.4, so remove it from here
        double constantEvalMult = pre * (1.0 / (k * cc.N));

        /*
        if (sparse_encaps) {
            constantEvalMult = pre * (1.0 / (k * cc.N) / 32);
        }
        */

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

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

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
                              const bool sparse_encaps) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    ContextData& cc = ctxt.cc;
    btsStageProbe("MR-entry", ctxt);
    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    if (!prescaled) {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 >= 1);
    } else {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 == 0);
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
        double targetSF = cc.sfAtLimb(cc.L - lvl * cc.compositeDegree());
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
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
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
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
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
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
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
    if (sparse_encaps) {
        if (cc.compositeDegree() > 1) {
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
