#include "CKKS/RNSPoly.cuh"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
//
// Created by carlosad on 27/11/24.
//

#include <ranges>
#include <vector>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/AksKeys.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool BATCHED = false;

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    constexpr bool PRINT = false;
    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;

    if constexpr (BATCHED) {
        /*
        CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
                                              .conf = {.cc_ = ctxt.cc_,
                                                       .dims = {{.size = 1}},
                                                       .level = ctxt.getLevel(),
                                                       .scale_degree = ctxt.NoiseLevel,
                                                       .isExt = ctxt.c0.isModUp()}};

        auto& LTconf = cc.GetBootPrecomputation(slots).LT;
        PlaintextBatch<Plaintext*> bptxt = {
            .conf = {.cc_ = ctxt.cc_,
                     .dims = {{.size = 1},
                              {.size = (LTconf.slots + LTconf.bStep - 1) / LTconf.bStep},
                              {.size = LTconf.bStep}},
                     .level = ctxt.getLevel(),
                     .scale_degree = ctxt.NoiseLevel,
                     .isExt = (decode ? LTconf.invA : LTconf.A)[0].c0.isModUp()}};
        for (auto& i : (decode ? LTconf.invA : LTconf.A)) {
            bptxt.cts.push_back(&i);
        }
        int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
        while (bptxt.cts.size() < rowSize_padded)
            bptxt.cts.push_back(nullptr);

        LinearTransform(bctxt, rowSize_padded, LTconf.bStep, bptxt, 1, 0);
        */
    } else {
        EvalLinearTransformPts(ctxt, slots, cc.GetBootPrecomputation(slots).LT.bStep,
                               decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A);
    }
}

void FIDESlib::CKKS::EvalLinearTransformPts(Ciphertext& ctxt, int slots, int bStep_, std::vector<Plaintext>& A) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    // Computing the baby-step bStep and the giant-step gStep.
    uint32_t bStep = bStep_;
    uint32_t gStep = ceil(static_cast<double>(slots) / bStep);

    std::vector<Ciphertext> fastRotation;
    for (auto i = fastRotation.size(); i < bStep; ++i)
        fastRotation.emplace_back(cc_);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);
        indexes.push_back(i);
    }

    bool ext = true;
    if (bStep == 1)
        ext = false;
    for (auto& i : A) {
        if (!i.c0.isModUp()) {
            ext = false;
        }
    }

    ctxt.rotate_hoisted(indexes, fastRotationPtr, ext);

    std::vector<Plaintext*> Aptr(slots, nullptr);
    for (int j = 0; j < gStep; ++j) {
        for (int i = 0; i < bStep; ++i) {
            if (bStep * j + i < slots)
                Aptr[bStep * j + i] = &(A[bStep * j + i]);
        }
    }
    LinearTransform(ctxt, slots, bStep, Aptr, 1, 0);
}

void FIDESlib::CKKS::EvalLTStages(Ciphertext& ctxt, std::vector<BootstrapPrecomputation::LTstep>& stages,
                                  const char* probeTag) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    int steps = 0;
    for (BootstrapPrecomputation::LTstep& step : stages) {
        if (g_btsStageStash) {
            cudaDeviceSynchronize();
            auto c = std::make_shared<Ciphertext>(ctxt.cc_);
            c->copy(ctxt);
            g_btsStageStash->emplace_back(std::string(probeTag) + std::to_string(steps), std::move(c));
        }
        ++steps;
        assert(step.slots == step.A.size());
        std::vector<Plaintext*> Aptr(step.slots, nullptr);
        for (int j = 0; j < step.gStep; ++j)
            for (int i = 0; i < step.bStep; ++i)
                if (step.bStep * j + i < step.slots)
                    Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
        const int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
        LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, step.rotOut[0]);
    }
}

void FIDESlib::CKKS::EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    constexpr bool PRINT = false;
    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;

    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Input stc ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
    //  No need for Encrypted Bit Reverse
    //Ciphertext& result = ctxt;
    // hoisted automorphisms
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();

    int steps = 0;
    for (BootstrapPrecomputation::LTstep& step :
         (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS)) {
        // Stage-divergence harness: stash the ciphertext ENTERING each LT stage so intra-CtS/StC
        // noise injection is attributable per stage. Inert unless a caller installed the stash.
        if (g_btsStageStash) {
            cudaDeviceSynchronize();
            auto c = std::make_shared<Ciphertext>(ctxt.cc_);
            c->copy(ctxt);
            g_btsStageStash->emplace_back(
                std::string(decode ? "StC-stage-" : "CtS-stage-") + std::to_string(steps),
                std::move(c));
        }
        ++steps;
        if (!decode && steps == 1 && cc.GetBootPrecomputation(slots).aks0) {
            auto& pre = cc.GetBootPrecomputation(slots);
            std::unique_ptr<Ciphertext> ref;
            if (g_btsStageStash) {  // reference: lever-1a path (exact scale, hybrid switch) + the standard stage 0
                ref = std::make_unique<Ciphertext>(ctxt.cc_);
                ref->copy(ctxt);
                const double tf = std::ldexp(1.0, pre.cts0_t);
                smallIntScalarMultiply(cc, *ref, pre.cts0_const * tf);
                ref->NoiseFactor *= tf;
                ref->keySwitch(*pre.sparse_btoa);
                auto c0 = std::make_shared<Ciphertext>(ctxt.cc_);
                c0->copy(*ref);
                g_btsStageStash->emplace_back("AKS-ref-switched", std::move(c0));
            }
            LinearTransformAKS(ctxt, step, *pre.aks0);  // CKKS/AksKeys.cuh
            if (ref) {
                std::vector<Plaintext*> Aptr(step.slots, nullptr);
                for (int j = 0; j < step.gStep; ++j)
                    for (int i = 0; i < step.bStep; ++i)
                        if (step.bStep * j + i < step.slots)
                            Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
                const int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
                LinearTransform(*ref, step.slots, step.bStep, Aptr, stride, step.rotOut[0]);
                cudaDeviceSynchronize();
                auto c = std::make_shared<Ciphertext>(ctxt.cc_);
                c->copy(ctxt);
                g_btsStageStash->emplace_back("AKS-out", std::move(c));
                g_btsStageStash->emplace_back("AKS-ref", std::shared_ptr<Ciphertext>(std::move(ref)));
            }
            continue;
        }
        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)

        if constexpr (AFFINE_LT && BATCHED) {
            /*
            CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
                                                  .conf = {.cc_ = ctxt.cc_,
                                                           .dims = {{.size = 1}},
                                                           .level = ctxt.getLevel(),
                                                           .scale_degree = ctxt.NoiseLevel,
                                                           .isExt = ctxt.c0.isModUp()}};

            if (bctxt.conf.scale_degree == 2)
                bctxt.Rescale();

            PlaintextBatch<Plaintext*> bptxt = {
                .conf = {
                    .cc_ = ctxt.cc_,
                    .dims = {{.size = 1}, {.size = (step.slots + step.bStep - 1) / step.bStep}, {.size = step.bStep}},
                    .level = ctxt.getLevel(),
                    .scale_degree = ctxt.NoiseLevel,
                    .isExt = step.A[0].c0.isModUp()}};
            for (auto& i : step.A) {
                bptxt.cts.push_back(&i);
            }
            int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
            while (bptxt.cts.size() < rowSize_padded)
                bptxt.cts.push_back(nullptr);

            LinearTransform(bctxt, rowSize_padded, step.bStep, bptxt, step.rotIn[1] - step.rotIn[0], step.rotOut[0]);
            */
        } else {

            {

                assert(step.slots == step.A.size());
                std::vector<Plaintext*> Aptr(step.slots, nullptr);
                for (int j = 0; j < step.gStep; ++j) {
                    for (int i = 0; i < step.bStep; ++i) {
                        if (step.bStep * j + i < step.slots)
                            Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
                    }
                }

                int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
                int offset = step.rotOut[0];
                // FIDESLIB_LT_TH_B1=k: triple-hoisted baby step (two hoisted layers, k * bStep/k) — price gate
                static const int thB1 = [] { const char* e = std::getenv("FIDESLIB_LT_TH_B1"); return e ? std::atoi(e) : 0; }();
                if (thB1 > 1 && step.bStep % thB1 == 0 && step.bStep / thB1 > 1)
                    LinearTransformTH(ctxt, step.slots, step.bStep, thB1, Aptr, stride, offset);
                else
                    LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, offset);
            }
        }
    }
}
