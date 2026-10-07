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
        LtPtMaskScope ptMaskScope(step.ptMask);
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
    const int nStages = (int)(decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS).size();
    for (BootstrapPrecomputation::LTstep& step :
         (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS)) {
        // FIDESLIB_LT_FUSED_RESCALE=2: CtS stages only, 3: StC stages only (bisection)
        const int fuseMode = [] { const char* e = std::getenv("FIDESLIB_LT_FUSED_RESCALE"); return e && *e ? std::atoi(e) : 1; }();
        // FIDESLIB_LT_FUSED_CTS_MASK: bit k = fuse CtS stage k (bisection; default all)
        const int ctsMask = [] { const char* e = std::getenv("FIDESLIB_LT_FUSED_CTS_MASK"); return e ? std::atoi(e) : -1; }();
        LtFuseScope fuseScope(!(decode && steps == nStages - 1) && !(fuseMode == 2 && decode) && !(fuseMode == 3 && !decode) &&
                              (decode || ((ctsMask >> steps) & 1)));
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
                LtPtMaskScope ptMaskScope(step.ptMask);
                if (thB1 > 1 && step.bStep % thB1 == 0 && step.bStep / thB1 > 1)
                    LinearTransformTH(ctxt, step.slots, step.bStep, thB1, Aptr, stride, offset);
                else
                    LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, offset);
            }
        }
    }
}


// Diagnostic (DiagSparsityDump): coefficient-form limbs 0..2 of the first `nd` diagonals of dense CtS stages 0, 1 and the
// last StC stage, plus the first three primes, written into `dir` (runs inside the library so the struct layout is the
// library's own).
void FIDESlib::CKKS::dumpLtDiagCoeffs(ContextData& C, const char* dir, int nd) {
    auto& pre = C.GetBootPrecomputation(C.N / 2);
    {
        std::ofstream f(std::string(dir) + "/primes.txt");
        for (int l = 0; l < 3; ++l) f << C.prime[l].p << "\n";
    }
    std::cout << "[diag_sparse] N " << C.N << " primes " << C.prime[0].p << " " << C.prime[1].p << " " << C.prime[2].p
              << " CtS stages " << pre.CtS.size() << " StC stages " << pre.StC.size() << "\n";
    struct Sel { const char* tag; std::vector<BootstrapPrecomputation::LTstep>* v; int stage; };
    for (Sel s : {Sel{"cts0", &pre.CtS, 0}, Sel{"cts1", &pre.CtS, 1}, Sel{"stc2", &pre.StC, (int)pre.StC.size() - 1}}) {
        auto& st = (*s.v)[s.stage];
        for (int k = 0; k < std::min<int>(nd, (int)st.A.size()); ++k) {
            std::vector<std::vector<uint64_t>> nttLimbs;
            st.A[k].c0.store(nttLimbs);
            cudaDeviceSynchronize();
            const int nl = (int)nttLimbs.size();
            std::vector<uint64_t> moduli(nl);
            for (int l = 0; l < nl; ++l) moduli[l] = C.prime[l].p;
            RNSPoly r(C, nl - 1);
            r.load(nttLimbs, moduli);
            r.INTT(C.batch, true);
            cudaDeviceSynchronize();
            std::vector<std::vector<uint64_t>> limbs;
            r.store(limbs);
            const std::string base = std::string(dir) + "/" + s.tag + "_" + std::to_string(k);
            for (int l = 0; l < 3; ++l) {
                std::ofstream f(base + "_l" + std::to_string(l) + ".bin", std::ios::binary);
                f.write((const char*)limbs[l].data(), limbs[l].size() * sizeof(uint64_t));
            }
            std::ofstream m(base + ".meta");
            m << st.A[k].NoiseFactor << " " << st.A[k].c0.getLevel() << "\n";
        }
        std::cout << "[diag_sparse] dumped " << s.tag << " (" << st.A.size() << " diagonals)\n";
    }
}
