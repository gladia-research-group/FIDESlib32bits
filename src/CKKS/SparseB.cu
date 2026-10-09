// Lever B: repetition-aware depth-1 sparse CtS/StC (eprint 2026/1023, Alg. 4/5). Plaintext vectors come from the
// precomputation loader; here they are re-levelled to the ciphertext's level and lifted to the special primes
// (hoisted BSGS) on first use, then run through the shipped LinearTransform / Accumulate kernels.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/SparseB.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

const std::vector<double>& FIDESlib::CKKS::sparseBChebyshevK24() {
    // (2pi)^(-1/32) cos(2pi(24 y - 0.25)/32), Chebyshev interpolation truncated at DEGREE 20 (same Paterson-Stockmeyer
    // depth as degree 14): max |err| 2.9e-13 on [-1, 1] (degree 14: 7.6e-8, which capped the route at ~10.6 bits; the
    // shipped K = 16 / degree 14: 1.6e-10); evaluator convention (c0 doubled).
    static const std::vector<double> c = {
        -5.01431878575762102e-01, -2.60978240793176830e-02, -2.75969415548868335e-01,
        -3.76057943042461193e-02, 6.98674483745098551e-01, 2.06638885145157200e-02,
        -1.93915485831718332e-01, -3.59503488102385325e-03, 2.34902992390395096e-02,
        3.23162556614403976e-04, -1.63628724411827371e-03, -1.80046855470203313e-05,
        7.47055232583680419e-05, 6.86710430736672347e-07, -2.41801471221991641e-06,
        -1.91108603715104241e-08, 5.85030407613337840e-08, 4.05831352269635484e-10,
        -1.09946346874119116e-09, -6.79891244700694627e-12, 1.65253168377321969e-11};
    return c;
}

// The diagonals at the ciphertext's level, lifted to Q+P for the hoisted dot (built once per level).
static std::vector<Plaintext>& ptsAt(Ciphertext& ctxt, std::vector<Plaintext>& top,
                                     std::map<int, std::vector<Plaintext>>& cache, const char* what, double nf = 1.0) {
    ContextData& cc = ctxt.cc;
    const int lv = ctxt.getLevel();
    auto it = cache.find(lv);
    if (it != cache.end())
        return it->second;
    const int d = cc.compositeDegree();
    std::vector<Plaintext> v;
    v.reserve(top.size());
    for (auto& pt : top) {
        if ((pt.c0.getLevel() - lv) % d != 0)
            throw std::runtime_error(std::string("[sparse_b] ") + what + ": cannot re-level from limb index " +
                                     std::to_string(pt.c0.getLevel()) + " to " + std::to_string(lv));
        Plaintext np = relevelPlaintext(ctxt.cc_, cc, pt, (lv - pt.c0.getLevel()) / d, 1.0);
        smallIntLiftToSpecial(cc, np.c0, 4);
        np.NoiseFactor *= nf;
        v.push_back(std::move(np));
    }
    cudaDeviceSynchronize();
    std::cerr << "[sparse_b] " << what << ": " << v.size() << " diagonals re-levelled to limb index " << lv << "\n";
    return cache.emplace(lv, std::move(v)).first->second;
}

static void bsgs(Ciphertext& ctxt, std::vector<Plaintext>& pts, int s, int bStep) {
    std::vector<Plaintext*> A(s, nullptr);
    for (int k = 0; k < s; ++k)
        A[k] = &pts[k];
    LinearTransform(ctxt, s, bStep, A, 1, 0);  // sum_k d_k (.) rot_k(ct), d_k pre-rotated by -(k / bStep) * bStep
}

void FIDESlib::CKKS::SparseBCoeffsToSlots(Ciphertext& ctxt, int slots, BootstrapPrecomputation& pre) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    auto& sb = *pre.sparseB;
    ContextData& cc = ctxt.cc;
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    ctxt.slots = cc.N / 2;  // full-ring layout: the partial-sum blocks are not 2*slots-periodic
    bsgs(ctxt, ptsAt(ctxt, sb.P, sb.P_at, "CtS", sb.ctsNF), sb.s, sb.bStep);
    Accumulate(ctxt, pre.accumulate_bStep, sb.s, sb.r / 4);  // sum_{t < r/4} rot(t s)
    Ciphertext aux(ctxt.cc_);
    aux.conjugate(ctxt);
    ctxt.add(aux);
}

void FIDESlib::CKKS::SparseBSlotsToCoeffs(Ciphertext& ctxt, int slots, BootstrapPrecomputation& pre) {
    CudaNvtxRange r(std::string{sc::current().function_name()});
    auto& sb = *pre.sparseB;
    ContextData& cc = ctxt.cc;
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    ctxt.slots = cc.N / 2;
    bsgs(ctxt, ptsAt(ctxt, sb.Q, sb.Q_at, "StC"), sb.s, sb.bStep);
    Accumulate(ctxt, pre.accumulate_bStep, sb.n, sb.r / 2);  // sum_{t < r/2} rot(t n); leaves ctxt.slots = n
    ctxt.slots = cc.N / 2;
    Ciphertext aux(ctxt.cc_);
    aux.rotate(ctxt, sb.n / 2);
    ctxt.add(aux);
}

void FIDESlib::CKKS::sparseBUnit(Ciphertext& ct, int slots, bool stc, int& n, int& r, int& s) {
    auto& pre = ct.cc.GetBootPrecomputation(slots);
    if (!pre.sparseB) { n = r = s = 0; return; }
    n = pre.sparseB->n; r = pre.sparseB->r; s = pre.sparseB->s;
    if (stc) SparseBSlotsToCoeffs(ct, slots, pre); else SparseBCoeffsToSlots(ct, slots, pre);
    cudaDeviceSynchronize();
}
