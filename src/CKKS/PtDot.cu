//
// LIN_PT_DOT: the baby-step inner sum of a BSGS linear, acc = sum_i x[i] * pt[i], written once by one kernel over both
// polynomials of the accumulator (Ciphertext::multPtSumInto). Same Barrett products and modular sums as multPtInto
// followed by add for each further term, so the residues are identical; each operand element is read once and the
// products never reach memory.
//
#include "AddSub.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {

namespace {

struct PtDotTable {
    void** x0[PTDOT_MAX];
    void** x1[PTDOT_MAX];
    void** p[PTDOT_MAX];
};

template <typename T>
__device__ __forceinline__ void ptDotCoeff(void** out0, void** out1, const PtDotTable& t, const int n, const int l,
                                           const int idx, const int primeid) {
    T p = ((const T*)t.p[0][l])[idx];
    T a0 = modmult<ALGO_BARRETT>(((const T*)t.x0[0][l])[idx], p, primeid);
    T a1 = modmult<ALGO_BARRETT>(((const T*)t.x1[0][l])[idx], p, primeid);
    for (int i = 1; i < n; ++i) {
        p = ((const T*)t.p[i][l])[idx];
        a0 = modadd(a0, modmult<ALGO_BARRETT>(((const T*)t.x0[i][l])[idx], p, primeid), primeid);
        a1 = modadd(a1, modmult<ALGO_BARRETT>(((const T*)t.x1[i][l])[idx], p, primeid), primeid);
    }
    ((T*)out0[l])[idx] = a0;
    ((T*)out1[l])[idx] = a1;
}

__global__ void ptDot_(void** out0, void** out1, const __grid_constant__ PtDotTable t, const __grid_constant__ int n,
                       const __grid_constant__ int primeid_init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    if (ISU64(primeid))
        ptDotCoeff<uint64_t>(out0, out1, t, n, blockIdx.y, idx, primeid);
    else
        ptDotCoeff<uint32_t>(out0, out1, t, n, blockIdx.y, idx, primeid);
}

}  // namespace

// Ordering as in LimbPartition::add(a, b): this partition's stream waits for the destination's other polynomial and
// every source, runs, and they all wait for it (their later writes / frees stay ordered after these reads).
void LimbPartition::ptDotFrom(LimbPartition& out1, const std::vector<const LimbPartition*>& x0,
                              const std::vector<const LimbPartition*>& x1, const std::vector<const LimbPartition*>& pts) {
    const int n = (int)x0.size();
    assert(n > 0 && n <= PTDOT_MAX && x1.size() == x0.size() && pts.size() == x0.size());
    cudaSetDevice(device);
    s.wait(out1.s);
    for (int k = 0; k < n; ++k) {
        s.wait(x0[k]->getS());
        s.wait(x1[k]->getS());
        s.wait(pts[k]->getS());
    }
    const int limbsize = getLimbSize(*level);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        const uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        PtDotTable t{};
        for (int k = 0; k < n; ++k) {
            t.x0[k] = x0[k]->limbptr.data + i;
            t.x1[k] = x1[k]->limbptr.data + i;
            t.p[k] = pts[k]->limbptr.data + i;
        }
        ptDot_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, out1.limbptr.data + i, t, n, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch)
        s.wait(STREAM(limb[i]));
    out1.s.wait(s);
    for (int k = 0; k < n; ++k) {
        x0[k]->getS().wait(s);
        x1[k]->getS().wait(s);
        pts[k]->getS().wait(s);
    }
}

void RNSPoly::ptDotFrom(RNSPoly& out1, const std::vector<const RNSPoly*>& x0, const std::vector<const RNSPoly*>& x1,
                        const std::vector<const RNSPoly*>& pts) {
    const size_t n = x0.size();
    for (size_t k = 0; k < n; ++k)
        assert(level <= x0[k]->level && level <= x1[k]->level && level <= pts[k]->level && !x0[k]->isModUp() &&
               !x1[k]->isModUp());
    for (size_t g = 0; g < cc.GPUid.size(); ++g) {
        std::vector<const LimbPartition*> a(n), b(n), p(n);
        for (size_t k = 0; k < n; ++k) {
            a[k] = &x0[k]->GPU.at(g);
            b[k] = &x1[k]->GPU.at(g);
            p[k] = &pts[k]->GPU.at(g);
        }
        GPU.at(g).ptDotFrom(out1.GPU.at(g), a, b, p);
    }
    SetModUp(false);
    out1.SetModUp(false);
}

}  // namespace FIDESlib::CKKS
