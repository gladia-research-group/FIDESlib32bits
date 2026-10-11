//
// FIDESLIB_NOCLONE=3: the write-once limb ops behind Ciphertext::subInto / multScalarInto / addScalarInto. Each writes
// its result from the source limbs in one pass (the in-place ops these replace ran on a full copy of the source first);
// same modular arithmetic as the in-place kernels, so the residues are identical.
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

__global__ void scalarMultFrom_(void** out, void** src, const __grid_constant__ LimbScalars s,
                                const __grid_constant__ int primeid_init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    if (ISU64(primeid))
        ((uint64_t*)out[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((const uint64_t*)src[blockIdx.y])[idx], s.v[primeid], primeid);
    else
        ((uint32_t*)out[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((const uint32_t*)src[blockIdx.y])[idx], (uint32_t)s.v[primeid], primeid);
}

__global__ void scalarAddFrom_(void** out, void** src, const __grid_constant__ LimbScalars s,
                               const __grid_constant__ int primeid_init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    if (ISU64(primeid))
        ((uint64_t*)out[blockIdx.y])[idx] = modadd(((const uint64_t*)src[blockIdx.y])[idx], s.v[primeid], primeid);
    else
        ((uint32_t*)out[blockIdx.y])[idx] =
            modadd(((const uint32_t*)src[blockIdx.y])[idx], (uint32_t)s.v[primeid], primeid);
}

}  // namespace

// The launches follow LimbPartition::add(a, b): this partition's stream waits for the sources, runs, and the sources'
// streams wait for it (their later writes / frees stay ordered after these reads).
void LimbPartition::subFrom(const LimbPartition& a, const LimbPartition& b) {
    cudaSetDevice(device);
    s.wait(a.getS());
    s.wait(b.getS());
    const int limbsize = getLimbSize(*level);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        const uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch)
        s.wait(STREAM(limb[i]));
    a.getS().wait(s);
    b.getS().wait(s);
}

void LimbPartition::multScalarFrom(const LimbPartition& src, const LimbScalars& sc) {
    cudaSetDevice(device);
    s.wait(src.getS());
    const int limbsize = getLimbSize(*level);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        const uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        scalarMultFrom_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, src.limbptr.data + i, sc, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch)
        s.wait(STREAM(limb[i]));
    src.getS().wait(s);
}

void LimbPartition::addScalarFrom(const LimbPartition& src, const LimbScalars& sc) {
    cudaSetDevice(device);
    s.wait(src.getS());
    const int limbsize = getLimbSize(*level);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        const uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        scalarAddFrom_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr()>>>(
            limbptr.data + i, src.limbptr.data + i, sc, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch)
        s.wait(STREAM(limb[i]));
    src.getS().wait(s);
}

void RNSPoly::subFrom(const RNSPoly& a, const RNSPoly& b) {
    assert(level <= a.level && level <= b.level && !a.isModUp() && !b.isModUp());
    for (size_t i = 0; i < cc.GPUid.size(); ++i)
        GPU.at(i).subFrom(a.GPU.at(i), b.GPU.at(i));
    SetModUp(false);
}

void RNSPoly::multScalarFrom(const RNSPoly& src, const std::vector<uint64_t>& elem) {
    assert(level <= src.level && !src.isModUp() && elem.size() <= (size_t)MAXP);
    LimbScalars sc{};
    for (size_t i = 0; i < elem.size(); ++i)
        sc.v[i] = elem[i];
    for (size_t i = 0; i < cc.GPUid.size(); ++i)
        GPU.at(i).multScalarFrom(src.GPU.at(i), sc);
    SetModUp(false);
}

void RNSPoly::addScalarFrom(const RNSPoly& src, const std::vector<uint64_t>& elem) {
    assert(level <= src.level && !src.isModUp() && elem.size() <= (size_t)MAXP);
    LimbScalars sc{};
    for (size_t i = 0; i < elem.size(); ++i)
        sc.v[i] = elem[i];
    for (size_t i = 0; i < cc.GPUid.size(); ++i)
        GPU.at(i).addScalarFrom(src.GPU.at(i), sc);
    SetModUp(false);
}

}  // namespace FIDESlib::CKKS
