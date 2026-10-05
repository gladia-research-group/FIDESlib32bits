#pragma once
// Single-pass 4-step NTT / INTT over a thread-block CLUSTER (sm_90+, traffic campaign phase B, 2026-10-04).
//
// The two-kernel transform (NTT.cu) writes its 256-KB intermediate to a scratch limb and re-reads it
// transposed in a second launch. Here one cluster of 8 blocks x 512 threads owns one limb: each block
// holds 4 tiles (4 x 8 KB) of the intermediate in shared memory and the transpose between the two passes
// is done through distributed shared memory (cluster.map_shared_rank), so a limb is read from and written
// to L2/DRAM exactly once and there is one launch per transform instead of two.
//
// Scope: u32 limbs, ALGO_SHOUP, plain modes (NTT_NONE / INTT_NONE). Other modes fall back to NTT.cu.
// Bit-exact with the two-pass kernels by construction (same butterflies, twiddles and swizzle).
#include <cuda_runtime.h>
#include "ConstantsGPU.cuh"

namespace FIDESlib {
/// FIDESLIB_NTT_CLUSTER (env, default 0) / setNttCluster: route the plain u32 transforms to the cluster kernel.
int nttClusterFlag();
void setNttCluster(int v);
/// One launch for `nlimbs` limbs: dat[l] -> res[l] (pointer tables, nullptr rows are skipped),
/// primeid_flattened[primeid_init + l] is limb l's prime. inverse = INTT (negacyclic, scaled by 1/N).
void launchNTTcluster(const Global::Globals* Globals, bool inverse, void** dat, int primeid_init, void** res, int nlimbs,
                      cudaStream_t s);
}  // namespace FIDESlib
