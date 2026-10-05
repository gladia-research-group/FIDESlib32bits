#pragma once
// Transient-scratch discard (traffic campaign, 2026-10-03). `discard.global.L2` drops lines from L2
// without the DRAM write-back; legal only for scratch whose every line is fully rewritten before it is
// read again. All helpers are no-ops unless the FIDESLIB_DISCARD_SCRATCH flag is set (see NTT.cu).
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace FIDESlib::CKKS {
class LimbPartition;
/// Discard n limbs of `bytes` each, addressed through a device pointer table (128-B aligned limbs).
void discardLimbTable(void** d_limbptr, int n, size_t bytes, cudaStream_t s);
/// The partition's regular limbs [0, limbsize at its level).
void discardRegularLimbs(LimbPartition& p, cudaStream_t s);
/// The partition's special (P) limbs.
void discardSpecialLimbs(LimbPartition& p, cudaStream_t s);
/// Every key-switch digit row of the partition (ModUp scratch).
void discardDigitLimbs(LimbPartition& p, cudaStream_t s);
/// FIDESLIB_DISCARD_D2: the per-op discards (digits after the key-switch dots, ModDown scratch). Measured
/// 2026-10-03: -0.3 GB/bts for ~150 extra launches, net wall LOSS vs D1 alone -> default 0.
int discardD2Flag();
/// FIDESLIB_MODUP_MERGE (lever B1): coarser ModUp launches on the hoisted path — one INTT over all limbs
/// instead of one per digit, and the special limbs NTT'd in the same launch as the regular limbs. Pure
/// launch geometry (bit-exact). setModupMerge overrides the env for in-process A/Bs.
int modupMergeFlag();
void setModupMerge(int v);
/// FIDESLIB_FUSED_RESCALE (lever A): Ciphertext::mult(b, rescale=true) performs the composite rescale inside
/// the key-switch ModDown (one base conversion from P u {q_a, q_b}) instead of a separate rescale pass.
/// Not bit-exact with mult + rescale (the approximate conversion rounds differently); gate on bits / KL.
int fusedRescaleFlag();
void setFusedRescale(int v);
/// FIDESLIB_PW_FUSE (lever E): out-of-place mult/square (no operand copy) and the fused copy*P of the identity
/// rotation in hoisted linear transforms. Pure reorderings: bit-exact.
int pwFuseFlag();
void setPwFuse(int v);
/// One launch for several limb tables: discards limbs [begin, begin+n) of each table (D3 chunk discard).
void discardLimbTables(const std::vector<void**>& tables, int begin, int n, size_t bytes, cudaStream_t s);
}  // namespace FIDESlib::CKKS
