// Compact storage of NTT-structured plaintexts: a slot vector that is a polynomial in X^k has an NTT image of constant
// blocks (or a period), so a CtS / StC stage diagonal keeps one value per block. The LT dot indexes it directly; any
// other reader expands it first (expandCompact).
#ifndef FIDESLIB_CKKS_PTCOMPACT_CUH
#define FIDESLIB_CKKS_PTCOMPACT_CUH

#include <cstdint>
#include <vector>

#include "BootstrapPrecomputation.cuh"
#include "Plaintext.cuh"

namespace FIDESlib::CKKS {

/// The read index of element `idx` of a limb stored under `enc`: a period mask (P - 1) indexes `idx & mask`, the
/// compact-block encoding 0x7F0000ss indexes `idx >> ss`, 0xFFFFFFFF the full limb.
__host__ __device__ inline int ptCompactIdx(int idx, uint32_t enc) {
    return ((enc >> 24) == 0x7Fu) ? (idx >> (enc & 0xFFu)) : (int)((uint32_t)idx & enc);
}

/// The full-storage read mask of a (possibly compact-encoded) stage mask: ~(L - 1) for blocks of L, else as is.
inline uint32_t ptCompactFullMask(uint32_t enc) {
    return ((enc >> 24) == 0x7Fu) ? ~((1u << (enc & 0xFFu)) - 1u) : enc;
}

/// Store `pt`'s limbs compactly under the stage mask `mask` (as ltStageMask returns it). Every limb, the special ones
/// included, is checked against the mask first; on any mismatch nothing changes and false is returned. On success
/// pt.compactMask holds the encoding the LT dot reads with.
bool compactPlaintext(Plaintext& pt, uint32_t mask);

/// A full-storage copy of a compact plaintext (its own limbs, scratch included); `pt` itself when it is full.
Plaintext expandCompact(const Plaintext& pt);

/// Compact every masked CtS / StC stage of `pre` (and its raise variants). A stage is compacted only when every one of
/// its plaintexts verifies; its ptMask then becomes the compact encoding. Returns (plaintexts compacted, bytes freed).
std::pair<size_t, size_t> compactStages(BootstrapPrecomputation& pre);

}  // namespace FIDESlib::CKKS

#endif
