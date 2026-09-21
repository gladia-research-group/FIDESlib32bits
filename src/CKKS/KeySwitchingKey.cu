//
// Created by carlosad on 26/09/24.
//

#include <algorithm>
#include <cstdlib>
#include <source_location>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

namespace {
/* KSK bit-packing (FIDESLIB_KSK_PACK, default 1): uniform packed width for this chain's keys, or
 * 0 = dense. Lossless; saves 12.5% key device memory at the cost of an unpack in the dot kernels.
 * Eligible: single-GPU, all-u32 (type==0) chains whose max prime width has a packed kernel arm. */
int kskPackBitsPolicy(Context& cc) {
    static const bool enabled = [] {
        const char* e = std::getenv("FIDESLIB_KSK_PACK");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (!enabled || cc->GPUid.size() != 1)
        return 0;
    const auto& hc = cc->precom.constants[0];
    if (hc.type != 0)
        return 0;
    const int K = (int)cc->splitSpecialMeta.at(0).size();
    int W = 0;
    for (int i = 0; i < cc->L + K; ++i)
        W = std::max<int>(W, (int)hc.prime_bits[i]);
    // Widths are compile-time in the dot kernels: arm only the instantiated set; other chains
    // fall back dense (add an instantiation to extend).
    return (W == 27 || W == 28) ? W : 0;
}
}  // namespace

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk, int q_band) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    CKKS::SetCurrentContext(cc);
    keyID = rkk.keyid;

    if (q_band >= 0 && cc->GPUid.size() > 1)
        q_band = -1;  // banding is single-GPU only
    // A seeded key's `a` component is regenerated on-GPU from its 256-bit seed (bit-identical to
    // rkk.r_key[0]), skipping that half of the key upload. `b`, and seedless keys, load normally.
    const bool seeded_a = !rkk.a_seed.empty() && cc->GPUid.size() == 1;
    // At regen level >= 2 both `a`-readers (hoistedRotateDotKSK and fusedDotKSK) regenerate from the
    // seed, so `a` is never read and is released instead of expanded. The N condition is the regen
    // kernels' whole-threads requirement; if it fails the launch gates stream, so `a` must exist.
    const bool release_a = seeded_a && kskRegenLevel() >= 2 &&
                           cc->N % (128 * (cc->precom.constants[0].type == 0 ? 16 : 8)) == 0;

    // When released, `a`'s rows are never allocated: the pointer tables the host staging paths read
    // live in bufferAUXptrs and are built by the LimbPartition constructor from the metas.
    if (!release_a)
        a.generateDecompAndDigit(true, q_band);
    b.generateDecompAndDigit(true, q_band);
    if (cc->GPUid.size() > 1) {
        a.grow(cc->L, false, true);
        b.grow(cc->L, false, true);
    }
    if (release_a)
        a.GPU.at(0).adoptKskASeed(rkk.a_seed, q_band);
    else if (seeded_a)
        a.GPU.at(0).expandKskADigits(rkk.a_seed);
    else
        a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
    b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

    if (const int W = kskPackBitsPolicy(cc)) {
        if (release_a)
            // `a` has no rows to pack; the width still has to match `b` (the compile-time KSK_BITS
            // the dot kernels select on).
            a.GPU.at(0).key_pack_bits = W;
        else
            a.GPU.at(0).packKeyLimbs(W);
        b.GPU.at(0).packKeyLimbs(W);
    }

    cudaDeviceSynchronize();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
    : my_range(loc, LIFETIME),
      keyID(""),
      cc((assert(cc != nullptr), CudaNvtxStart(std::string{sc::current().function_name()}.substr()), cc)),
      a(*cc, -1, false, true),
      b(*cc, -1, false, true) {
    CudaNvtxStop();
    /*
    if (cc.GPUid.size() > 1) {
        for (int j = 0; j < cc.dnum; ++j) {
            mgpu_a.emplace_back(cc, -1);
            mgpu_b.emplace_back(cc, -1);
        }
    }
     */
}
}  // namespace FIDESlib::CKKS
