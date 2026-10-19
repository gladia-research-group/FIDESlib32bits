//
// Level-adaptive relinearization (FIDESLIB_LAR=1, default off).
//
// The production eval key is a dnum-digit hybrid key over Q_L * P with alpha = |P| primes per
// digit, so a relin at n live limbs pays ceil(n / alpha) digit ModUps to n + K limbs each. Below
// the top of the chain the primes q_n..q_L are idle; a second key that moves them into the special
// modulus (P' = P * q_nHi..q_L) and splits Q_nHi into beta' = 2..3 wide digits does the same relin
// with fewer, wider digits (Han-Ki hybrid key switching with a level-dependent gadget). Same total
// modulus Q_L * P as the eval key, so the security level is unchanged; the key is generated with
// fresh randomness from the held secret (LoadLarKeys), never derived from the eval key.
//
// Single GPU, all-u32 chains only. The default key-switch path is untouched; MGPUkeySwitchCore
// takes this path only for the eval key at levels a class covers (larTryKeySwitch).
//
#ifndef FIDESLIB_CKKS_LEVELADAPTIVEKS_CUH
#define FIDESLIB_CKKS_LEVELADAPTIVEKS_CUH

#include <cstdint>
#include <string>
#include <vector>
#include "CKKS/forwardDefs.cuh"

namespace FIDESlib::CKKS {

/** One key class: Q_nHi split into contiguous digits [start[j], start[j+1]), P' = q_nHi..q_{L} plus P. */
struct LarGeometry {
    int nHi = 0;                 // live limbs the class is built for (Q ids 0..nHi-1)
    int nLo = 0;                 // smallest live-limb count the class serves
    std::vector<int> start;      // beta' + 1 digit boundaries, start[0] = 0, start.back() = nHi
    std::vector<int> pPrimeIds;  // P' prime ids in the order the targets carry them
};

/** FIDESLIB_LAR (default 0): build and use the level-adaptive keys. */
bool larEnabled();
/** Runtime dispatch switch (tests flip it to compare both paths on one context). */
void larSetActive(bool on);

/** The classes this context gets (FIDESLIB_LAR_CLASSES="nHi:beta[,nHi:beta]", default "42:2";
 *  FIDESLIB_LAR_NLO = the lowest class's floor, default 28). Empty when disabled or unsupported. */
std::vector<LarGeometry> larPlanClasses(Context& cc);

/** Host keys per class, per digit: a[j][id], b[j][id] = the EVAL-form limb of prime id (0..nQ+K-1). With a seed
 *  (8 words) `a` is not stored: digit j's limb mod p is kskexpand::expand_coeff(seed, j, p, slot) (KskSeedExpand.cuh,
 *  the production eval key's ChaCha12 lane) and the dot regenerates it, so `a` may be empty. */
struct LarHostKey {
    LarGeometry geo;
    std::vector<std::vector<std::vector<uint64_t>>> a, b;
    std::vector<uint32_t> seed;
};
/** FIDESLIB_LAR_SEEDED (default 0): seed-expanded `a` (regenerated in the dot) instead of a stored one. */
bool larSeeded();
/** Upload the keys and every per-level table; `evalKeyTag` is the eval key's tag (dispatch filter). */
void InstallLarKeys(Context& cc, std::vector<LarHostKey>&& keys, const KeyHash& evalKeyTag);

/** Relinearize `in` (NTT form over Q_n) with the class covering n: aux <- <d, b>/P', in <- <d, a>/P'.
 *  Returns false (nothing launched) when the key is not the eval key or no class covers the level. */
bool larTryKeySwitch(RNSPoly& in, RNSPoly& aux, const KeySwitchingKey& ksk);

}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_LEVELADAPTIVEKS_CUH
