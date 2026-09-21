//
// Created by carlosad on 24/12/25.
//
#include "CKKS/Parameters.cuh"

namespace FIDESlib::CKKS {
static TYPE pickLimbType(uint64_t p) {
    // Use U32 for primes ≤ 2^31 (leaves 1 bit for signed NTT arithmetic).
    // 27–28-bit CKKS moduli fit comfortably; larger primes fall back to U64.
    bool use_u32 = p <= (1ULL << 31);
    // std::fprintf(stderr, "[pickLimbType] p=%lu -> %s\n", p, use_u32 ? "U32" : "U64");
    return use_u32 ? U32 : U64;
}

Parameters Parameters::adaptTo(RawParams& raw) const {
    std::vector<PrimeRecord> new_primes;
    for (auto i : raw.moduli) {
        new_primes.push_back(PrimeRecord{.p = i, .type = pickLimbType(i)});
    }
    std::vector<PrimeRecord> new_SPECIALprimes;
    for (auto i : raw.SPECIALmoduli) {
        new_SPECIALprimes.push_back(PrimeRecord{.p = i, .type = pickLimbType(i)});
    }
    // std::fprintf(stderr, "[adaptTo] primes=%zu special=%zu all U32=%s\n",
    //              new_primes.size(), new_SPECIALprimes.size(),
    //              (!new_primes.empty() && new_primes[0].type && *new_primes[0].type == U32) ? "YES" : "NO");

    Parameters res{.logN = raw.logN,
                   .L = raw.L,
                   .dnum = raw.dnum,
                   .K = raw.K,
                   .compositeDegree = raw.compositeDegree,
                   .primes = std::move(new_primes),
                   .Sprimes = std::move(new_SPECIALprimes),
                   .ModReduceFactor = raw.ModReduceFactor,
                   .ScalingFactorReal = raw.ScalingFactorReal,
                   .ScalingFactorRealBig = raw.ScalingFactorRealBig,
                   .scalingTechnique = raw.scalingTechnique,
                   .raw = raw,
                   .batch = batch};
    //std::cout << "Adapt out" << std::endl;
    return res;
}
}  // namespace FIDESlib::CKKS