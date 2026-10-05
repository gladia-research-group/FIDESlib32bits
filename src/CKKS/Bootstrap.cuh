//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
/// Stage-divergence harness hook (default nullptr = off): install a vector to receive a
/// clone of the ciphertext at each bootstrap stage checkpoint (pre-CtS, post-CtS, pre-StC,
/// post-StC, end). Single-threaded use only; caller must reset to nullptr afterwards.
extern std::vector<std::pair<std::string, std::shared_ptr<Ciphertext>>>* g_btsStageStash;
void BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled);
// void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
/// No-op kept for API compatibility (always returns 0).
int BootstrapPrecapture(Context& cc);
double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false,
              const bool sparse_encaps = false, const double exactScale = 0.0, const bool aksStage0 = false);
}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAP_CUH
