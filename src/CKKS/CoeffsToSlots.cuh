//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_COEFFSTOSLOTS_CUH
#define GPUCKKS_COEFFSTOSLOTS_CUH
#include "forwardDefs.cuh"

#include <vector>
#include "BootstrapPrecomputation.cuh"

namespace FIDESlib::CKKS {
void EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode);

void EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode);

/// The single-LT bootstrap transform with an explicit plaintext set (bStep from the route's LT precomputation).
void EvalLinearTransformPts(Ciphertext& ctxt, int slots, int bStep, std::vector<Plaintext>& A);
/// The multi-stage CtS/StC runner on an explicit stage vector (plain hoisted BSGS, no AKS); `probeTag` names the
/// stage-divergence stash entries ("StC1-stage-").
void EvalLTStages(Ciphertext& ctxt, std::vector<BootstrapPrecomputation::LTstep>& stages, const char* probeTag);
}  // namespace FIDESlib::CKKS
#endif  //GPUCKKS_COEFFSTOSLOTS_CUH
