//
// Created by carlosad on 12/11/24.
//

#ifndef GPUCKKS_APPROXMODEVAL_CUH
#define GPUCKKS_APPROXMODEVAL_CUH

#include <cinttypes>
#include <vector>
#include "CKKS/forwardDefs.cuh"
namespace FIDESlib::CKKS {
/** OpenFHE: for degree 5 or less uses naïve implementation, on FIDESlib, its always Patterson Stockmayer
 *  I suggest to only use range [-1, 1]
 * */
void evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound = -1.0,
                         double upper_bound = 1.0);

void evalHornerSeries(Ciphertext& ctxt, const std::vector<double>& coefficients);

void approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey,
                        uint64_t post);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post);

/** Runtime scope for the arcsine EvalMod correction: 1 = force on, 0 = force
 *  off, -1 = FIDESLIB_ARCSINE env default. Levels must be reserved at context
 *  build (FIDESLIB_ARCSINE or FIDESLIB_ARCSINE_RESERVE). */
void setArcsineOverride(int v);

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_APPROXMODEVAL_CUH