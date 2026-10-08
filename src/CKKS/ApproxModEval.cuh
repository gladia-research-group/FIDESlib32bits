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
                        uint64_t post, bool evalRound = false);

/// The dense EvalMod of a real payload: the Re chain alone. A real slot vector has m_{N-k} = -m_k, so the Im
/// half of its coefficients repeats the Re half and needs no reduction of its own.
void approxModReductionReal(Ciphertext& ctxtEnc, const KeySwitchingKey& keySwitchingKey, uint64_t post);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post);
/// Same with a per-route EvalMod (lever B: the K = 24 series of the repetition-aware sparse layout).
void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post, std::vector<double>& coefficients, int daIts);

/** Runtime scope for the arcsine EvalMod correction: 1 = force on, 0 = force off, -1 = default (off).
 *  The 3 levels it consumes must be reserved at context build (FIDESLIB_SPARSE_ARCSINE). */
void setArcsineOverride(int v);

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_APPROXMODEVAL_CUH