//
// Created by carlosad on 12/06/25.
//

#ifndef ACCUMULATEBROADCAST_CUH
#define ACCUMULATEBROADCAST_CUH
#include <vector>
#include "CKKS/Ciphertext.cuh"
namespace FIDESlib::CKKS {
/**
 * Compute rotation indices for a rotate and accumulate where:
 *
 *  out_i = \sum_{j = 0}^{size - 1} in_{i%size + j*size}
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, stride, size - should be powers of two
 */
std::vector<int> GetAccumulateRotationIndices(const int bStep, const int stride, const int size);

/**
 * Compute rotation indices for a rotate and broadcast where:
 *
 *  in = {a_0, ... a_initsize, 0, ..., 0}
 *
 *  out_i = in_{i%initsize} iff i < outsize
 *
 *  out_i = 0 iff i >= outsize
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, initsize, outsize - should be powers of two
 */
std::vector<int> GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize);

/**
 * Compute a rotate and accumulate where:
 *
 *  out_i = \sum_{j = 0}^{size - 1} in_{i%size + j*size}
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, stride, size - should be powers of two
 */
void Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size);

/**
 *  Compute a rotate and broadcast where:
 *  in = {a_0, ... a_initsize, 0, ..., 0}
 *
 *  out_i = in_{i%initsize} iff i < outsize
 *
 *  out_i = 0 iff i >= outsize
 *
 *  with a bsgs algorithm using baby-step size bStep
 *
 *  bStep, initsize, outsize - should be powers of two
 */
void Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize);
}  // namespace FIDESlib::CKKS
#endif  //ACCUMULATEBROADCAST_CUH
