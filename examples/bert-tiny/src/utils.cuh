#ifndef FIDESLIB_BERT_TINY_UTILS_CUH
#define FIDESLIB_BERT_TINY_UTILS_CUH

#include <openfhe.h>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <CKKS/KeySwitchingKey.cuh>
#include "CKKS/Ciphertext.cuh"
#include <CKKS/Plaintext.cuh>
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "../test/ParametrizedTest.cuh"

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Parameters.cuh"

extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;
extern FIDESlib::CKKS::Parameters params;

extern lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;

void prepare_gpu_context_bert(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
                              FIDESlib::CKKS::EncoderConfiguration& conf);

void create_cpu_context();

void prepare_cpu_context(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, size_t num_slots,
                         size_t blockSize, FIDESlib::CKKS::EncoderConfiguration& conf);

#endif  // FIDESLIB_BERT_TINY_UTILS_CUH