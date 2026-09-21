#ifndef FIDESLIB_CKKS_TRANSFORMER_CUH
#define FIDESLIB_CKKS_TRANSFORMER_CUH

#include <CKKS/Context.cuh>
#include <cassert>
#include <iostream>
#include <type_traits>
#include <filesystem>
#include <optional>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"

#include "Inputs.cuh"
#include "MatMul.cuh"
#include "MatMul.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>
#include <chrono>

#include <cuda_runtime.h>

using namespace lbcrypto;

namespace FIDESlib::CKKS {

    extern std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
    extern lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
    extern bool TIMING;

    struct EncoderConfiguration {
        bool verbose = true;
        int numSlots;
        int blockSize;
        int token_length;
        int bStep = 4;
        int num_heads = 2;
        uint32_t bStepBoot = 4;
        int bStepAcc = 4;
        uint32_t levelsStC = 3;
        uint32_t levelsCtS = 3;
        int level_matmul = 9;
        bool prescale = false;
        int level_transpose = 7;
        size_t rows = 128;
        size_t cols = 128;
    };

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(
        PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
        TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
        PtMasks_GPU& masks, EncoderConfiguration& conf, int layerNo);

    void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots,
                        bool input_prescaled = false);
                        
    void MatrixAddScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value);

    void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value);
    
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixAdd(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2);
    
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixConcat(std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& matrices, 
                                                                    std::vector<FIDESlib::CKKS::Plaintext>& masks, int blockSize);

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMask(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, FIDESlib::CKKS::Plaintext& mask);

    void MatrixRotate(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int index);
    
    void dropMatrixLevel(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level);
    
    int32_t classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input, 
                                lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey, std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens, 
                                const MatrixMatrixProductPrecomputations_GPU& precomp, PtWeights_GPU& weights_layer, PtMasks_GPU& masks, int numSlots, int blockSize, 
                                int token_length, bool bts, const std::string& output_path);

    std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bstep, int bStepAcc, int colSize = 0);

    int tokenizer(const std::string& sentence, const std::string& dataset, const std::string& model_name, const std::string& model_path, const std::string& output_filename);
    int tokenizer_pair(const std::string& sentence1, const std::string& sentence2, const std::string& model_name, const std::string& output_path, const std::string& output_filename);

    size_t CountNumTokens(const std::string& file_path);

    void process_sentences_from_csv( std::string& file_path,
                                     std::string& output_file,
                                     std::string& model_name,
                                     std::string& model_path,
                                     std::string& output_path,
                                     EncoderConfiguration& base_conf,
                                     lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                    FIDESlib::CKKS::Context& GPUcc,
                                    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                                     PtWeights_GPU& weights_layer0,
                                     PtWeights_GPU& weights_layer1,
                                     PtMasks_GPU& masks,
                                     MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
                                     TransposePrecomputations_GPU& Tprecomp_gpu,
                                    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                    lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk, 
                                    const std::string& dataset,
                                    int test_case = 0, double num_sigma = 0);
}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_TRANSFORMER_CUH