//
// Created by seyda on 6/10/25.
//

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <CKKS/KeySwitchingKey.cuh>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/AccumulateBroadcast.cuh"

using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

class TransformerTests1 : public GeneralParametrizedTest {};

TEST_P(TransformerTests1, EmbeddingGeneration) {
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);

    const bool sparse_encaps = true;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
    FIDESlib::CKKS::Context cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
    FIDESlib::CKKS::ContextData& GPUcc = *cc_;

    GPUcc.batch = 1024;

    // ------- Model + paths ------- 
    std::string dataset = "sst2";

    std::string model_name = "bert-tiny-" + dataset;
    const std::filesystem::path model_path_fs = std::filesystem::path(root_dir) / examples / bert-tiny / "weights" / ("weights-" + model_name);
    std::string model_path = model_path_fs.string();
    std::string tokens_file  = "tokens_" + dataset + ".txt";
    const std::filesystem::path tokens_path = model_path_fs / tokens_file;
    std::string val_csv_file = dataset + "_validation_all.csv";
    const std::filesystem::path val_csv_path = model_path_fs / val_csv_file;

    std::string output_path = "a1.txt";
    std::string path = val_csv_path.string();

    // ------- Generate Keys and Move to GPU --------
    keys = cc->KeyGen();
    keys_ = keys;
    cc->EvalMultKeyGen(keys.secretKey);
    auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
    eval_key_gpu.Initialize(eval_key);
    GPUcc.AddEvalKey(std::move(eval_key_gpu));

    EncoderConfiguration conf{.numSlots = cc->GetEncodingParams()->GetBatchSize(), .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())), .token_length = 63};

    std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
    GenAndAddRotationKeys(cc, keys, cc_, rotation_indices); 

    ct_tokens = encryptMatrixtoCPU(tokens_path.string(), keys.publicKey, conf.numSlots, conf.blockSize, conf.rows, conf.cols);

    // Bootstrapping Precomputation
    cc->EvalBootstrapSetup( {3, 3}, {16, 16}, conf.numSlots, 0, true,
                            GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
    cc->EvalBootstrapKeyGen(keys.secretKey, conf.numSlots); 
    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, cc_);

    // Loading weights and biases
    struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, conf.numSlots, conf.blockSize, conf.level_matmul+1);

    struct PtWeights_GPU weights_layer0 = GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul+1, conf.num_heads);
    struct PtWeights_GPU weights_layer1 = GetPtWeightsGPU(cc_, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul+1, conf.num_heads);

    struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(
        cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul+2, conf.level_matmul+2, conf.prescale, conf.numSlots);

    TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul);

    process_sentences_from_csv(path, tokens_file,
                            model_name, model_path, output_path, conf,
                            keys.publicKey, cc_, ct_tokens,
                            weights_layer0, weights_layer1, masks,
                            precomp_gpu, Tprecomp_gpu,
                            cc, keys.secretKey, dataset);
}
// INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests1, testing::Values(tparams64_15_LLM_flex));
INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests1, testing::Values(tparams64_16_LLM_sq_flex));
}  // namespace FIDESlib::Testing
