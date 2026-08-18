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

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"

using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

class TransformerTests1 : public GeneralParametrizedTest {};

TEST_P(TransformerTests1, EmbeddingGeneration) {

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc = GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), generalTestParams.GPUs);
    GPUcc->batch = 100;

    // ------- Generate Keys and Move to GPU--------
    keys = cc->KeyGen();
    keys_ = keys;
    cc->EvalMultKeyGen(keys.secretKey);
    auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc);
    eval_key_gpu.Initialize(eval_key);
    GPUcc->AddEvalKey(std::move(eval_key_gpu));

    std::string model_name = "bert-tiny-rte";
    std::string model_path = std::string(root_dir + "examples/bert-tiny/weights-" + model_name);

    // Tokenizer
    std::string sentence =
        "a gorgeous , high-spirited musical from india that exquisitely blends music , dance , song , and high drama .";
    std::string output_file = "tokens_rte1.txt";

    // std::cout << "Tokenizing the following sentence: '" << sentence << "'\n";
    int token_length = tokenizer(sentence, model_name, model_path, output_file);

    EncoderConfiguration conf{.numSlots = (int)cc->GetEncodingParams()->GetBatchSize(),
                              .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())),
                              .token_length = token_length};

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
    encryptMatrixtoGPU(std::string(model_path + "/tokens_rte1.txt"), tokens_gpu, keys.publicKey, GPUcc, conf.numSlots,
                       conf.blockSize, conf.rows, conf.cols, conf.level_matmul);

    if (conf.verbose)
        std::cout << "Block size: " << conf.blockSize << std::endl;
    std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
    GenAndAddRotationKeys(cc, keys, GPUcc, rotation_indices);

    // Bootstrapping Precomputation
    cc->EvalBootstrapSetup({conf.levelsCtS, conf.levelsStC}, {conf.bStepBoot, conf.bStepBoot}, conf.numSlots);
    cc->EvalBootstrapKeyGen(keys.secretKey, conf.numSlots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, GPUcc);

    // Loading weights and biases
    struct PtMasks_GPU masks = GetPtMasks_GPU(GPUcc, cc, conf.numSlots, conf.blockSize, conf.level_matmul + 1);

    struct PtWeights_GPU weights_layer0 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols,
                        conf.level_matmul + 1, conf.num_heads);
    struct PtWeights_GPU weights_layer1 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols,
                        conf.level_matmul + 1, conf.num_heads);

    struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
        getMatrixMatrixProductPrecomputations_GPU(GPUcc, cc, masks, conf.blockSize, conf.bStep, conf.level_matmul + 1,
                                                  conf.level_matmul + 1, conf.prescale, conf.numSlots);

    // TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(GPUcc, cc, conf.blockSize, conf.bStep, conf.level_transpose);
    TransposePrecomputations_GPU Tprecomp_gpu =
        getMatrixTransposePrecomputations_GPU(GPUcc, cc, conf.blockSize, conf.bStep, conf.level_matmul);

    ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/tokens_rte1.txt"), keys.publicKey, conf.numSlots,
                                   conf.blockSize, conf.rows, conf.cols);

    std::string path = model_path + "/rte_validation_short.csv";
    std::string output_path = "a.txt";
    process_sentences_from_csv(path, output_file, model_name, model_path, output_path, conf, keys.publicKey, GPUcc,
                               ct_tokens, weights_layer0, weights_layer1, masks, precomp_gpu, Tprecomp_gpu, cc,
                               keys.secretKey, 0);
}
INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests1, testing::Values(tparams64_15_LLM_flexext));
}  // namespace FIDESlib::Testing
