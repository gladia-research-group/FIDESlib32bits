//
// Created by seyda on 6/6/25.
//

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <CKKS/KeySwitchingKey.cuh>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include "Transformer.cuh"

using namespace std;
using namespace FIDESlib::CKKS;
using namespace lbcrypto;
using namespace std::chrono;

namespace FIDESlib::Testing {

    class MMTests : public GeneralParametrizedTest {};

    TEST_P(MMTests, MatrixMultiplication) {

        bool verbose = true;

        cc->Enable(lbcrypto::PKE);
        cc->Enable(lbcrypto::KEYSWITCH);
        cc->Enable(lbcrypto::LEVELEDSHE);
        cc->Enable(lbcrypto::ADVANCEDSHE);  
        cc->Enable(lbcrypto::FHE);         

        FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
        FIDESlib::CKKS::Context cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
        FIDESlib::CKKS::ContextData& GPUcc = *cc_;

        // Parameters
        GPUcc.batch = 128;
        int matmul_level = 10;
        int bStepAcc = 4;
        int numSlots = cc->GetEncodingParams()->GetBatchSize();
        int blockSize = 128;
        int bStep = 4;
        int num_heads = 2;
        size_t rows = 128;
        size_t cols = 128;
        int token_length = 22;

        std::string dataset = "rte";

        // Inputs
        std::string model_name = "bert-tiny-sst2";
        std::string model_path = std::string(root_dir + "weights/weights-" + model_name);
        std::string output_file = "tokens1.txt";

        std::string sentence = "a bloated gasbag thesis grotesquely impressed by its own gargantuan aura of self-importance ...";
        tokenizer(sentence, dataset, model_name, model_path, output_file);

        // Keys 
        keys = cc->KeyGen();
        keys_ = keys;
        cc->EvalMultKeyGen(keys.secretKey);
        auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
        eval_key_gpu.Initialize(eval_key);
        GPUcc.AddEvalKey(std::move(eval_key_gpu));

        std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(blockSize, bStep, bStepAcc, num_heads);
        GenAndAddRotationKeys(cc, keys, cc_, rotation_indices); 

        struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, numSlots, blockSize, matmul_level+1);

        // // Bootstrapping Precomputation
        // cc->EvalBootstrapSetup({3, 3}, {4, 4}, numSlots);
        // cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

        // FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys_, numSlots, GPUcc);

        auto ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/" + output_file), keys.publicKey, numSlots, blockSize, rows, cols, false);
        // auto ct_tokens = encryptMatrixtoCPU("../weights/dummy/A.txt", keys.publicKey, numSlots, blockSize, rows, cols, false);
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
        encryptMatrixtoGPU(std::string(model_path + "/" + output_file), tokens_gpu, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level);
        // encryptMatrixtoGPU("../weights/dummy/A.txt", tokens_gpu, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level, false);

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q;
        std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>> QKT_heads;

        struct PtWeights_GPU weights_layer0 = GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, numSlots, blockSize, rows, cols, matmul_level+1, num_heads);

        std::vector<std::vector<FIDESlib::CKKS::Plaintext>> b, c;
        encodeMatrixtoGPU("../weights/dummy/B.txt", b, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level, false);
        encodeMatrixtoGPU("../weights/dummy/c.txt", c, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level, false);    
    
        // printMatrix(decryptGPUMatrix(tokens_gpu, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "tokens_gpu: ", true, 0);

        // Bootstrap(tokens_gpu[0][0], numSlots, false);
        printMatrix(decryptGPUMatrix(tokens_gpu, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "tokens_gpu: ", false);

        // ------- PCMM on GPU ------
        if (verbose) std::cout << "Precomp: ";
        auto start_gpu = std::chrono::high_resolution_clock::now();
        // struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(cc_, cc, masks, blockSize, bStep, matmul_level+1, matmul_level+1, false, numSlots);
        struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(cc_, cc, blockSize, bStep, matmul_level+2, matmul_level+2, false, numSlots);
        TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, blockSize, bStep, matmul_level);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;

        dropMatrixLevel(tokens_gpu, matmul_level);
        int N = 1;

        std::cout << "PCMM 1: " << std::endl;

        start_gpu = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++){
            PCMM_GPU(tokens_gpu, weights_layer0.Wk, blockSize, K, precomp_gpu, weights_layer0.bk);
            PCMM_GPU(tokens_gpu, weights_layer0.Wq, blockSize, Q, precomp_gpu, weights_layer0.bq);
        }
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())/N << " ms." << std::endl;

        std::cout << "# limbs: " << K[0][0].getLevel() << " " << K[0][0].NoiseLevel << std::endl;
        printMatrix(decryptGPUMatrix(K, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 32, "K: ", false, 4, true);
        printMatrix(decryptGPUMatrix(Q, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 32, "Q: ", false, 4, true);

        auto K_T = MatrixTranspose_GPU(std::move(K), blockSize, Tprecomp_gpu);
}

INSTANTIATE_TEST_SUITE_P(LLMTests, MMTests, testing::Values(tparams64_15_LLM_flexext));
}  // namespace FIDESlib::Testing
