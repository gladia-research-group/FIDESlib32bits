//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextMultiplication)(benchmark::State& state) {

    if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L <= level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    {
        FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

        std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

        lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
        lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, state.range(3));

        ptxt1->SetLevel(state.range(3));
        ptxt2->SetLevel(state.range(3));
        auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
        FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
        {

            FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
            FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
            kskEval.Initialize(rawKskEval);
            GPUcc->AddEvalKey(std::move(kskEval));
            {
                FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
                {
                    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
                    {
                        state.counters["p_batch"] = state.range(2);
                        state.counters["p_limbs"] = state.range(3);
                        CudaCheckErrorMod;
                        int its = 0;
                        for (auto _ : state) {
                            its++;
                            //std::cout << "Hello" << std::endl;
                            GPUct1.mult(GPUct2, false);
                            GPUct1.NoiseFactor = GPUct2.NoiseFactor;
                            GPUct1.NoiseLevel = 1;
                            //std::cout << "Bye" << std::endl;
                            if constexpr (SYNC)
                                CudaCheckErrorMod;
                            else if (its % 100 == 99)
                                CudaCheckErrorMod;
                        }
                        CudaCheckErrorMod;
                    }
                    CudaCheckErrorMod;
                }
                CudaCheckErrorMod;
            }
            CudaCheckErrorMod;
        }
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextSquaring)(benchmark::State& state) {
    if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L <= level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));

    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(rawKskEval);
    GPUcc->AddEvalKey(std::move(kskEval));

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    int its = 0;
    for (auto _ : state) {
        its++;
        GPUct1.square(false);
        if constexpr (SYNC)
            CudaCheckErrorMod;
        else if (its % 100 == 99)
            CudaCheckErrorMod;
        CudaCheckErrorMod;
        GPUct1.NoiseLevel = 1;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, MultScalar)(benchmark::State& state) {
    if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L <= level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));

    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    CudaCheckErrorMod;
    int its = 0;
    for (auto _ : state) {
        its++;
        GPUct1.multScalar(1.01231331, false);
        if constexpr (SYNC)
            CudaCheckErrorMod;
        else if (its % 100 == 99)
            CudaCheckErrorMod;
        GPUct1.NoiseLevel = 1;
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextMultiplication)
    ->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG});
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextSquaring)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG});
BENCHMARK_REGISTER_F(GeneralFixture, MultScalar)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG});
}  // namespace FIDESlib::Benchmarks