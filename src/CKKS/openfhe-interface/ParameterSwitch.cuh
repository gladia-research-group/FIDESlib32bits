//
// Created by carlosad on 14/09/25.
//

#ifndef PARAMETERSWITCH_CUH
#define PARAMETERSWITCH_CUH

#include "pke/openfhe.h"

namespace FIDESlib {
namespace CKKS {

// A dense->sparse switching key that is applied only at the bottom level (keepQ towers) is read there at digit 0's
// q_0..q_{keepQ-1} and the special primes. Zero every other B limb: digit j of the full hybrid key is a clean RLWE
// sample of the sparse secret modulo QP/Q_j (~2^1464 on the n32 chain), far above what a sparse secret can afford;
// the bottom part alone lives modulo q_0...q_{keepQ-1}*P.
inline void KeepBottomLimbs(const std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>& ek, uint32_t numQ,
                            uint32_t keepQ) {
    std::vector<lbcrypto::DCRTPoly> b = ek->GetBVector();
    for (size_t j = 0; j < b.size(); ++j) {
        for (uint32_t i = 0; i < b[j].GetNumOfElements(); ++i) {
            if (j == 0 && (i < keepQ || i >= numQ))
                continue;
            auto t = b[j].GetElementAtIndex(i);
            t.SetValuesToZero();
            b[j].SetElementAtIndex(i, std::move(t));
        }
    }
    ek->SetBVector(std::move(b));
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createSwitchableContextBasedOnContext(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int limbs, int digits, int hamming_weight);

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>,
                    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
          std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
                           lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
                           const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& a, int hamming_weight_b);

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>,
                    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
          std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
                           lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
                           const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& a, int hamming_weight_b);

}  // namespace CKKS
}  // namespace FIDESlib

#endif  //PARAMETERSWITCH_CUH
