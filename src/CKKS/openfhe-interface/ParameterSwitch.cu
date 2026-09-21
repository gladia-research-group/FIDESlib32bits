//
// Created by carlosad on 14/09/25.
//

#include "CKKS/openfhe-interface/ParameterSwitch.cuh"

namespace FIDESlib {
namespace CKKS {
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createSwitchableContextBasedOnContext(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int limbs, int digits, int hamming_weight) {
    std::shared_ptr<lbcrypto::CryptoParametersCKKSRNS> init_param =
        std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto& init_encode_param = init_param->GetEncodingParams();
    auto& init_elem_param = init_param->GetElementParams();
    auto& init_elem_P_param = init_param->GetParamsP();
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_res;

    lbcrypto::CryptoParametersCKKSRNS param{*init_param};
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    //parameters.SetNoiseEstimate();
    parameters.SetBatchSize(init_encode_param->GetBatchSize());
    parameters.SetDecryptionNoiseMode(param.GetDecryptionNoiseMode());
    //parameters.SetDesiredPrecision();
    parameters.SetDigitSize(digits);
    parameters.SetExecutionMode(param.GetExecutionMode());
    parameters.SetFirstModSize(init_elem_param->GetParams().at(0)->GetModulus().GetMSB());
    parameters.SetInteractiveBootCompressionLevel(
        param.GetMPIntBootCiphertextCompressionLevel() /* param.m_MPIntBootCiphertextCompressionLevel*/);
    parameters.SetKeySwitchTechnique(init_param->GetKeySwitchTechnique());
    parameters.SetMaxRelinSkDeg(init_param->GetMaxRelinSkDeg());
    parameters.SetMultiplicativeDepth(limbs - 1);

    parameters.SetNumAdversarialQueries(param.GetNumAdversarialQueries());
    parameters.SetNumLargeDigits(param.GetNumPartQ());
    parameters.SetPREMode(param.GetPREMode());
    parameters.SetRingDim(param.GetElementParams()->GetRingDimension());
    //auto val = param.GetScalingFactorInt(1);
    //int scale = param.GetScalingFactorIntBig(0).GetMSB();
    //double factor = std::log2(param.GetScalingFactorRealBig(0));
    //double rounded = std::round(param.GetScalingFactorReal(factor));
    // The helper's scale mirrors the source chain's FIRST SCALE PRIME. Three composite traps:
    // (1) read the composite degree from init_param, NOT from the local `param` copy —
    //     CryptoParametersCKKSRNS's copy path DROPS the composite fields (measured: source
    //     degree 2, copy degree 1; NumPartQ is dropped the same way);
    // (2) under composite the first `compositeDegree` primes are the (split) first modulus,
    //     so the first scale prime lives at index compositeDegree (== 1 on classic chains);
    // (3) composite prime pairs straddle 2^(scale/d) to average the target scale, so any
    //     individual prime's MSB can equal MAX_MODULUS_SIZE, which the FLEXIBLEAUTO
    //     validation rejects (needs < MAX). This is a keygen-only depth-`limbs-1` context,
    //     so the exact scale is not load-bearing: clamp into the legal range.
    const int srcCompositeDegree = (int)init_param->GetCompositeDegree();
    int scale = init_elem_param->GetParams().at(srcCompositeDegree)->GetModulus().GetMSB();
    if (srcCompositeDegree > 1 && scale >= (int)MAX_MODULUS_SIZE)
        scale = (int)MAX_MODULUS_SIZE - 1;
    parameters.SetScalingModSize(scale);
    // COMPOSITESCALING source chains: this helper context is the sparse-encapsulation
    // switching-key context — depth `limbs-1` (usually 0), SINGLE-prime by construction
    // (its scale is one prime's MSB). Inheriting COMPOSITESCALING* here would need a
    // composite degree + register word size it has no business having (and CCParams
    // defaults to degree 1 / regWord = NATIVEINT, which OpenFHE rejects at NATIVEINT=32:
    // "composite degree == 1 with register size < 64"). Build it FLEXIBLEAUTO instead —
    // for a keygen-only context the scaling technique does not affect the keys.
    {
        auto st = init_param->GetScalingTechnique();  // from the ORIGINAL (see above)
        if (st == lbcrypto::COMPOSITESCALINGAUTO || st == lbcrypto::COMPOSITESCALINGMANUAL)
            st = lbcrypto::FLEXIBLEAUTO;
        parameters.SetScalingTechnique(st);
    }
    parameters.SetSecretKeyDist(hamming_weight == cc->GetRingDimension() / 2 ? lbcrypto::UNIFORM_TERNARY
                                                                             : lbcrypto::SPARSE_TERNARY);
    parameters.SetSecurityLevel(param.GetStdLevel());
    //parameters.SetStandardDeviation()
    parameters.SetStatisticalSecurity(param.GetStatisticalSecurity());

    cc_res = GenCryptoContext(parameters);
    cc_res->Enable(lbcrypto::PKE | lbcrypto::KEYSWITCH | lbcrypto::LEVELEDSHE);

    return cc_res;
}

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>,
                    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
          std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
                           lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
                           const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& a, int hamming_weight_b) {
    lbcrypto::DCRTPoly::TugType tug;
    lbcrypto::DCRTPoly sNew(tug, cca->GetElementParams(), Format::EVALUATION, hamming_weight_b);
    // sparse key used for the modraising step
    auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
    skNew->SetPrivateElement(std::move(sNew));

    // lbcrypto::PrivateKeyImpl < lbcrypto::DCRTPoly >> nkNewLow(ccb);
    // (*skNew);
    skNew->SetKeyTag(a->GetKeyTag());
    auto scaling =
        std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cca->GetCryptoParameters())->GetScalingTechnique();

    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> atob;
    const int srcCompositeDegree = (int)std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
                                       cca->GetCryptoParameters())
                                       ->GetCompositeDegree();
    if (srcCompositeDegree > 1) {
        // COMPOSITESCALING: the M-4 (dense->sparse) key is a STANDARD hybrid key in the MAIN
        // context — the single-tower helper context cannot represent the d-limb composite
        // bottom (same convention as the openfhe-1.4.2-native32-bootstrap.patch CPU fix).
        // skNew's element was sampled on cca's params above; rewrap it as a cca key.
        auto skNewMain = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cca);
        skNewMain->SetPrivateElement(skNew->GetPrivateElement());
        skNewMain->SetKeyTag(a->GetKeyTag());
        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            cca->GetScheme()->KeySwitchGen(a, skNewMain));
    } else if (scaling != lbcrypto::FLEXIBLEAUTOEXT) {
        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            ccb->GetScheme()->KeySwitchGen(a, skNew));
    } else {
        lbcrypto::DCRTPoly saNew = a->GetPrivateElement();
        saNew.SetElementAtIndex(ccb->GetElementParams()->GetParams().size() - 1, saNew.GetAllElements().back());
        saNew.DropLastElements(saNew.GetAllElements().size() - ccb->GetElementParams()->GetParams().size());
        auto skaNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cca);
        skaNew->SetPrivateElement(std::move(saNew));

        skaNew->SetKeyTag(a->GetKeyTag());

        lbcrypto::DCRTPoly sbNew = skNew->GetPrivateElement();
        sbNew.SetElementAtIndex(ccb->GetElementParams()->GetParams().size() - 1, sbNew.GetAllElements().back());
        sbNew.DropLastElements(sbNew.GetAllElements().size() - ccb->GetElementParams()->GetParams().size());
        auto skbNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(ccb);
        skbNew->SetPrivateElement(std::move(sbNew));

        skbNew->SetKeyTag(a->GetKeyTag());

        atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            ccb->GetScheme()->KeySwitchGen(skaNew, skbNew));
    }
    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> btoa =
        std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
            cca->GetScheme()->KeySwitchGen(skNew, a));

    return {{atob, btoa}, skNew};
}


std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>,
                    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
          std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
                           lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
                           const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& a, int hamming_weight_b)  {
    return createContextSwitchingKeys(cca, ccb, a.secretKey, hamming_weight_b);
}

}  // namespace CKKS
}  // namespace FIDESlib