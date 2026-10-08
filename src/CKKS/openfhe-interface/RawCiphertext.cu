//
// Created by carlosad on 24/04/24.
//
#include <bit>
#include <cmath>
#include <complex>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/AksKeys.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/SparseB.cuh"

static int sparseBEnv();
static bool sparseBGeometry(int N, int slots, int& n, int& r, int& s, int& b);
#include "CKKS/KskSeedExpand.cuh"
#include "CKKS/openfhe-interface/ParameterSwitch.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "Math.cuh"
using namespace lbcrypto;

/**
* Converts a vector of polynomial limbs to a single flattened array
*/
std::vector<std::vector<uint64_t>> FIDESlib::CKKS::GetRawArray(
    const std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>>& polys) {
    // total size is r * N
    int numRes = polys.size();
    int numElements = (polys[0].GetValues() /*.m_values*/).GetLength();

    std::vector<std::vector<uint64_t>> flattened;
    flattened.reserve(numRes);

    using NativeInt = std::decay_t<decltype(polys[0].GetValues()[0])>;
    for (int r = 0; r < numRes; ++r) {
        const auto& vals = polys[r].GetValues();
        if constexpr (sizeof(NativeInt) == sizeof(uint64_t) && std::is_trivially_copyable_v<NativeInt>) {
            // NativeIntegerT is one uint64_t member, no virtuals; the NativeVector storage is
            // contiguous — range-construct = a single memcpy, no zero-init, no per-element walk.
            const uint64_t* p = reinterpret_cast<const uint64_t*>(&vals[0]);
            flattened.emplace_back(p, p + numElements);
        } else {
            std::vector<uint64_t> limb(numElements);
            for (int i = 0; i < numElements; i++)
                limb[i] = vals[i].ConvertToInt();
            flattened.push_back(std::move(limb));
        }
    }
    return flattened;
};

/**
* Gets the moduli from a vector of polynomial limbs and returns a single array
*/
// const& (was by value): this copied the entire limb vector purely to read
// numRes moduli scalars. Pure parameter-passing change.
static std::vector<uint64_t> GetModuli(const std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>>& polys) {
    int numRes = polys.size();
    std::vector<uint64_t> moduli(numRes);
    for (int r = 0; r < numRes; r++) {
        moduli[r] = polys[r].GetModulus().ConvertToInt();
    }
    return moduli;
};

/**
* Converts a ciphertext from openFHE into the RawCiphertext format
*/
FIDESlib::CKKS::RawCipherText FIDESlib::CKKS::GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                               lbcrypto::Ciphertext<DCRTPoly> ct, int REV) {
    RawCipherText result;  //{ .cc = cc };
    //result.originalCipherText=ct;
    result.numRes = ct->GetElements()[0].GetAllElements().size();
    result.N = ((ct->GetElements()[0].GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
    result.sub_0 = GetRawArray(ct->GetElements()[0].GetAllElements());
    result.sub_1 = GetRawArray(ct->GetElements()[1].GetAllElements());

    // We read (hopefully) in eval form, and OpenFHE should be bit_reversed, so REVERSE == 0.
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
        for (auto& i : result.sub_1)
            bit_reverse_vector(i);
    }
    result.moduli = GetModuli(ct->GetElements()[0].GetAllElements());
    result.format = ct->GetElements()[0].GetFormat();

    result.Noise = ct->GetScalingFactor();       // m_scalingFactor;
    result.NoiseLevel = ct->GetNoiseScaleDeg();  // m_noiseScaleDeg;
    result.keyid = ct->GetKeyTag();              // keyTag;
    result.slots = ct->GetSlots();

    return result;
};

/**
* Converts a ciphertext from the RawCiphertext format back to the OpenFHE ciphertext format*/
void FIDESlib::CKKS::GetOpenFHECipherText(lbcrypto::Ciphertext<DCRTPoly> result, RawCipherText raw, int REV) {

    int size = result->GetElements()[0].GetAllElements().size();
    if (size < raw.numRes) {
        raw.numRes = size;
        raw.sub_0.resize(size);
        raw.sub_1.resize(size);
    }
    assert(result->GetElements()[0].GetAllElements().size() >= raw.numRes);
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : raw.sub_0)
            bit_reverse_vector(i);
        for (auto& i : raw.sub_1)
            bit_reverse_vector(i);
    }
    DCRTPoly sub_0 = result->GetElements().at(0);
    DCRTPoly sub_1 = result->GetElements().at(1);
    auto& dcrt_0 = sub_0.GetAllElements();
    auto& dcrt_1 = sub_1.GetAllElements();
    result->SetLevel(result->GetLevel() + result->GetElements().at(0).GetNumOfElements() - raw.numRes);
    dcrt_0.resize(raw.numRes);
    dcrt_1.resize(raw.numRes);
    for (int r = 0; r < raw.numRes; r++) {
        for (int i = 0; i < raw.N; i++) {
            (*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
            (*dcrt_1.at(r).m_values).at(i).SetValue(raw.sub_1.at(r).at(i));
        }
    }

    //sub_0.m_vectors=dcrt_0;
    //sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.GetParams()->GetParams() /*m_params->m_params*/.size();
         i > sub_0.GetAllElements() /*.m_vectors*/.size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams() /*.m_params*/);
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    for (size_t i = sub_1.GetParams()->GetParams().size(); i > sub_1.GetAllElements().size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_1.GetParams());
        newP->PopLastParam();
        sub_1.m_params.reset(newP);
    }

    std::vector<lbcrypto::DCRTPoly> ct_new = {sub_0, sub_1};
    result->SetElements(ct_new);

    result->SetScalingFactor(raw.Noise);       // Getm_scalingFactor*/ = raw.Noise;
    result->SetNoiseScaleDeg(raw.NoiseLevel);  // /*m_noiseScaleDeg*/ = raw.NoiseLevel;
    result->SetKeyTag(raw.keyid);
    result->SetSlots(raw.slots);
}

void FIDESlib::CKKS::GetOpenFHEPlaintext(lbcrypto::Plaintext result, RawPlainText raw, int REV) {

    assert(result->GetElement<DCRTPoly>().GetAllElements().size() >= raw.numRes);
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : raw.sub_0)
            bit_reverse_vector(i);
    }
    DCRTPoly sub_0 = result->GetElement<DCRTPoly>();
    auto& dcrt_0 = sub_0.GetAllElements();
    result->SetLevel(result->GetLevel() + result->GetElement<DCRTPoly>().GetNumOfElements() - raw.numRes);
    dcrt_0.resize(raw.numRes);
    for (int r = 0; r < raw.numRes; r++) {
        for (int i = 0; i < raw.N; i++) {
            (*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
        }
    }

    //sub_0.m_vectors=dcrt_0;
    //sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.GetParams()->GetParams().size(); i > sub_0.GetAllElements().size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams());
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    result->GetElement<DCRTPoly>() /*encodedVectorDCRT*/ = sub_0;
    result->SetScalingFactor(raw.Noise);       //scalingFactor = raw.Noise;
    result->SetNoiseScaleDeg(raw.NoiseLevel);  // noiseScaleDeg = raw.NoiseLevel;
    result->SetSlots(raw.slots);
    /*
    std::cout << result << std::endl;
bool ok = result->Decode();
    std::cout << ok << " " << result << std::endl;
    */
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                             lbcrypto::Plaintext pt) {
    RawPlainText result;  //{.cc = cc};
    result.originalPlainText = pt;
    if (pt->GetElement<DCRTPoly>().GetAllElements().empty())
        throw std::runtime_error(
            "GetRawPlainText: plaintext CPU payload was released after pinned staging "
            "(FHE_STAGE_RELEASE_CPU) — a second extraction of a staged weight is a bug");
    result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
    result.N = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
    result.sub_0 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
    result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());


    result.format = pt->GetElement<DCRTPoly>().GetFormat();

    if constexpr (REVERSE) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
    }

    result.Noise = pt->GetScalingFactor();
    result.NoiseLevel = pt->GetNoiseScaleDeg();
    result.slots = pt->GetSlots();

    return result;
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                             ReadOnlyPlaintext pt) {
    RawPlainText result;  //{.cc = cc};
    //result.originalPlainText = pt;
    if (pt->GetElement<DCRTPoly>().GetAllElements().empty())
        throw std::runtime_error(
            "GetRawPlainText: plaintext CPU payload was released after pinned staging "
            "(FHE_STAGE_RELEASE_CPU) — a second extraction of a staged weight is a bug");
    result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
    result.N = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
    result.sub_0 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
    result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());


    result.format = pt->GetElement<DCRTPoly>().GetFormat();

    if constexpr (REVERSE) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
    }

    result.Noise = pt->GetScalingFactor();
    result.NoiseLevel = pt->GetNoiseScaleDeg();
    result.slots = pt->GetSlots();

    return result;
}
FIDESlib::CKKS::RawParams FIDESlib::CKKS::GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                       FIDESlib::BOOT_CONFIG boot_conf) {
    RawParams result;
    result.N = cc->GetRingDimension();
    result.logN = std::bit_width((uint32_t)result.N) - 1;
    //cc->GetCryptoParameters()
    result.L = cc->GetCryptoParameters()->GetElementParams()->GetParams().size() - 1;
    //result.L = cc->params->m_params->m_params.size() - 1;
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    result.scalingTechnique = cryptoParams->GetScalingTechnique();
    result.compositeDegree = cryptoParams->GetCompositeDegree();
    //result.qbit = cc->params->m_params->m_params->;
    //auto aux = cc->GetCryptoParameters()->GetParamsPK()->GetParamPartition();

    for (auto& i : /*cc->params->m_params->m_params*/
         cc->GetCryptoParameters()->GetElementParams()->GetParams()) {
        result.moduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
        result.root_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
        result.cyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
    }

    // intnat::ChineseRemainderTransformFTTNat<intnat::NativeVector>::m_rootOfUnityReverseTableByModulus
    for (size_t i = 0; i < result.moduli.size(); ++i) {
        using namespace intnat;
        //  NumberTheoreticTransformNat<NativeVector>().PreCompute
        using FFT = ChineseRemainderTransformFTTNat<NativeVector>;
        auto mapSearch = FFT::m_rootOfUnityReverseTableByModulus.find(result.moduli[i]);
        if (mapSearch == FFT::m_rootOfUnityReverseTableByModulus.end() ||
            mapSearch->second.GetLength() != (size_t)result.N /*CycloOrderHf*/) {
            FFT().PreCompute(result.root_of_unity[i], result.N << 1, result.moduli[i]);
        }

        if (mapSearch == FFT::m_rootOfUnityReverseTableByModulus.end() ||
            mapSearch->second.GetLength() != (size_t)result.N /*CycloOrderHf*/) {
            assert("OpenFHE has not generated the NTT tables we want yet :(" == nullptr);
        }

        {
            int size = FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].GetLength();

            for (int k = 0; k < size; ++k) {
                result.psi[i].push_back(FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].at(k).ConvertToInt<uint64_t>() /*.m_value*/);
                result.psi_inv[i].push_back(
                    FFT::m_rootOfUnityInverseReverseTableByModulus[result.moduli[i]].at(k).ConvertToInt<uint64_t>() /*m_value*/);
            }
            result.N_inv.push_back(FFT::m_cycloOrderInverseTableByModulus.at(result.moduli[i]).at(result.logN).ConvertToInt<uint64_t>() /*.m_value*/);
        }
    }

    //    intnat::NumberTheoreticTransformNat<intnat::NativeVector>().
    //    mubintvec<ubint<unsigned long>>;

    result.ModReduceFactor.resize(result.L + 1);
    for (size_t i = 0; i < result.ModReduceFactor.size(); ++i) {
        result.ModReduceFactor[/*result.L - */ i] = cryptoParams->GetModReduceFactor(i);
    }
    result.ScalingFactorReal.resize(result.L + 1);
    for (size_t i = 0; i < result.ScalingFactorReal.size(); ++i) {
        result.ScalingFactorReal[result.L - i] = cryptoParams->GetScalingFactorReal(i);
    }

    result.ScalingFactorRealBig.resize(result.L + 1);
    for (size_t i = 0; i < result.ScalingFactorRealBig.size(); ++i) {
        result.ScalingFactorRealBig[result.L - i] = cryptoParams->GetScalingFactorRealBig(i);
    }

    {
        auto& src = cryptoParams->m_QlQlInvModqlDivqlModq;
        auto& dest = result.m_QlQlInvModqlDivqlModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(src[i].size());
            for (size_t j = 0; j < src[i].size(); ++j) {
                dest[i][j] = src[i][j].ConvertToInt<uint64_t>();
            }
        }
    }

    /// Key Switching precomputations !!!
    result.dnum = cryptoParams->GetNumPartQ();
    result.K = cryptoParams->GetParamsP()->GetParams().size();
    assert(cryptoParams->GetNumPartQ() == cryptoParams->GetNumberOfQPartitions());
    cryptoParams->GetNumPerPartQ();

    {
        auto& src = cryptoParams->GetParamsP()->m_params;
        for (auto& i : src) {
            result.SPECIALmoduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
            result.SPECIALroot_of_unity.push_back(
                i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
            result.SPECIALcyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
        }
    }

    {
        auto& src = cryptoParams->m_paramsPartQ;
        for (auto& i : src) {
            result.PARTITIONmoduli.emplace_back();
            for (auto& j : i->GetParams()) {
                result.PARTITIONmoduli.back().push_back(
                    j->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
            }
        }
    }

    {
        auto& src = cryptoParams->GetPHatInvModp();
        auto& dest = result.PHatInvModp;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>();
        }
    }

    {
        auto& src = cryptoParams->GetPInvModq();
        auto& dest = result.PInvModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>();  // m_value;
        }
    }

    {
        auto& src = cryptoParams->GetPHatModq();
        auto& dest = result.PHatModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(src[i].size());
            for (size_t j = 0; j < src[i].size(); ++j) {
                dest[i][j] = src[i][j].ConvertToInt<uint64_t>();  // m_value;
            }
        }
    }

    {
        auto& dest = result.PartQlHatInvModq;
        auto& src = cryptoParams->m_PartQlHatInvModq;
        dest.resize(src.size());
        for (size_t k = 0; k < dest.size(); ++k) {
            dest[k].resize(src[k].size());
            for (size_t i = 0; i < dest[k].size(); ++i) {
                dest[k][i].resize(src[k][i].size());
                for (size_t j = 0; j < src[k][i].size(); ++j) {
                    dest[k][i][j] = src[k][i][j].ConvertToInt<uint64_t>();  // m_value;
                }
            }
        }
    }

    {
        auto& dest = result.PartQlHatModp;
        auto& src = cryptoParams->m_PartQlHatModp;
        dest.resize(result.dnum);
        dest.resize(src.size());

        for (size_t k = 0; k < dest.size(); ++k) {
            dest[k].resize(src[k].size());
            for (size_t i = 0; i < dest[k].size(); ++i) {
                dest[k][i].resize(src[k][i].size());
                for (size_t j = 0; j < src[k][i].size(); ++j) {
                    dest[k][i][j].resize(src[k][i][j].size());
                    for (size_t l = 0; l < src[k][i][j].size(); ++l) {
                        dest[k][i][j][l] = src[k][i][j][l].ConvertToInt<uint64_t>();  // m_value;
                    }
                }
            }
        }
    }

    if (cc->GetScheme()->m_FHE) {
        if (boot_conf == FIDESlib::ENCAPS) {
            // r=5 double-angles with a degree-14 Chebyshev refit of (2pi)^(-1/32)*cos(2pi(16y-0.25)/32);
            // depth-equivalent to the degree-32 / r=3 pair (the two extra squarings are exact).
            {
                result.coefficientsCheby = {
                    -5.73829476916553172e-01, 2.63718536771466207e-02,  -9.15574236933522245e-01,
                    -3.08975417541565676e-02, 2.85601052580403802e-01,  4.83129148738224799e-03,
                    -2.74350673185783482e-02, -3.16919293037672828e-04, 1.31295181914509542e-03,
                    1.15825536926191591e-05,  -3.79010152666429525e-05, -2.71035793019274615e-07,
                    7.33952552563662122e-07,  4.41686895092293209e-09,  -1.03169742989480305e-08};  // degree 14, r=5
                result.doubleAngleIts = 5;
            }
            result.bootK = 16.0;
            result.sparse_encaps = true;
            //result.bootK = 1.0;  // do not divide by k as we already did it during precomputation
        } else if (boot_conf == FIDESlib::ENCAPS_2) {
            result.coefficientsCheby = {
                0.24554573401685137,    -0.047919064883347899,   0.28388702040840819,      -0.029944538735513584,
                0.35576522619036460,    0.015106561885073030,    0.29532946674499999,      0.071203602333739374,
                -0.10347347339668074,   0.044997590512555294,    -0.42750712431925747,     -0.090342129729094875,
                0.36762876269324946,    0.049318066039335348,    -0.14535986272411980,     -0.015106938483063579,
                0.035951935499240355,   0.0031036582188686437,   -0.0062644606607068463,   -0.00046609430477154916,
                0.00082128798852385086, 0.000053910533892372678, -0.000084551549768927401, -4.9773801787288514e-6,
                7.0466620439083618e-6,  3.7659807574103204e-7,   -4.8648510153626034e-7,   -2.3830267651437146e-8,
                2.8329709716159918e-8,  1.2817720050334158e-9,   -1.4122220430105397e-9,   -5.9306213139085216e-11,
                6.3298928388417848e-11};

            // result.bootK = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_SPARSE;
            result.bootK = 16.0;
            result.sparse_encaps = true;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE + 1;
        } else if (boot_conf == FIDESlib::SPARSE) {
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
            // k = K_SPARSE;
            result.bootK = cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY
                               ? 1.0
                               : std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                                     ->K_SPARSE;  // do not divide by k as we already did it during precomputation
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
            result.sparse_encaps = cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY ? false : true;
            // } else if (cryptoParams->GetSecretKeyDist() == SPARSE_ENCAPSULATED) {    // Switch to this with OpenFHE v1.4, remove the flag
        } else if (boot_conf == FIDESlib::UNIFORM) {
            // result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
            result.bootK = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                               ->K_UNIFORM;  // lbcrypto::FHECKKSRNS::K_UNIFORM;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
        } else if (boot_conf == FIDESlib::UNIFORM_2) {
            result.coefficientsCheby = {
                2.207266599864877165693144e-01,  -2.682587999577537660883531e-03, 2.381211781853223574678680e-01,
                -2.217484225267288364819018e-03, 2.812572129228093631425622e-01,  -1.118768081838177096479225e-03,
                3.175289620003503565648373e-01,  7.418465825669621170612711e-04,  2.838568337485863901648031e-01,
                2.959589426714506928128845e-03,  1.111409007132102000348084e-01,  4.045004272966680990142319e-03,
                -1.773752281024201515879923e-01, 1.966283967074625767951224e-03,  -3.431231212979015121611326e-01,
                -2.725087963359526001261290e-03, -7.807165506705357471695095e-02, -3.945018814199019625832410e-03,
                3.567953191665136358778909e-01,  2.327088764749175604090725e-03,  7.009711220121370156554974e-02,
                3.696241718778929454675142e-03,  -4.332158158181031448741294e-01, -5.611596576812074611828596e-03,
                4.036825197782618612762917e-01,  3.850187461178859998217616e-03,  -2.204550282997065902002021e-01,
                -1.747584499380709470439665e-03, 8.550195814336092325902428e-02,  5.904770390752328004455030e-04,
                -2.553292477957162798229973e-02, -1.575954197449093332605158e-04, 6.145509744931569942605343e-03,
                3.446140101881852877904744e-05,  -1.228526976412591372248007e-03, -6.331581547383084787063417e-06,
                2.084131601810473930890683e-04,  9.958114491488687767422640e-07,  -3.049845562371355744382857e-05,
                -1.360239966314108502055247e-07, 3.899969756237415622062044e-06,  1.632622019731083661639916e-08,
                -4.404121172058021798864386e-07, -1.738463098910276156704066e-09, 4.430993120182116282036096e-08,
                1.655643173042383465668248e-10,  -4.003300390835467618282280e-09, -1.412477955902056943370032e-11,
                3.507103856919978252042301e-10};
            result.bootK = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                               ->K_UNIFORM;  // lbcrypto::FHECKKSRNS::K_UNIFORM;
            result.doubleAngleIts = 7;
        }

    }

    result.p = cryptoParams->GetPlaintextModulus();

    return result;
}
FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetKeySwitchKey(
    std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> ek) {

    std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
    std::vector<std::vector<std::vector<uint64_t>>> b;
    std::string keytag;

    //for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
    for (/*auto a_raw = ek.get()->m_rKey;*/ auto& i : {ek->GetAVector(), ek->GetBVector()}) {
        std::vector<std::vector<std::vector<uint64_t>>> a_inner;
        std::vector<std::vector<uint64_t>> a_inner_moduli;
        for (auto& j : i) {
            auto v = GetRawArray(j.GetAllElements() /*.m_vectors*/);
            a_inner_moduli.emplace_back();
            auto& a_aux = a_inner_moduli.back();
            for (auto& p : j.GetParams()->GetParams() /*m_params->m_params*/) {
                a_aux.push_back(p->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
            }
            a_inner.push_back(v);
        }
        a.push_back(a_inner);
        a_moduli.push_back(a_inner_moduli);
    }
    keytag = ek->GetKeyTag();

    RawKeySwitchKey raw(std::move(a_moduli), std::move(a), std::move(b), std::move(keytag));
#ifdef OPENFHE_HAS_KSKA_SEED
    raw.a_seed = ek->GetASeed();   // key seed for on-GPU regeneration of the `a` half
#endif
    return raw;
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {

    //const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    auto& keyMap = keys.publicKey->GetCryptoContext()->GetAllEvalMultKeys();
    //lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        const std::vector<EvalKey<DCRTPoly>>& key = keyMap[keys.secretKey->GetKeyTag()];
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));
        return GetKeySwitchKey(ek);
    } else {
        assert("EvalKey is not present !!!" == nullptr);
    }

    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(
    const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap = cc->GetAllEvalMultKeys();
    //lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
        const std::vector<EvalKey<DCRTPoly>>& key = keyMap[publicKey->GetKeyTag()];
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));

        return GetKeySwitchKey(ek);
    } else {
        assert("EvalKey is not present !!!" == nullptr);
    }

    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(
    const KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    auto& keyMap = cc->GetAllEvalAutomorphismKeys();

    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        auto& keyMap2 = keyMap[keys.secretKey->GetKeyTag()];
        uint32_t x = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);
        if (keyMap2->find(x) == keyMap2->end()) {
            cc->EvalAtIndexKeyGen(keys.secretKey, {index});
        }
        {
            const auto& key = keyMap2->at(x);
            assert(key != nullptr);
            const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

            return GetKeySwitchKey(ek);
        }
    } else {
        assert("RotKey is not present !!!" == nullptr);
        std::cout << "RotKey is not present !!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(
    const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int index) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap = cc->GetAllEvalAutomorphismKeys();

    if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
        auto& keyMap2 = keyMap[publicKey->GetKeyTag()];
        uint32_t x = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);

        {
            const auto& key = keyMap2->at(x);
            assert(key != nullptr);
            const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

            return GetKeySwitchKey(ek);
        }
    } else {
        assert("RotKey is not present !!!" == nullptr);
        std::cout << "RotKey is not present !!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetConjugateKeySwitchKey(
    const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap2 = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

    if (keyMap2.find(2 * cc->GetRingDimension() - 1) != keyMap2.end()) {
        const auto& key = keyMap2.at(2 * cc->GetRingDimension() - 1);
        assert(key != nullptr);
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);
        //std::cout << std::endl << "Clave " << ek->GetKeyTag() << "\n";
        return GetKeySwitchKey(ek);
    } else {
        assert("RotKey is not present for rotation !!!" == nullptr);
        std::cout << "RotKey is not present for conjugation!!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

#include "CKKS/BootstrapPrecomputation.cuh"

#include "CKKS/KeySwitchingKey.cuh"

#include "CKKS/LimbPartition.cuh"

#include "CKKS/RNSPoly.cuh"

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> FIDESlib::CKKS::GenRotationKeys(
    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
        return GenRotationKeys(keys.secretKey, indexes);
}

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> FIDESlib::CKKS::GenRotationKeys(
    const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
    std::set<int> indexes2(indexes.begin(), indexes.end());
    std::vector<int> indexes3;
    for (int i : indexes2) {
        if (i) {
            indexes3.emplace_back(i);
        }
    }
    auto evalKeys = cc->GetScheme()->EvalAtIndexKeyGen(nullptr, keys, indexes3);
    CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag());
    return evalKeys;
    //std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
    //    ->cc->EvalAtIndexKeyGen(keys.secretKey, indexes3);
}

void FIDESlib::CKKS::AddRotationKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                     FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    std::set<int> indexes2(indexes.begin(), indexes.end());
    std::vector<int> indexes3;
    for (int i : indexes2) {
        if (i && !GPUcc->HasRotationKey(i, publicKey->GetKeyTag())) {
            indexes3.emplace_back(i);
        }
    }
    for (int i : indexes3) {
        auto clave_rotacion = FIDESlib::CKKS::GetRotationKeySwitchKey(publicKey, i);
        //std::cout << "Load rotation key " << i << std::endl;
        FIDESlib::CKKS::KeySwitchingKey clave_rotacion_gpu(GPUcc);
        clave_rotacion_gpu.Initialize(clave_rotacion);
        GPUcc->AddRotationKey(i, std::move(clave_rotacion_gpu));
    }
}

void FIDESlib::CKKS::GenAndAddRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                           const KeyPair<lbcrypto::DCRTPoly>& keys, FIDESlib::CKKS::Context& GPUcc,
                                           std::vector<int> indexes) {
    GenRotationKeys(keys, indexes);
    AddRotationKeys(keys.publicKey, GPUcc, indexes);
}

constexpr bool remove_extension = false;
constexpr bool MAKE_CTS_LT_FRIENDLY = true;
constexpr bool MAKE_STC_LT_FRIENDLY = true;

std::vector<int> FIDESlib::CKKS::GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots,
                                                     FIDESlib::CKKS::BootstrapPrecomputation* result_) {
    // ContextData& GPUcc = *GPUcc_;
    std::vector<int> indexes;
    BootstrapPrecomputation result;
    auto precom =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

    if (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1 &&
        precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) {

        result.LT.slots = slots;
        result.LT.bStep = (precom->m_dim1 == 0) ? ceil(sqrt(slots)) : precom->m_dim1;

        for (int i = 1; i < result.LT.bStep; ++i) {
            indexes.push_back(i);
        }

#if AFFINE_LT
        indexes.push_back(result.LT.bStep);
#else
        for (int i = result.LT.bStep; i < result.LT.slots; i += result.LT.bStep) {
            indexes.push_back(i);
        }
#endif
    } else {
        {  // CoeffToSlots metadata
            uint32_t M = cc->GetCyclotomicOrder();
            uint32_t N = cc->GetRingDimension();
            int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            int32_t stop = -1;
            int32_t flagRem = 0;

            auto algo = cc->GetScheme();

            if (remCollapse != 0) {
                stop = 0;
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                result.CtS.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                    if (flagRem == 1 && i == 0) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        result.CtS[i].bStep = gRem;
                        result.CtS[i].gStep = bRem;
                        result.CtS[i].slots = numRotationsRem;
                        result.CtS[i].rotIn.resize(gRem);
                        result.CtS[i].rotOut.resize(bRem);
                    } else {
                        result.CtS[i].bStep = g;
                        result.CtS[i].gStep = b;
                        result.CtS[i].slots = numRotations;
                        result.CtS[i].rotIn.resize(g);
                        result.CtS[i].rotOut.resize(b);
                    }
                }

                for (int32_t s = levelBudget - 1; s > stop; s--) {
                    for (int32_t j = 0; j < g; j++) {
                        result.CtS[s].rotIn[j] =
                            ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) *
                                               (1 << ((s - flagRem) * layersCollapse + remCollapse)),
                                           slots);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        result.CtS[s].rotOut[i] =
                            ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
                    }
                }

                if (flagRem) {
                    for (int32_t j = 0; j < gRem; j++) {
                        result.CtS[stop].rotIn[j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        result.CtS[stop].rotOut[i] = ReduceRotation((gRem * i), M / 4);
                    }
                }

                if constexpr (AFFINE_LT && MAKE_CTS_LT_FRIENDLY) {
                    for (int32_t s = 0; s < levelBudget; s++) {
                        int offset = result.CtS.at(s).rotIn[0];
                        for (auto& i : result.CtS.at(s).rotIn) {
                            i = (i - offset);
                        }
                        for (auto& i : result.CtS.at(s).rotOut) {
                            i = (i + offset);
                        }
                    }
                }
            }

            //std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
        }

        {  // SlotToCoeff metadata
            uint32_t M = cc->GetCyclotomicOrder();
            uint32_t N = cc->GetRingDimension();

            int32_t levelBudget = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            auto algo = cc->GetScheme();

            int32_t flagRem = 0;

            if (remCollapse != 0) {
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                result.StC.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {

                    if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        result.StC[i].bStep = gRem;
                        result.StC[i].gStep = bRem;
                        result.StC[i].slots = numRotationsRem;
                        result.StC[i].rotIn.resize(gRem);
                        result.StC.at(i).rotOut.resize(bRem);
                    } else {
                        result.StC[i].bStep = g;
                        result.StC[i].gStep = b;
                        result.StC[i].slots = numRotations;
                        result.StC[i].rotIn.resize(g);
                        result.StC.at(i).rotOut.resize(b);
                    }
                }

                for (int32_t s = 0; s < levelBudget - flagRem; s++) {
                    for (int32_t j = 0; j < g; j++) {
                        result.StC.at(s).rotIn.at(j) = ReduceRotation(
                            (j - int32_t((numRotations + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        result.StC.at(s).rotOut.at(i) = ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4);
                    }
                }

                if (flagRem) {
                    int32_t s = levelBudget - flagRem;
                    for (int32_t j = 0; j < gRem; j++) {
                        result.StC.at(s).rotIn.at(j) = ReduceRotation(
                            (j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        result.StC.at(s).rotOut.at(i) = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4);
                    }
                }

                if constexpr (AFFINE_LT && MAKE_STC_LT_FRIENDLY) {
                    for (int32_t s = 0; s < levelBudget; s++) {
                        int offset = result.StC.at(s).rotIn[0];
                        for (auto& i : result.StC.at(s).rotIn) {
                            i = (i - offset);
                        }
                        for (auto& i : result.StC.at(s).rotOut) {
                            i = (i + offset);
                        }
                    }
                    /*
                            for (int32_t s = 0; s < levelBudget; s++) {
                                int offset = result.StC.at(s).rotIn[0];
                                for (auto& i : result.StC.at(s).rotIn) {
                                    i = (i - offset);
                                }
                                for (auto& i : result.StC.at(s).rotOut) {
                                    i = (i + offset);
                                }
                            }
                            */
                }
            }

            //std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
        }

        std::reverse(result.CtS.begin(), result.CtS.end());

        int acc_offset = 0;
        if constexpr (AFFINE_LT && MAKE_CTS_LT_FRIENDLY) {
            for (int32_t s = 0; s < result.CtS.size(); s++) {
                int offset = result.CtS.at(s).rotOut[0];
                acc_offset += result.CtS.at(s).rotOut[0];
                for (int i = 1; i < result.CtS.at(s).gStep; ++i) {
                    result.CtS.at(s).rotOut[i] -= offset;
                    result.CtS.at(s).rotOut[i] %= std::min(2 * slots, (int)cc->GetRingDimension() / 2);
                }
                //offset = result.CtS.at(s).rotOut[0];

                for (int i = 0; i < result.CtS.at(s).gStep; ++i) {
                    for (int j = 0; j < result.CtS.at(s).bStep; ++j) {
                        if (i * result.CtS.at(s).bStep + j < result.CtS.at(s).slots) {
                            if (j > 0) {
                                if (result.CtS.at(s).rotIn[j] - result.CtS.at(s).rotIn[j - 1] !=
                                    result.CtS.at(s).rotIn[1] - result.CtS.at(s).rotIn[0]) {
                                    int new_in = result.CtS.at(s).rotIn[j - 1] + result.CtS.at(s).rotIn[1] -
                                                 result.CtS.at(s).rotIn[0];
                                    /*
                                            result.CtS.at(s).A[i * result.CtS.at(s).bStep + j].automorph(
                                                ReduceRotation(new_in - result.CtS.at(s).rotIn[j], M / 4));
                                            */
                                    result.CtS.at(s).rotIn[j] = new_in;
                                }
                            }
                        }
                    }
                }
            }
        }

        if constexpr (AFFINE_LT && MAKE_STC_LT_FRIENDLY) {
            for (int32_t s = 0; s < result.StC.size(); s++) {
                int offset = result.StC.at(s).rotOut[0];
                acc_offset += result.StC.at(s).rotOut[0];
                for (int i = 1; i < result.StC.at(s).gStep; ++i) {
                    result.StC.at(s).rotOut[i] -= offset;
                    result.StC.at(s).rotOut[i] %= std::min(2 * slots, (int)cc->GetRingDimension() / 2);
                }

                for (int i = 0; i < result.StC.at(s).gStep; ++i) {
                    for (int j = 0; j < result.StC.at(s).bStep; ++j) {
                        if (i * result.StC.at(s).bStep + j < result.StC.at(s).slots) {
                            if (j > 0) {

                                if (result.StC.at(s).rotIn[j] - result.StC.at(s).rotIn[j - 1] !=
                                    result.StC.at(s).rotIn[1] - result.StC.at(s).rotIn[0]) {
                                    int new_in = result.StC.at(s).rotIn[j - 1] + result.StC.at(s).rotIn[1] -
                                                 result.StC.at(s).rotIn[0];
                                    /*
                                            result.StC.at(s).A[i * result.StC.at(s).bStep + j].automorph(
                                                ReduceRotation(new_in - result.StC.at(s).rotIn[j], M / 4));
                                                */
                                    result.StC.at(s).rotIn[j] = new_in;
                                }
                            }
                        }
                    }
                }
            }
        }

        indexes.emplace_back(acc_offset);

        for (auto& v : {&result.CtS, &result.StC}) {
            for (auto& i : *v) {
                for (auto& j : i.rotIn) {
                    indexes.push_back(j);
                }
#if AFFINE_LT
                // We do not include rotOut[0], it is later set to 0 but the last that is set to acc_offset
                for (auto& j : {/*i.rotOut[0],*/ i.rotOut.size() > 1 ? i.rotOut[1] /*- i.rotOut[0]*/ : 0}) {
                    indexes.push_back(j);
                }
#else
                for (auto& j : i.rotOut) {
                    if (j && !GPUcc.HasRotationKey(j)) {
                        indexes.push_back(j);
                    }
                }
#endif
            }
        }
    }

    int slots_transform = std::min((int)slots * 2, (int)cc->GetCyclotomicOrder() / 4);
    for (auto& i : indexes) {
        auto j_ = i % slots_transform;
        if (j_ < 0)
            j_ += slots_transform;
        if (j_ > slots_transform / 2)
            j_ += cc->GetCyclotomicOrder() / 4 - slots_transform;
        i = j_;
    }

    if (cc->GetRingDimension() / 2 != slots) {
        // FIDESLIB_ACCUM_BSTEP: the sparse fold's baby step (power of two, default 4), either one value for every
        // route or "slots:bStep,..." per route (e.g. "1:2,512:8"). Rotation keys follow the choice.
        int bStep = 4;
        if (const char* e = std::getenv("FIDESLIB_ACCUM_BSTEP"); e && *e) {
            const std::string v(e);
            if (v.find(':') == std::string::npos) {
                bStep = std::atoi(v.c_str());
            } else {
                size_t i = 0;
                while (i < v.size()) {
                    size_t j = v.find(',', i);
                    if (j == std::string::npos) j = v.size();
                    const std::string tok = v.substr(i, j - i);
                    const size_t c = tok.find(':');
                    if (c != std::string::npos && std::atoi(tok.substr(0, c).c_str()) == slots)
                        bStep = std::atoi(tok.substr(c + 1).c_str());
                    i = j + 1;
                }
            }
            if (bStep < 2 || (bStep & (bStep - 1)))
                throw std::runtime_error("FIDESLIB_ACCUM_BSTEP: baby step must be a power of two >= 2");
        }
        result.accumulate_bStep = bStep;
        std::vector<int> rotations = GetAccumulateRotationIndices(bStep, slots, cc->GetRingDimension() / 2 / slots);
        for (auto idx : rotations) {
            indexes.push_back(idx);
        }
    }
    if (slots == 1) {  // SPRU for the s = 1 route (CKKS/Spru.cuh): trace, product and recombination rotations, full keys
        const char* e = std::getenv("FIDESLIB_SPRU");
        if (e && std::atoi(e) > 0) {
            const int h = std::atoi(e) > 1 ? std::atoi(e) : 64, n = 2, N2 = (int)cc->GetRingDimension() / 2;
            for (int idx : GetAccumulateRotationIndices(4, h * n, N2 / (h * n)))
                indexes.push_back(idx);
            for (int s = n; s < h * n; s <<= 1)
                indexes.push_back(s);
            indexes.push_back(1);
        }
    }
    {   // lever B: full-ring indices (not reduced to the 2*slots view: the partial-sum layouts are not periodic)
        int n, r, sB, b;
        if (sparseBGeometry((int)cc->GetRingDimension(), slots, n, r, sB, b)) {
            for (int i = 1; i < b; ++i)
                indexes.push_back(i);           // baby steps
            indexes.push_back(b);               // giant step (Horner, one key)
            for (int idx : GetAccumulateRotationIndices(4, sB, r / 4))
                indexes.push_back(idx);         // CtS block partial sum
            for (int idx : GetAccumulateRotationIndices(4, n, r / 2))
                indexes.push_back(idx);         // StC block partial sum
            indexes.push_back(n / 2);
        }
    }

    if (result_)
        *result_ = std::move(result);
    return indexes;
}
void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots) {
    GenBootstrapKeys(keys.secretKey, slots);
}
void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, int slots) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
    std::vector<int> indexes = GetBootstrapIndexes(cc, slots, nullptr);
    cc->EvalMultKeyGen(keys);

    auto evalKeys = GenRotationKeys(keys, GetBootstrapIndexes(cc, slots, nullptr));
    auto conjKey =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ConjugateKeyGen(keys);

    (*evalKeys)[cc->GetCyclotomicOrder() - 1] = conjKey;

    auto cc_switch = CKKS::createSwitchableContextBasedOnContext(cc, 1, 1, cc->GetRingDimension() / 2);

    //auto keys_switch = cc_switch->KeyGen();
    auto [swtch, sk_sparse] = CKKS::createContextSwitchingKeys(cc, cc_switch, keys, 32);
    (*evalKeys)[cc->GetCyclotomicOrder() - 2] = swtch.first;
    (*evalKeys)[cc->GetCyclotomicOrder() - 4] =
        swtch.second;  // Use a pair index so no collision with 5^k mod 2N exists

    // We can discard sk_sparse and cc_switch

    CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(
        evalKeys, keys->GetKeyTag());  // Reinsert all keys to add the particular conj key and sse keys
}

void FIDESlib::CKKS::AddBootstrapKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots,
                                      FIDESlib::CKKS::Context& GPUcc_) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    FIDESlib::CKKS::BootstrapPrecomputation& result = GPUcc_->GetBootPrecomputation(slots);

    ContextData& GPUcc = *GPUcc_;
    std::vector<int> indexes = GetBootstrapIndexes(cc, slots, nullptr);

    //std::cout << "Add eval key" << std::endl;
    {
        KeySwitchingKey ksk(GPUcc_);
        RawKeySwitchKey rksk = GetEvalKeySwitchKey(publicKey);
        ksk.Initialize(rksk);
        GPUcc.AddEvalKey(std::move(ksk));
    }

    //std::cout << "Add conjugate key" << std::endl;
    {
        KeySwitchingKey ksk(GPUcc_);
        RawKeySwitchKey rksk = GetConjugateKeySwitchKey(publicKey);
        ksk.Initialize(rksk);
        GPUcc.AddRotationKey(GPUcc.N * 2 - 1, std::move(ksk));
    }
    //std::cout << "Add rotation keys" << std::endl;

    AddRotationKeys(publicKey, GPUcc_, indexes);

    if (GPUcc.param.raw->sparse_encaps) {
        auto& evalKeys = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

        if (GPUcc.compositeDegree() > 1) {
            // COMPOSITESCALING: both secret-switching keys are MAIN-context standard hybrid
            // keys (see BootstrapPrecomputation::sparse_atob) — no helper GPU context at all.
            result.sparse_atob = std::make_shared<FIDESlib::CKKS::KeySwitchingKey>(GPUcc_);
            {
                std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                    std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
                        evalKeys[2 * GPUcc.N - 2]);
                FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(res);
                result.sparse_atob->Initialize(rawKskEval);
            }
            result.sparse_btoa = std::make_shared<FIDESlib::CKKS::KeySwitchingKey>(GPUcc_);
            {
                std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                    std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
                        evalKeys[2 * GPUcc.N - 4]);
                FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(res);
                result.sparse_btoa->Initialize(rawKskEval2);
                // GHS form for the small raised ciphertext (FIDESLIB_BTS_SHIFT): sum the digit keys limb-wise.
                // sum_j D^_j D*_j == 1 (mod Q) and the P factor kills the mod-P ambiguity, so (sum b_j, sum a_j)
                // encrypts P*s~ under s with noise sum e_j.
                if (const char* e = std::getenv("FIDESLIB_BTS_SHIFT"); e && std::atoi(e) > 0) {
                    const auto& A = rawKskEval2.r_key[0];  // [digit][limb][coef]
                    const auto& B = rawKskEval2.r_key[1];
                    const auto& M = rawKskEval2.r_key_moduli[0];
                    std::vector<uint64_t> moduli;
                    for (int i = 0; i <= GPUcc.L; ++i) moduli.push_back(GPUcc.prime[i].p);
                    for (auto& sp : GPUcc.specialPrime) moduli.push_back(sp.p);
                    const size_t nl = moduli.size(), N = GPUcc.N;
                    std::vector<std::vector<uint64_t>> sa(nl, std::vector<uint64_t>(N, 0)), sb(nl, std::vector<uint64_t>(N, 0));
                    for (size_t j = 0; j < A.size(); ++j)
                        for (size_t k = 0; k < A[j].size(); ++k) {
                            const uint64_t q = M[j][k];
                            size_t l = 0;
                            while (l < nl && moduli[l] != q) ++l;
                            if (l == nl) throw std::runtime_error("ghs_btoa: key limb modulus not in Q+P");
                            for (size_t n = 0; n < N; ++n) {
                                sa[l][n] = (sa[l][n] + A[j][k][n]) % q;
                                sb[l][n] = (sb[l][n] + B[j][k][n]) % q;
                            }
                        }
                    result.ghs_btoa = FIDESlib::CKKS::MakeGhsKeyStage(GPUcc_, sa, sb, moduli);
                    std::cerr << "[bts_shift] GHS btoa key built from " << A.size() << " digit keys over " << nl << " limbs\n";
                }
            }
            // result.sparse_context stays unset: the d>1 raise never enters the helper path,
            // and an accidental .lock() should fail loudly rather than hand back a live context.
        } else {
            auto cc_switch = CKKS::createSwitchableContextBasedOnContext(cc, 1, 1, cc->GetRingDimension() / 2);

            FIDESlib::CKKS::RawParams raw_param2 = FIDESlib::CKKS::GetRawParams(cc_switch);
            //FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), devices};
            FIDESlib::CKKS::Context cc_switch_ =
                CKKS::GenCryptoContextGPU(GPUcc.param.adaptTo(raw_param2), GPUcc.GPUid);
            FIDESlib::CKKS::ContextData& GPUcc2 = *cc_switch_;

            //std::cout << "Add atob key" << std::endl;
            FIDESlib::CKKS::KeySwitchingKey ksk_atob(cc_switch_);

            {
                std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                    std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
                        evalKeys[2 * GPUcc.N - 2]);
                FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(res);
                ksk_atob.Initialize(rawKskEval);
            }
            //std::cout << "Add btoa key" << std::endl;
            FIDESlib::CKKS::KeySwitchingKey ksk_btoa(GPUcc_);
            {
                std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                    std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
                        evalKeys[2 * GPUcc.N - 4]);
                FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(res);
                ksk_btoa.Initialize(rawKskEval2);
            }

            CKKS::AddSecretSwitchingKey(std::move(ksk_atob), std::move(ksk_btoa));

            result.sparse_context = cc_switch_;
        }
    }

    std::cout << "Rotation keys loaded: " << GPUcc.precom.keys.begin()->second.rot_keys.size() << " ~ "
              << 2 * ((long long)GPUcc.precom.keys.begin()->second.rot_keys.size() * GPUcc.dnum *
                      (GPUcc.L + GPUcc.K + 1) * GPUcc.N * (NATIVEINT / 8) / (1 << 20))
              << "MB" << std::endl;
}


// FIDESLIB_BTS_STC_FIRST (lever A): the StC stages re-levelled to the input side of the bootstrap. Entry limb index =
// 2d-1 (ModRaise's adjust + rescale) + d per StC stage; stage s lands at entry - s*d. relevelPlaintext keeps the
// encoded values (scaleDec included) across the FLEXIBLEAUTO scale change, one rounding per coefficient.

// FIDESLIB_BTS_SPARSE_B (lever B): the sparse route's geometry; s = 0 when the route does not qualify (dense, or
// n <= r/2 where the paper's Algorithms 2/3 apply — not built, the single-LT route is already depth 1 + 1).
static int sparseBEnv() {
    const char* e = std::getenv("FIDESLIB_BTS_SPARSE_B");
    return e ? std::atoi(e) : 0;
}
static bool sparseBGeometry(int N, int slots, int& n, int& r, int& s, int& b) {
    if (sparseBEnv() <= 0 || slots == N / 2 || slots < 1)
        return false;
    n = 2 * slots;
    r = N / n;
    if (!(n > r / 2))
        return false;
    s = 2 * n / r;
    // baby-step size: default = all s diagonals hoisted (one ModUp shared by s-1 rotations, no un-hoisted giant steps —
    // the shipped stages hoist 16); FIDESLIB_BTS_SPARSE_B_BSTEP overrides (sqrt(s) is the paper's un-hoisted optimum)
    b = s;
    if (const char* e = std::getenv("FIDESLIB_BTS_SPARSE_B_BSTEP"); e && std::atoi(e) > 0)
        b = std::min(s, std::atoi(e));
    return true;
}
static double envDouble(const char* k, double dflt) {
    const char* e = std::getenv(k);
    return (e && *e) ? std::atof(e) : dflt;
}

// Host build of the diagonal plaintexts p_i / q_i (paper Sec. 3.3) in OpenFHE's slot/root conventions:
// U(j, l) = ksi^(l 5^j), ksi = exp(2 pi i / (4 slots)), j < n/2, l < n (U = [U0 | i U0] of EvalBootstrapSetup).
// CtS diagonals V = gc * conj(U)^T (gc carries the EvalMod input normalization for K), StC diagonals from gd * U
// (gd = scaleDec = q0 / sf as the shipped StC). BSGS pre-rotation: pts[k] = rot_{-(k/b) b}(diag_k) so that
// LinearTransform(stride 1, offset 0) computes sum_k diag_k (.) rot_k(ct).
static void buildSparseB(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, FIDESlib::CKKS::Context& GPUcc_,
                         FIDESlib::CKKS::ContextData& GPUcc, int slots, FIDESlib::CKKS::BootstrapPrecomputation& result) {
    using cd = std::complex<double>;
    int n, r, s, b;
    if (!sparseBGeometry(GPUcc.N, slots, n, r, s, b))
        return;
    const int N2 = GPUcc.N / 2;
    const uint32_t m = 4 * slots, mmask = m - 1;
    std::vector<uint32_t> rotGroup(slots);
    for (uint32_t j = 0, f = 1; j < (uint32_t)slots; ++j, f = (f * 5) & mmask)
        rotGroup[j] = f;
    std::vector<cd> ksi(m);
    for (uint32_t k = 0; k < m; ++k)
        ksi[k] = cd(std::cos(2 * M_PI * k / m), std::sin(2 * M_PI * k / m));
    auto U = [&](int j, int l) { return ksi[((uint32_t)l * rotGroup[j]) & mmask]; };
    auto sb = std::make_unique<FIDESlib::CKKS::BootstrapPrecomputation::SparseB>();
    sb->n = n; sb->r = r; sb->s = s; sb->bStep = b; sb->gStep = (s + b - 1) / b;
    sb->ctsNF = result.cts0_const != 0 ? std::ldexp(1.0, -result.cts0_t) : 1.0;
    if (envDouble("FIDESLIB_BTS_SPARSE_B_K16", 0.0) > 0) {  // control: the shipped K = 16 EvalMod (pfail not for production)
        sb->cheb = GPUcc.GetCoeffsChebyshev();
        sb->daIts = GPUcc.GetDoubleAngleIts();
        sb->bootK = (double)GPUcc.GetBootK();
    } else {
        sb->cheb = FIDESlib::CKKS::sparseBChebyshevK24();
        sb->daIts = 5;
        sb->bootK = 24.0;
    }
    double qDouble = 1.0;
    for (int j = 0; j < GPUcc.compositeDegree(); ++j)
        qDouble *= (double)GPUcc.prime[j].p;
    // FIDESLIB_BTS_SHIFT: the exact post-raise scaling carries 2^t on the ciphertext and the SHIPPED stage-0 plaintexts
    // carry 2^-t; B's CtS replaces that stage, so it must carry the 2^-t itself
    const double shiftT = result.cts0_const != 0 ? std::ldexp(1.0, -result.cts0_t) : 1.0;
    const double gc = envDouble("FIDESLIB_BTS_SPARSE_B_GC", 1.0) / sb->bootK * shiftT;
    const double gd = envDouble("FIDESLIB_BTS_SPARSE_B_GD", 1.0) * qDouble / GPUcc.sfAtLimb(GPUcc.L);  // scaleDec
    const int half = n / 2;
    auto encode = [&](std::vector<cd>& v) {
        lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, N2);
        FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, p);
        return FIDESlib::CKKS::Plaintext(GPUcc_, raw);
    };
    for (int k = 0; k < s; ++k) {
        const int shift = (k / b) * b;  // BSGS giant step of this diagonal
        std::vector<cd> pk(N2), qk(N2);
        // p_k = (p_{k,a}; p_{k,a}) over the r/2 row blocks a; p_{k,a}[j] = V(a s + (j mod s), (k + j) mod n/2)
        // q_k = (q_{k,a}; 0_{n/2});                           q_{k,a}[j] = U(j, a s + ((j + k) mod s)) * gd
        for (int a = 0; a < r / 2; ++a)
            for (int j = 0; j < half; ++j) {
                const cd v = gc * std::conj(U((k + j) % half, a * s + (j % s)));
                pk[a * n + j] = v;
                pk[a * n + half + j] = v;
                qk[a * n + j] = gd * U(j, a * s + ((j + k) % s));
                qk[a * n + half + j] = cd(0.0, 0.0);
            }
        std::vector<cd> pr(N2), qr(N2);  // rot_{-shift}: out[l] = in[l - shift]
        for (int l = 0; l < N2; ++l) {
            const int src = ((l - shift) % N2 + N2) % N2;
            pr[l] = pk[src];
            qr[l] = qk[src];
        }
        sb->P.push_back(encode(pr));
        sb->Q.push_back(encode(qr));
    }
    cudaDeviceSynchronize();
    std::cerr << "[sparse_b] slots=" << slots << ": n=" << n << " r=" << r << " s=" << s << " bStep=" << b
              << " gStep=" << sb->gStep << " diagonals " << sb->P.size() << "+" << sb->Q.size()
              << " at the top level; K=" << sb->bootK << " degree " << (sb->cheb.size() - 1) << " r=" << sb->daIts
              << " gc=" << gc << " gd=" << gd << "\n";
    result.sparseB = std::move(sb);
}



// FIDESLIB_LT_COMPACT: the index mask under which every diagonal of a CtS/StC stage is exact in its stored (NTT) layout —
// either a period P (mask P-1) or constant aligned blocks of L (mask ~(L-1)); all ones when neither saves at least 2x.
static uint32_t ltStageMask(FIDESlib::CKKS::BootstrapPrecomputation::LTstep& st) {
    size_t Pmax = 1, Lmin = 0, n = 0;
    std::vector<std::vector<std::vector<uint64_t>>> keep;
    for (auto& pt : st.A) {
        std::vector<std::vector<uint64_t>> limbs;
        pt.c0.store(limbs);
        if (limbs.empty()) continue;
        n = limbs[0].size();
        if (Lmin == 0) Lmin = n;
        std::vector<std::vector<uint64_t>> chk;
        for (size_t l : {(size_t)0, (size_t)1, limbs.size() - 1}) {
            if (l >= limbs.size()) continue;
            const auto& v = limbs[l];
            size_t P = 1;
            while (P < n) { bool ok = true; for (size_t i = P; i < n && ok; ++i) ok = v[i] == v[i % P]; if (ok) break; P <<= 1; }
            size_t L = n;
            while (L > 1) { bool ok = true; for (size_t b = 0; b < n && ok; b += L) for (size_t i = b + 1; i < b + L && ok; ++i) ok = v[i] == v[b]; if (ok) break; L >>= 1; }
            Pmax = std::max(Pmax, P);
            Lmin = std::min(Lmin, L);
            chk.push_back(v);
        }
        keep.push_back(std::move(chk));
    }
    if (n == 0) return 0xFFFFFFFFu;
    const size_t dP = Pmax, dB = n / std::max<size_t>(Lmin, 1);
    if (std::min(dP, dB) * 2 > n) return 0xFFFFFFFFu;
    const uint32_t mask = dP <= dB ? (uint32_t)(Pmax - 1) : ~(uint32_t)(Lmin - 1);
    for (auto& chk : keep)  // the chosen mask must reproduce every checked limb exactly
        for (auto& v : chk)
            for (size_t i = 0; i < n; ++i)
                if (v[i] != v[i & mask]) return 0xFFFFFFFFu;
    return mask;
}

// FIDESLIB_BTS_RAISE_DROP="slots:k,..." (composite levels; e.g. "1:4,512:2"): per-route raise below the top modulus.
static int raiseDropFor(int slots) {
    const char* e = std::getenv("FIDESLIB_BTS_RAISE_DROP");
    if (!e || !*e) return 0;
    const std::string v(e);
    size_t i = 0;
    while (i < v.size()) {
        size_t j = v.find(',', i);
        if (j == std::string::npos) j = v.size();
        const std::string tok = v.substr(i, j - i);
        const size_t c = tok.find(':');
        if (c != std::string::npos && std::atoi(tok.substr(0, c).c_str()) == slots)
            return std::atoi(tok.substr(c + 1).c_str());
        i = j + 1;
    }
    return 0;
}
static int stcFirstEnv() {
    const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST");
    return e ? std::atoi(e) : 0;
}
// FIDESLIB_BTS_STC_FIRST_ROUTES: "all" (default), "dense", "sparse", or a comma list of slot counts — which routes take the
// StC-first order (bisection aid: the e2e numerics differ from the single-route harness).
static bool stcFirstRoute(int slots, int N) {
    const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_ROUTES");
    if (!e || !*e || std::string(e) == "all") return true;
    const std::string v(e);
    if (v == "dense") return slots == N / 2;
    if (v == "sparse") return slots != N / 2;
    std::string tok; size_t i = 0;
    while (i <= v.size()) {
        if (i == v.size() || v[i] == ',') { if (!tok.empty() && std::atoi(tok.c_str()) == slots) return true; tok.clear(); }
        else tok += v[i];
        ++i;
    }
    return false;
}

static int btsDeg(FIDESlib::CKKS::ContextData& GPUcc) {  // Bootstrap.cu: deg = round(log2(q0 / 2^p)), q0 = composite bottom
    double qDouble = 1.0;
    for (int j = 0; j < GPUcc.compositeDegree(); ++j)
        qDouble *= (double)GPUcc.prime[j].p;
    return (int)std::lround(std::log2(qDouble / std::pow(2.0, (double)GPUcc.param.raw->p)));
}

static FIDESlib::CKKS::Plaintext relevelTo(FIDESlib::CKKS::Context& GPUcc_, FIDESlib::CKKS::ContextData& GPUcc,
                                           const FIDESlib::CKKS::Plaintext& pt, int targetLevel, const char* what,
                                           double factor = 1.0) {
    const int d = GPUcc.compositeDegree();
    const int cur = pt.c0.getLevel();
    if ((targetLevel - cur) % d != 0)
        throw std::runtime_error(std::string("[stc_first] ") + what + ": plaintext at limb index " + std::to_string(cur) +
                                 " cannot be re-levelled to " + std::to_string(targetLevel) + " (not a multiple of d)");
    return FIDESlib::CKKS::relevelPlaintext(GPUcc_, GPUcc, pt, (targetLevel - cur) / d, factor);
}

static bool btsRealEnv() {
    const char* e = std::getenv("FIDESLIB_BTS_REAL");
    return e && std::atoi(e) > 0;
}

// StC stage 0 for a real payload (BootstrapPrecomputation::stcRealA0): OpenFHE's diagonal `ij` with the entry that
// reads coefficient 0 halved. In OpenFHE's bit-reversed order coefficient 0 sits in StC input slot 0 (the LT-friendly
// rotation below moves it to slot 1), and diagonal ij multiplies the input rotated by j - offset (j = ij mod g,
// EvalSlotsToCoeffs), so that entry is slot k = (offset - j) mod slots. Every other slot is kept exactly: the
// plaintext's integer polynomial A loses round(Re(A(zeta^e) zeta^(-e t)) / N) per coefficient t, e = 5^k mod 2N,
// which is half of slot k. A is read through a 4-limb CRT checked against a 5th limb.
static FIDESlib::CKKS::RawPlainText realStc0Raw(const ReadOnlyPlaintext& pt, int slots, int numRotations, int g, int ij) {
    DCRTPoly a = pt->GetElement<DCRTPoly>();
    const Format fmt = a.GetFormat();
    if (fmt == Format::EVALUATION)
        a.SwitchFormat();
    const auto& tw = a.GetAllElements();
    if (tw.size() < 5)
        throw std::runtime_error("realStc0Raw: the StC plaintext needs >= 5 limbs");
    const uint32_t N = tw[0].GetLength(), M = 2 * N;
    using u128 = unsigned __int128;
    uint64_t q[5];
    for (int l = 0; l < 5; ++l)
        q[l] = tw[l].GetModulus().ConvertToInt();
    auto powmod = [](uint64_t b, uint64_t e, uint64_t m) {
        uint64_t r = 1;
        for (b %= m; e; e >>= 1, b = (u128)b * b % m)
            if (e & 1)
                r = (u128)r * b % m;
        return r;
    };
    u128 P[4] = {1, q[0], (u128)q[0] * q[1], (u128)q[0] * q[1] * q[2]};
    uint64_t inv[4] = {1, 0, 0, 0};
    for (int l = 1; l < 4; ++l)
        inv[l] = powmod((uint64_t)(P[l] % q[l]), q[l] - 2, q[l]);
    const u128 Q = P[3] * q[3];
    std::vector<long double> c(N);
    for (uint32_t t = 0; t < N; ++t) {
        u128 x = tw[0][t].ConvertToInt();
        for (int l = 1; l < 4; ++l) {
            const uint64_t r = tw[l][t].ConvertToInt(), xm = (uint64_t)(x % q[l]);
            x += P[l] * ((u128)((r + q[l] - xm) % q[l]) * inv[l] % q[l]);
        }
        const bool neg = x > Q / 2;
        const u128 mag = neg ? Q - x : x;
        const uint64_t m4 = (uint64_t)(mag % q[4]);
        if ((neg ? (q[4] - m4) % q[4] : m4) != tw[4][t].ConvertToInt())
            throw std::runtime_error("realStc0Raw: StC plaintext coefficient exceeds the 4-limb CRT");
        c[t] = neg ? -(long double)mag : (long double)mag;
    }
    const int32_t offset = (numRotations + 1) / 2 - 1;
    const uint32_t k = (uint32_t)(((offset - ij % g) % slots + slots) % slots);
    uint64_t e = 1;
    for (uint32_t i = 0; i < k; ++i)
        e = e * 5 % M;
    const long double w = 3.14159265358979323846264338327950288L / N;
    long double re = 0, im = 0;
    for (uint32_t t = 0; t < N; ++t) {
        const long double th = w * (long double)((e * t) % M);
        re += c[t] * cosl(th);
        im += c[t] * sinl(th);
    }
    DCRTPoly corr(a.GetParams(), Format::COEFFICIENT, true);
    std::vector<long double> d(N);
    for (uint32_t t = 0; t < N; ++t) {
        const long double th = w * (long double)((e * t) % M);
        d[t] = roundl((re * cosl(th) + im * sinl(th)) / N);
    }
    for (size_t l = 0; l < tw.size(); ++l) {
        const uint64_t ql = tw[l].GetModulus().ConvertToInt();
        NativeVector v(N, tw[l].GetModulus());
        for (uint32_t t = 0; t < N; ++t) {
            long double r = fmodl(d[t], (long double)ql);
            if (r < 0)
                r += ql;
            v[t] = NativeInteger((uint64_t)r);
        }
        NativePoly p(tw[l].GetParams(), Format::COEFFICIENT, true);
        p.SetValues(std::move(v), Format::COEFFICIENT);
        corr.SetElementAtIndex(l, std::move(p));
    }
    a -= corr;
    if (fmt == Format::EVALUATION)
        a.SwitchFormat();
    FIDESlib::CKKS::RawPlainText raw;
    raw.numRes = a.GetAllElements().size();
    raw.N = N;
    raw.sub_0 = FIDESlib::CKKS::GetRawArray(a.GetAllElements());
    raw.moduli = GetModuli(a.GetAllElements());
    raw.format = a.GetFormat();
    if constexpr (FIDESlib::CKKS::REVERSE) {
        for (auto& i : raw.sub_0)
            FIDESlib::bit_reverse_vector(i);
    }
    raw.Noise = pt->GetScalingFactor();
    raw.NoiseLevel = pt->GetNoiseScaleDeg();
    raw.slots = pt->GetSlots();
    return raw;
}

// Sparse routes: (1+i ; 1-i) over the 2*slots view, top level (see BootstrapPrecomputation::stc_first_mask).
static std::unique_ptr<FIDESlib::CKKS::Plaintext> makeStcFirstMask(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                                   FIDESlib::CKKS::Context& GPUcc_, int slots, bool multiStage) {
    std::vector<std::complex<double>> m(2 * slots);
    // Measured with genuine complex data (CKKS_COMPLEX=1, chain105/106): the single-LT route (slots=1) leaves (a-b ; a+b)/2 as
    // derived (StC = diag(U0, i U0) on (z;z), fold, (1+Y^s)Q/2) -> mask (1-i ; 1+i); the multi-stage FFT route leaves the
    // halves the other way, (a+b ; a-b)/2 -> mask (1+i ; 1-i). The wrong orientation returns conj(z) exactly.
    const double sgn = multiStage ? 1.0 : -1.0;
    for (int j = 0; j < slots; ++j) {
        m[j] = std::complex<double>(1.0, sgn);
        m[j + slots] = std::complex<double>(1.0, -sgn);
    }
    lbcrypto::Plaintext p = cc->MakeCKKSPackedPlaintext(m, 1, 0, nullptr, 2 * slots);
    FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cc, p);
    return std::make_unique<FIDESlib::CKKS::Plaintext>(GPUcc_, raw);
}

void FIDESlib::CKKS::AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots,
                                            FIDESlib::CKKS::Context& GPUcc_,
                                            FIDESlib::CKKS::BootstrapPrecomputation& result) {
    ContextData& GPUcc = *GPUcc_;
    auto precom =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

    if (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1 &&
        precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) {

        if (!GPUcc.HasBootPrecomputation(slots)) {
            if constexpr (1) {  // extended limbs computation
                auto auxA = precom->m_U0hatTPre;
                auto auxInvA = precom->m_U0Pre;

                result.LT.A.clear();
                for (int i = 0; i < auxA.size(); ++i) {
                    RawPlainText raw = GetRawPlainText(cc, auxA.at(i));
                    result.LT.A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.LT.A.back().c0.freeSpecialLimbs();
                }

                result.LT.invA.clear();
                for (int i = 0; i < auxInvA.size(); ++i) {
                    RawPlainText raw = GetRawPlainText(cc, auxInvA.at(i));
                    result.LT.invA.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.LT.invA.back().c0.freeSpecialLimbs();
                }
                // FIDESLIB_BTS_SHIFT for the single-LT precomputation (slots whose level budget is {1,1}, e.g. the
                // slots=1 route): same re-level as the multi-stage CtS/StC below — LT.A is the only CtS stage (carries
                // 2^-t), LT.invA the only StC stage — so this route's landing moves with the dense one.
                if (const int bts_shift = [] {
                        const char* e = std::getenv("FIDESLIB_BTS_SHIFT");
                        return e ? std::atoi(e) : 0;
                    }();
                    bts_shift > 0) {
                    const int bts_t = [] {
                        const char* e = std::getenv("FIDESLIB_BTS_SHIFT_T");
                        return e ? std::atoi(e) : 12;
                    }();
                    const double tfac = std::ldexp(1.0, -bts_t);
                    double qDouble = 1.0;
                    for (int j = 0; j < GPUcc.compositeDegree(); ++j)
                        qDouble *= (double)GPUcc.prime[j].p;
                    const double pre = GPUcc.sfAtLimb(GPUcc.L - GPUcc.compositeDegree() * raiseDropFor(slots)) / qDouble;
                    const double c = pre * (1.0 / (GPUcc.GetBootK() * GPUcc.N)) * GPUcc.getBtsPreScale();
                    std::vector<Plaintext> nA, nInv;
                    for (auto& pt : result.LT.A) {
                        nA.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, bts_shift - raiseDropFor(slots), tfac));
                        nA.back().NoiseFactor *= tfac;
                    }
                    // the shipped StC carries OpenFHE's scaleDec = q0 / sf(top); a raise to a lower target leaves the message
                    // at sf(target): the StC compensates sf(top) / sf(target) (= 1 for the shipped raise)
                    const double fdrop = GPUcc.sfAtLimb(GPUcc.L) /
                                         GPUcc.sfAtLimb(GPUcc.L - GPUcc.compositeDegree() * raiseDropFor(slots));
                    for (auto& pt : result.LT.invA)
                        nInv.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, bts_shift - raiseDropFor(slots), fdrop));
                    result.raise_drop = raiseDropFor(slots);
                    result.LT.A = std::move(nA);
                    result.LT.invA = std::move(nInv);
                    result.cts0_const = c;
                    result.cts0_t = bts_t;
                    cudaDeviceSynchronize();
                    std::cerr << "[bts_shift] slots=" << slots << " (single LT): " << result.LT.A.size() << " + "
                              << result.LT.invA.size() << " plaintexts re-levelled by " << bts_shift
                              << " composite level(s); LT at level " << result.LT.A.at(0).c0.getLevel() << ", t = " << bts_t
                              << "\n";
                }
                if (stcFirstEnv() > 0 && stcFirstRoute(slots, GPUcc.N)) {
                    const int d = GPUcc.compositeDegree();
                    result.stc_first_mode = stcFirstEnv() >= 2 ? 2 : 1;
                    const int entry = result.stc_first_mode == 2 ? (2 * d - 1) : (2 * d - 1) + d;  // one StC stage
                    result.LT_first.clear();
                    result.stc_first_deg = btsDeg(GPUcc);
                    if (result.stc_first_mode == 1)  // mode 2 builds its (only = last) stage lazily per call
                        for (auto& pt : result.LT.invA)
                            result.LT_first.push_back(relevelTo(GPUcc_, GPUcc, pt, entry, "LT.invA",
                                                                std::ldexp(1.0, -result.stc_first_deg)));
                    result.stc_first_entry = entry;
                    if (slots != (int)GPUcc.N / 2)
                        result.stc_first_mask = makeStcFirstMask(cc, GPUcc_, slots, /*multiStage=*/false);
                    cudaDeviceSynchronize();
                    std::cerr << "[stc_first] mode " << result.stc_first_mode << " slots=" << slots << " (single LT): " << result.LT_first.size()
                              << " StC plaintexts re-levelled from limb index " << result.LT.invA.at(0).c0.getLevel()
                              << " to " << entry << "; input entry " << entry << "\n";
                }
            }
        }

    } else {

        if (!GPUcc.HasBootPrecomputation(slots)) {
            uint32_t M = cc->GetCyclotomicOrder();

            auto& A = precom->m_U0hatTPreFFT;
            auto& invA = precom->m_U0PreFFT;

            for (int i = 0; i < A.size(); ++i) {
                for (int j = 0; j < A.at(A.size() - 1 - i).size(); ++j) {
                    RawPlainText raw = GetRawPlainText(cc, A.at(A.size() - 1 - i).at(j));
                    result.CtS.at(i).A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.CtS.at(i).A.back().c0.freeSpecialLimbs();
                }
            }

            for (int i = 0; i < invA.size(); ++i) {
                for (int j = 0; j < invA.at(i).size(); ++j) {
                    RawPlainText raw = GetRawPlainText(cc, invA.at(i).at(j));
                    result.StC.at(i).A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.StC.at(i).A.back().c0.freeSpecialLimbs();
                }
            }
            if (btsRealEnv() && slots == (int)GPUcc.N / 2)
                for (int j = 0; j < invA.at(0).size(); ++j) {
                    result.stcRealA0.emplace_back(
                        GPUcc_, realStc0Raw(invA.at(0).at(j), slots, precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS],
                                            precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP], j));
                    if constexpr (remove_extension)
                        result.stcRealA0.back().c0.freeSpecialLimbs();
                }

            // FIDESLIB_BTS_SHIFT = s (composite levels, default 0). FIDESlib applies the FLEXIBLEAUTO adjustment
            // BEFORE raising, so the raised ciphertext is canonical at the top level, while OpenFHE's precompute
            // (written for its own post-raise adjust) encodes the first CtS stage d towers lower: stage 0 drops
            // the top d primes for nothing. Re-level every CtS/StC plaintext s composite levels up on the GPU
            // (exact small-integer lift + the real rescale sf(old)/sf(new), CKKS/SmallInt.cuh); the bootstrap
            // output gains s composite levels with unchanged kernels and keys.
            if (const int bts_shift = [] {
                    const char* e = std::getenv("FIDESLIB_BTS_SHIFT");
                    return e ? std::atoi(e) : 0;
                }();
                bts_shift > 0) {  // dense AND sparse-slot precomputations (the exact scaling commutes with Accumulate)
                // Dense bootstraps only (the sparse-slot ones run Accumulate between the raise and CtS). The
                // post-raise multScalar(constantEvalMult) + rescale is replaced by an EXACT small-integer scaling
                // of the raised (still small-coefficient) ciphertext inside ModRaise (CKKS/SmallInt.cuh), which
                // keeps the level; `cts0_const` records the constant Bootstrap() must see to take that path.
                // (Folding c into the stage-0 plaintexts instead costs ~2.5 bits: their 2^32-sized integers
                // multiply the unscaled q0*I term before c shrinks the signal.)
                double qDouble = 1.0;
                for (int j = 0; j < GPUcc.compositeDegree(); ++j)
                    qDouble *= (double)GPUcc.prime[j].p;
                const double pre = GPUcc.sfAtLimb(GPUcc.L - GPUcc.compositeDegree() * raiseDropFor(slots)) / qDouble;  // composite chain (Bootstrap.cu)
                const double c = pre * (1.0 / (GPUcc.GetBootK() * GPUcc.N)) * GPUcc.getBtsPreScale();
                int n = 0;
                if (const char* e = std::getenv("FIDESLIB_RELEVEL_SELFTEST"); e && std::atoi(e)) {
                    relevelSelfTest(GPUcc_, GPUcc, result.CtS.at(0).A.at(0), "CtS0[0]");
                    relevelSelfTest(GPUcc_, GPUcc, result.CtS.at(1).A.at(3), "CtS1[3]");
                    relevelSelfTest(GPUcc_, GPUcc, result.StC.at(0).A.at(0), "StC0[0]");
                }
                const int bts_t = [] {
                    const char* e = std::getenv("FIDESLIB_BTS_SHIFT_T");
                    return e ? std::atoi(e) : 12;
                }();
                const double tfac = std::ldexp(1.0, -bts_t);
                // lever 1b layout: stage 0 is done by the aggregated switch at level L with full-precision
                // plaintexts; it consumes no level, so stages >= 1 and StC move up by TWO composite levels and
                // stage 1 carries the 2^-t
                const bool aks_layout = [&] {  // only the dense precomputation has an AKS stage 0
                    const char* e = std::getenv("FIDESLIB_AKS");
                    return e && std::atoi(e) > 0 && slots == (int)GPUcc.N / 2;
                }();
                const bool shift_stc = [] {  // diagnostic: FIDESLIB_BTS_SHIFT_STC=0 leaves StC alone
                    const char* e = std::getenv("FIDESLIB_BTS_SHIFT_STC");
                    return !(e && std::atoi(e) == 0);
                }();
                for (auto* v : {&result.CtS, &result.StC})
                    for (size_t si = 0; si < v->size(); ++si) {
                        if (v == &result.StC && !shift_stc)
                            continue;
                        auto& st = (*v)[si];
                        const bool stage0 = (v == &result.CtS && si == 0);
                        std::vector<Plaintext> nv;
                        nv.reserve(st.A.size());
                        const bool carries_t = aks_layout ? (v == &result.CtS && si == 1) : stage0;
                        if (aks_layout && raiseDropFor(slots))
                            throw std::runtime_error("FIDESLIB_BTS_RAISE_DROP is not supported with the AKS layout");
                        const int sh = ((aks_layout && !stage0) ? bts_shift + 1 : bts_shift) - raiseDropFor(slots);
                        // the last StC stage compensates a lower raise target: sf(top) / sf(target) (see the single-LT block)
                        const double fdrop = (v == &result.StC && si + 1 == v->size())
                                                 ? GPUcc.sfAtLimb(GPUcc.L) /
                                                       GPUcc.sfAtLimb(GPUcc.L - GPUcc.compositeDegree() * raiseDropFor(slots))
                                                 : 1.0;
                        for (auto& pt : st.A) {
                            nv.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, sh, (carries_t ? tfac : 1.0) * fdrop));
                            if (carries_t)
                                nv.back().NoiseFactor *= tfac;  // integers x 2^-t at scale sf x 2^-t: same value
                            if (const char* td = std::getenv("BTS_TRACE_DIR"); td && (n == 0 || n == 5 || n == 70 || (v == &result.StC && (&pt == &st.A.front())))) {
                                const std::string base = std::string(td) + "/pt" + std::to_string(n) + (v == &result.StC ? "-stc" : "-cts") + std::to_string(si);
                                exactPlainDump(pt, (base + "-orig.ct").c_str());
                                exactPlainDump(nv.back(), (base + "-new.ct").c_str());
                            }
                            ++n;
                        }
                        if (std::getenv("BTS_TRACE_DIR") && shift_stc) {  // keep the originals for the reference run
                            auto& ov = (v == &result.CtS) ? result.CtS_orig : result.StC_orig;
                            BootstrapPrecomputation::LTstep o;
                            o.slots = st.slots; o.bStep = st.bStep; o.gStep = st.gStep; o.rotIn = st.rotIn; o.rotOut = st.rotOut;
                            o.A = std::move(st.A);
                            ov.push_back(std::move(o));
                        }
                        st.A = std::move(nv);
                        if (v == &result.StC && si == 0 && !result.stcRealA0.empty()) {  // the real-payload stage 0 moves with it
                            std::vector<Plaintext> nr;
                            nr.reserve(result.stcRealA0.size());
                            for (auto& pt : result.stcRealA0)
                                nr.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, sh, fdrop));
                            result.stcRealA0 = std::move(nr);
                        }
                    }
                result.cts0_const = c;
                result.cts0_t = bts_t;
                result.raise_drop = raiseDropFor(slots);
                cudaDeviceSynchronize();
                std::cerr << "[bts_shift] slots=" << slots << " raise_drop=" << result.raise_drop << ": " << n << " CtS/StC plaintexts re-levelled by "
                          << bts_shift << " composite level(s); CtS stage 0 at level "
                          << result.CtS.at(0).A.at(0).c0.getLevel() << "; EvalMod constant " << c
                          << " applied exactly in ModRaise, t = " << bts_t << "\n";
            }

            // Single-prime LT stages (FIDESLIB_STC_SINGLE / FIDESLIB_CTS_SINGLE): stages [lo, lo+n) (n even) rescale by
            // one 27-bit prime instead of a composite level, so the transform lands n primes higher. Stage lo+j sits at
            // limb l0-j with scale q_(l0-j); the last one also carries sf(l0-n)/sf(l0), so NF/sf(level) is unchanged when
            // the ciphertext re-enters the composite grid at l0-n; later stages move up n/2 composite levels.
            auto singlePrime = [&](std::vector<BootstrapPrecomputation::LTstep>& stages, int lo, int n, const char* tag) -> int {
                const int d = GPUcc.compositeDegree();
                if (n <= 0 || n % 2 || lo + n > (int)stages.size())
                    return 0;
                const int top = stages.at(0).A.at(0).c0.getLevel();
                for (size_t si = 1; si < stages.size(); ++si)
                    if (d != 2 || stages.at(si).A.at(0).c0.getLevel() != top - d * (int)si)
                        throw std::runtime_error(std::string(tag) + ": needs composite degree 2 and stages one level apart");
                const int l0 = top - d * lo;
                // a scale ratio carried by a single stage (CtS stage 0: 2^-t, FIDESLIB_BTS_SHIFT_T) would leave it 27 - t
                // bits: it moves to the first composite stage after the window, or is spread over a whole-CtS window
                double moved = 1.0;
                for (int j = 0; j < n; ++j)
                    moved *= stages.at(lo + j).A.at(0).NoiseFactor / GPUcc.sfAtLimb(stages.at(lo + j).A.at(0).c0.getLevel());
                const bool spread = moved != 1.0 && lo + n >= (int)stages.size();
                const double share = spread ? std::pow(moved, 1.0 / n) : 1.0;
                for (size_t si = lo; si < stages.size(); ++si) {
                    auto& st = stages.at(si);
                    const int j = (int)si - lo;
                    std::vector<Plaintext> nv;
                    nv.reserve(st.A.size());
                    for (auto& pt : st.A) {
                        if (j < n) {
                            double sc = GPUcc.modReduceFactorAt(l0 - j);
                            if (j == n - 1)
                                sc *= GPUcc.sfAtLimb(l0 - n) / GPUcc.sfAtLimb(l0);
                            nv.push_back(relevelPlaintextNF(GPUcc_, GPUcc, pt, l0 - j, sc * share));
                        } else if (j == n && moved != 1.0) {
                            nv.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, n / 2, moved));  // integers x moved
                            nv.back().NoiseFactor *= moved;                                     // at scale sf x moved
                        } else {
                            nv.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, n / 2, 1.0));
                        }
                    }
                    st.A = std::move(nv);
                }
                cudaDeviceSynchronize();
                std::cerr << "[" << tag << "] slots=" << slots << ": stages " << lo << ".." << lo + n - 1
                          << " on one prime (limbs " << l0 << ".." << l0 - n + 1 << ")";
                if (moved != 1.0)
                    std::cerr << ", scale ratio " << moved << (spread ? " spread" : " moved to stage " + std::to_string(lo + n));
                std::cerr << "\n";
                return n;
            };
            const bool stcFirstHere = stcFirstEnv() > 0 && stcFirstRoute(slots, GPUcc.N);
            // FIDESLIB_CTS_SINGLE=1: dense route only (CtS acts on I + m/q0: one prime costs ~8 bits unless EvalRound
            // cancels it, which covers the dense route), =2: every route
            if (const char* e = std::getenv("FIDESLIB_CTS_SINGLE");
                e && (std::atoi(e) >= 2 || (std::atoi(e) == 1 && slots == (int)GPUcc.N / 2)) && !stcFirstHere) {
                if (const char* a = std::getenv("FIDESLIB_AKS"); a && std::atoi(a) > 0)
                    throw std::runtime_error("FIDESLIB_CTS_SINGLE is not supported with the AKS layout");
                // FIDESLIB_CTS_SINGLE_LO / _N: the window (default stages 1..2; LO=0 N=4 is the whole CtS)
                const char* el = std::getenv("FIDESLIB_CTS_SINGLE_LO");
                const char* en = std::getenv("FIDESLIB_CTS_SINGLE_N");
                const int lo = el && *el ? std::atoi(el) : 1, n = en && *en ? std::atoi(en) : 2;
                result.cts_single = singlePrime(result.CtS, lo, n, "cts_single");
                result.cts_single_lo = lo;
                if (result.cts_single) {  // EvalMod arrives n/2 levels higher: StC follows
                    for (auto& st : result.StC) {
                        std::vector<Plaintext> nv;
                        nv.reserve(st.A.size());
                        for (auto& pt : st.A)
                            nv.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, result.cts_single / 2, 1.0));
                        st.A = std::move(nv);
                    }
                    std::vector<Plaintext> nr;
                    nr.reserve(result.stcRealA0.size());
                    for (auto& pt : result.stcRealA0)
                        nr.push_back(relevelPlaintext(GPUcc_, GPUcc, pt, result.cts_single / 2, 1.0));
                    result.stcRealA0 = std::move(nr);
                }
            }
            // under EvalRound StC carries q0*I and needs its composite scale: the dense route keeps it
            const bool erDense = [&] {
                const char* er = std::getenv("FIDESLIB_EVALROUND");
                return er && std::atoi(er) > 0 && slots == (int)GPUcc.N / 2;
            }();
            if (const char* e = std::getenv("FIDESLIB_STC_SINGLE"); e && std::atoi(e) > 0 && !stcFirstHere && !erDense)
            {
                result.stc_single = singlePrime(result.StC, 0, 2, "stc_single");
                if (result.stc_single && !result.stcRealA0.empty()) {  // the real-payload stage 0 is a single stage 0
                    std::vector<Plaintext> nr;
                    nr.reserve(result.stcRealA0.size());
                    for (auto& pt : result.stcRealA0)
                        nr.push_back(relevelPlaintextNF(GPUcc_, GPUcc, pt, pt.c0.getLevel(),
                                                        GPUcc.modReduceFactorAt(pt.c0.getLevel())));
                    result.stcRealA0 = std::move(nr);
                }
            }

            buildSparseB(cc, GPUcc_, GPUcc, slots, result);

            auto ltFriendly = [&](std::vector<BootstrapPrecomputation::LTstep>& CtS,
                                  std::vector<BootstrapPrecomputation::LTstep>& StC,
                                  std::vector<Plaintext>* stcRealA0 = nullptr) {
            int acc_offset = 0;
            if constexpr (MAKE_CTS_LT_FRIENDLY) {
                for (int32_t s = 0; s < CtS.size(); s++) {
                    int offset = CtS.at(s).rotOut[0];
                    acc_offset += CtS.at(s).rotOut[0];
                    CtS.at(s).rotOut[0] = 0;

                    for (int i = 0; i < CtS.at(s).gStep; ++i) {
                        for (int j = 0; j < CtS.at(s).bStep; ++j) {
                            if (i * CtS.at(s).bStep + j < CtS.at(s).slots) {
                                CtS.at(s).A[i * CtS.at(s).bStep + j].automorph(
                                    ReduceRotation(-acc_offset, M / 4));
                            }
                        }
                    }
                }
            }

            if constexpr (MAKE_STC_LT_FRIENDLY) {
                for (int32_t s = 0; s < StC.size(); s++) {
                    int offset = StC.at(s).rotOut[0];
                    acc_offset += StC.at(s).rotOut[0];

                    StC.at(s).rotOut[0] = 0;

                    for (int i = 0; i < StC.at(s).gStep; ++i) {
                        for (int j = 0; j < StC.at(s).bStep; ++j) {
                            if (i * StC.at(s).bStep + j < StC.at(s).slots) {
                                StC.at(s).A[i * StC.at(s).bStep + j].automorph(
                                    ReduceRotation(-acc_offset, M / 4));
                                if (s == 0 && stcRealA0 && !stcRealA0->empty())
                                    (*stcRealA0)[i * StC.at(s).bStep + j].automorph(ReduceRotation(-acc_offset, M / 4));
                            }
                        }
                    }

                    if (s == StC.size() - 1) {
                        StC.at(s).rotOut[0] = acc_offset;
                        StC.at(s).rotOut[0] %= std::min(2 * slots, (int)cc->GetRingDimension() / 2);
                        for (int i = 1; i < StC.at(s).gStep; ++i) {
                            StC.at(s).rotOut[i] += acc_offset;
                            StC.at(s).rotOut[i] %= std::min(2 * slots, (int)cc->GetRingDimension() / 2);
                        }
                    }
                }
            }
            };
            ltFriendly(result.CtS, result.StC, &result.stcRealA0);
            // default ON (bit-exact); FIDESLIB_LT_COMPACT=0 keeps the full streaming reads
            if (const char* e = std::getenv("FIDESLIB_LT_COMPACT"); !(e && *e) || std::atoi(e) > 0) {
                for (auto* v : {&result.CtS, &result.StC})
                    for (size_t si = 0; si < v->size(); ++si) {
                        auto& st = (*v)[si];
                        st.ptMask = ltStageMask(st);
                        std::cerr << "[lt_compact] slots=" << slots << (v == &result.CtS ? " CtS" : " StC") << " stage "
                                  << si << ": mask 0x" << std::hex << st.ptMask << std::dec << "\n";
                    }
            }
            if (!result.CtS_orig.empty())
                ltFriendly(result.CtS_orig, result.StC_orig);
            if (stcFirstEnv() > 0 && stcFirstRoute(slots, GPUcc.N)) {
                const int d = GPUcc.compositeDegree();
                const int nS = (int)result.StC.size();
                result.stc_first_mode = stcFirstEnv() >= 2 ? 2 : 1;
                const int entry = (2 * d - 1) + (result.stc_first_mode == 2 ? nS - 1 : nS) * d;
                result.StC_first.clear();
                result.stc_first_deg = btsDeg(GPUcc);
                int n = 0;
                for (int si = 0; si < nS - (result.stc_first_mode == 2 ? 1 : 0); ++si) {  // mode 2: last stage per call
                    auto& st = result.StC.at(si);
                    BootstrapPrecomputation::LTstep o;
                    o.slots = st.slots; o.bStep = st.bStep; o.gStep = st.gStep; o.rotIn = st.rotIn; o.rotOut = st.rotOut;
                    o.A.reserve(st.A.size());
                    const double fac = si == nS - 1 ? std::ldexp(1.0, -result.stc_first_deg) : 1.0;
                    for (auto& pt : st.A) {
                        o.A.push_back(relevelTo(GPUcc_, GPUcc, pt, entry - si * d, "StC", fac));
                        ++n;
                    }
                    result.StC_first.push_back(std::move(o));
                }
                result.stc_first_entry = entry;
                if (slots != (int)GPUcc.N / 2)
                    result.stc_first_mask = makeStcFirstMask(cc, GPUcc_, slots, /*multiStage=*/true);
                cudaDeviceSynchronize();
                std::cerr << "[stc_first] mode " << result.stc_first_mode << " slots=" << slots << ": " << n
                          << " StC plaintexts in " << nS << " stages re-levelled from limb index " << result.StC.at(0).A.at(0).c0.getLevel() << " to "
                          << entry << "; input entry " << entry << ", deg " << result.stc_first_deg
                          << (result.stc_first_mask ? " (sparse: mask built)" : "") << "\n";
            }
        }
    }
}

// Per-site level-aware ModRaise (plan `raise_drop`): the route's precomputation re-levelled for a raise that stops
// `drop` composite levels below the top. Every CtS/StC/LT plaintext moves down by drop - base.raise_drop (exact
// small-integer re-level, one rounding), the last StC stage additionally carries sf(base top) / sf(variant top)
// (the shipped StC compensates q0 / sf(raise top)), the stage-0 2^-t scale tag is preserved, the exact post-raise
// constant is recomputed for the lower top, keys are shared with the base.
void FIDESlib::CKKS::AddBootstrapRaiseVariant(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> /*cc*/, int slots,
                                              FIDESlib::CKKS::Context& GPUcc_, int drop) {
    ContextData& GPUcc = *GPUcc_;
    BootstrapPrecomputation& base = GPUcc.GetBootPrecomputationBase(slots);
    if (drop <= base.raise_drop || base.raise_variants.count(drop))
        return;
    if (base.aks0 || base.sparseB || base.stc_first_mode > 0)
        throw std::runtime_error("AddBootstrapRaiseVariant: not supported with FIDESLIB_AKS / BTS_SPARSE_B / BTS_STC_FIRST");
    const int d = GPUcc.compositeDegree();
    const int shift = -(drop - base.raise_drop);
    const double sfBase = GPUcc.sfAtLimb(GPUcc.L - d * base.raise_drop);
    const double sfVar = GPUcc.sfAtLimb(GPUcc.L - d * drop);
    const double fdrop = sfBase / sfVar;
    auto relevel = [&](const Plaintext& pt, double factor) {
        // stage-0 plaintexts carry 2^-t in their scale bookkeeping (NoiseFactor = sf x 2^-t): keep that ratio
        const double tag = pt.NoiseFactor / GPUcc.sfAtLimb(pt.c0.getLevel());
        Plaintext np = relevelPlaintext(GPUcc_, GPUcc, pt, shift, factor);
        np.NoiseFactor *= tag;
        return np;
    };
    auto var = std::make_unique<BootstrapPrecomputation>();
    var->LT.slots = base.LT.slots;
    var->LT.bStep = base.LT.bStep;
    for (auto& pt : base.LT.A)
        var->LT.A.push_back(relevel(pt, 1.0));
    for (auto& pt : base.LT.invA)  // single-LT route: invA is its only StC stage
        var->LT.invA.push_back(relevel(pt, fdrop));
    auto copySteps = [&](const std::vector<BootstrapPrecomputation::LTstep>& src,
                         std::vector<BootstrapPrecomputation::LTstep>& dst, bool isStC) {
        for (size_t si = 0; si < src.size(); ++si) {
            BootstrapPrecomputation::LTstep o;
            o.slots = src[si].slots;
            o.bStep = src[si].bStep;
            o.gStep = src[si].gStep;
            o.rotIn = src[si].rotIn;
            o.rotOut = src[si].rotOut;
            o.ptMask = src[si].ptMask;
            const double f = (isStC && si + 1 == src.size()) ? fdrop : 1.0;
            // single-prime stages: to the variant's limb at its own prime, the grid re-entry factor recomputed
            const int sLo = isStC ? 0 : base.cts_single_lo;
            const int sN = isStC ? base.stc_single : base.cts_single;
            const int j = (int)si - sLo;
            const bool single = sN > 0 && j >= 0 && j < sN;
            o.A.reserve(src[si].A.size());
            for (auto& pt : src[si].A) {
                if (!single) {
                    o.A.push_back(relevel(pt, f));
                    continue;
                }
                const int l = pt.c0.getLevel(), lv = l + d * shift;
                const int l0 = l + j, l0v = lv + j;
                double nf = pt.NoiseFactor / GPUcc.modReduceFactorAt(l) * GPUcc.modReduceFactorAt(lv);
                if (j == sN - 1)
                    nf *= (GPUcc.sfAtLimb(l0v - sN) / GPUcc.sfAtLimb(l0v)) / (GPUcc.sfAtLimb(l0 - sN) / GPUcc.sfAtLimb(l0));
                o.A.push_back(relevelPlaintextNF(GPUcc_, GPUcc, pt, lv, nf * f));
            }
            dst.push_back(std::move(o));
        }
    };
    copySteps(base.CtS, var->CtS, false);
    copySteps(base.StC, var->StC, true);
    for (auto& pt : base.stcRealA0) {
        if (base.stc_single == 0) {
            var->stcRealA0.push_back(relevel(pt, base.StC.size() == 1 ? fdrop : 1.0));
            continue;
        }
        // single-prime StC stage 0: the variant's limb at its own prime
        const int l = pt.c0.getLevel(), lv = l + d * shift;
        const double nf = pt.NoiseFactor / GPUcc.modReduceFactorAt(l) * GPUcc.modReduceFactorAt(lv);
        var->stcRealA0.push_back(relevelPlaintextNF(GPUcc_, GPUcc, pt, lv, nf));
    }
    var->accumulate_bStep = base.accumulate_bStep;
    var->correctionFactor = base.correctionFactor;
    var->sparse_encaps = base.sparse_encaps;
    var->sparse_context = base.sparse_context;
    var->sparse_atob = base.sparse_atob;
    var->sparse_btoa = base.sparse_btoa;
    var->ghs_btoa = base.ghs_btoa;
    var->cts0_t = base.cts0_t;
    var->raise_drop = drop;
    var->stc_single = base.stc_single;
    var->cts_single = base.cts_single;
    var->cts_single_lo = base.cts_single_lo;
    if (base.cts0_const != 0) {
        double qDouble = 1.0;
        for (int j = 0; j < d; ++j)
            qDouble *= (double)GPUcc.prime[j].p;
        const double pre = sfVar / qDouble;
        var->cts0_const = pre * (1.0 / (GPUcc.GetBootK() * GPUcc.N)) * GPUcc.getBtsPreScale();
    }
    cudaDeviceSynchronize();
    std::cerr << "[bts_raise] slots=" << slots << " variant drop=" << drop << " (base " << base.raise_drop
              << "): plaintexts re-levelled by " << shift << " composite level(s), raise top " << (GPUcc.L - d * drop)
              << " of " << GPUcc.L << "\n";
    base.raise_variants[drop] = std::move(var);
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots,
                                                FIDESlib::CKKS::Context& GPUcc_) {
    ContextData& GPUcc = *GPUcc_;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();

    if (!GPUcc.HasBootPrecomputation(slots)) {
        FIDESlib::CKKS::BootstrapPrecomputation result_;

        FIDESlib::CKKS::BootstrapPrecomputation& result =
            GPUcc.HasBootPrecomputation(slots) ? GPUcc_->GetBootPrecomputation(slots) : result_;

        std::vector<int> indexes = GetBootstrapIndexes(cc, slots, &result);

        AddBootstrapPlaintexts(cc, slots, GPUcc_, result);

        result.correctionFactor =
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_correctionFactor;

        if (GPUcc.param.raw->sparse_encaps) {
            result.sparse_encaps = true;
        }

        GPUcc.AddBootPrecomputation(slots, std::move(result));
    }

    AddBootstrapKeys(publicKey, slots, GPUcc_);
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                const KeyPair<lbcrypto::DCRTPoly>& keys, int slots,
                                                FIDESlib::CKKS::Context& GPUcc_) {

    GenBootstrapKeys(keys, slots);
    AddBootstrapPrecomputation(keys.publicKey, slots, GPUcc_);
}
