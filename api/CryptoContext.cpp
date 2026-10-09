#include <cstdlib>
#include <sstream>
#include <random>
#include <unordered_map>
#include "CryptoContext.hpp"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/AksKeys.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/DeviceEncode.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CudaUtils.cuh"
#include "Definitions.hpp"
#include "PolyApprox.cuh"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "ciphertext-fwd.h"
#include "cryptocontext-fwd.h"
#include "lattice/hal/lat-backend.h"

#include <any>
#include <atomic>
#include <optional>
#include <cmath>
#include <fstream>
#include <complex>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <functional>
#include <openfhe.h>

// Serialization headers - required for cereal type registration.
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <memory>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

// NATIVEINT=32 uses ubint<unsigned int>; NATIVEINT=64 uses ubint<unsigned long>.
#if NATIVEINT == 32
#define FIDES_DCRTPOLY_FULL lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned int>>>
#else
#define FIDES_DCRTPOLY_FULL lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>
#endif

template <> std::map<std::string, std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
template <>
std::map<std::string, std::shared_ptr<std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalAutomorphismKeyMap;

namespace fideslib {

static std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
	{ .p = 2251799661248513 },
	{ .p = 2251799661641729 },
	{ .p = 2251799665180673 },
	{ .p = 2251799682088961 },
	{ .p = 2251799678943233 },
	{ .p = 2251799717609473 },
	{ .p = 2251799710138369 },
	{ .p = 2251799708827649 },
	{ .p = 2251799707385857 },
	{ .p = 2251799713677313 },
	{ .p = 2251799712366593 },
	{ .p = 2251799716691969 },
	{ .p = 2251799714856961 },
	{ .p = 2251799726522369 },
	{ .p = 2251799726129153 },
	{ .p = 2251799747493889 },
	{ .p = 2251799741857793 },
	{ .p = 2251799740416001 },
	{ .p = 2251799746707457 },
	{ .p = 2251799756013569 },
	{ .p = 2251799775805441 },
	{ .p = 2251799763091457 },
	{ .p = 2251799767154689 },
	{ .p = 2251799765975041 },
	{ .p = 2251799770562561 },
	{ .p = 2251799769776129 },
	{ .p = 2251799772266497 },
	{ .p = 2251799775281153 },
	{ .p = 2251799774887937 },
	{ .p = 2251799797432321 },
	{ .p = 2251799787995137 },
	{ .p = 2251799787601921 },
	{ .p = 2251799791403009 },
	{ .p = 2251799789568001 },
	{ .p = 2251799795466241 },
	{ .p = 2251799807131649 },
	{ .p = 2251799806345217 },
	{ .p = 2251799805165569 },
	{ .p = 2251799813554177 },
	{ .p = 2251799809884161 },
	{ .p = 2251799810670593 },
	{ .p = 2251799818928129 },
	{ .p = 2251799816568833 },
	{ .p = 2251799815520257 } };

static std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
	{ .p = 2305843009220116481 },
	{ .p = 2305843009221820417 },
	{ .p = 2305843009224179713 },
	{ .p = 2305843009225228289 },
	{ .p = 2305843009227980801 },
	{ .p = 2305843009229160449 },
	{ .p = 2305843009229946881 },
	{ .p = 2305843009231650817 },
	{ .p = 2305843009235189761 },
	{ .p = 2305843009240301569 },
	{ .p = 2305843009242923009 },
	{ .p = 2305843009244889089 },
	{ .p = 2305843009245413377 },
	{ .p = 2305843009247641601 } };

static std::unordered_map<PKESchemeFeature, lbcrypto::PKESchemeFeature> PKESchemeFeatureMap = {
	{ PKESchemeFeature::PKE, lbcrypto::PKE },
	{ PKESchemeFeature::KEYSWITCH, lbcrypto::KEYSWITCH },
	{ PKESchemeFeature::PRE, lbcrypto::PRE },
	{ PKESchemeFeature::LEVELEDSHE, lbcrypto::LEVELEDSHE },
	{ PKESchemeFeature::ADVANCEDSHE, lbcrypto::ADVANCEDSHE },
	{ PKESchemeFeature::MULTIPARTY, lbcrypto::MULTIPARTY },
	{ PKESchemeFeature::FHE, lbcrypto::FHE },
	{ PKESchemeFeature::SCHEMESWITCH, lbcrypto::SCHEMESWITCH },
};

CryptoContextImpl<DCRTPoly>::~CryptoContextImpl() {
	if (plaintext_ready_events_mutex) {
		plaintext_ready_events_mutex->lock();
		for (auto& kv : plaintext_ready_events) {
			if (kv.second != nullptr) {
				cudaEventDestroy(kv.second);
			}
		}
		plaintext_ready_events.clear();
		plaintext_ready_events_mutex->unlock();
	}
	lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::ClearEvalMultKeys();
	lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::ClearEvalAutomorphismKeys();
}

// ---- Enable features ----

void CryptoContextImpl<DCRTPoly>::Enable(PKESchemeFeature feature) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	context->Enable(PKESchemeFeatureMap[feature]);
}

void CryptoContextImpl<DCRTPoly>::Enable(uint32_t featureMask) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	context->Enable(featureMask);
}

// ---- Getters ----

uint32_t CryptoContextImpl<DCRTPoly>::GetCyclotomicOrder() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	return context->GetCyclotomicOrder();
}

uint32_t CryptoContextImpl<DCRTPoly>::GetRingDimension() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	return context->GetRingDimension();
}

double CryptoContextImpl<DCRTPoly>::GetPreScaleFactor(uint32_t slots) {
	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	return FIDESlib::CKKS::GetPreScaleFactor(context_gpu, static_cast<int32_t>(slots));
}

// ---- Setters ----

void CryptoContextImpl<DCRTPoly>::SetAutoLoadPlaintexts(bool autoload) {
	this->auto_load_plaintexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetAutoLoadCiphertexts(bool autoload) {
	this->auto_load_ciphertexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetDevices(const std::vector<int>& devices) {
	if (this->loaded) {
		OPENFHE_THROW("SetDevices must be called before LoadContext");
	}

	this->devices = devices;
}

void CryptoContextImpl<DCRTPoly>::SetPlaintextStreams(cudaStream_t load_stream, cudaStream_t compute_stream) {
	this->plaintext_load_stream = load_stream;
	this->plaintext_compute_stream = compute_stream;
	this->plaintext_streams_enabled = (load_stream != nullptr || compute_stream != nullptr);
}

void CryptoContextImpl<DCRTPoly>::ClearPlaintextStreams() {
	this->plaintext_load_stream = nullptr;
	this->plaintext_compute_stream = nullptr;
	this->plaintext_streams_enabled = false;
}

// ---- Load to devices ----

std::vector<uint32_t> CryptoContextImpl<DCRTPoly>::CoefficientOrderProbe() {
	// Two structured plaintexts: coefficients all 1 (gives the per-limb constant factor FIDESlib's INTT leaves) and
	// coefficient k = k + 1 (gives the index map). Only limb 0 is used (N + 1 < q_0).
	auto& c		 = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& cpu_cc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const uint32_t N = cpu_cc->GetRingDimension();
	auto stored = [&](bool ramp) {
		std::vector<double> vals(N / 2, 1.0);
		auto pt	   = cpu_cc->MakeCKKSPackedPlaintext(vals, 1, 0);
		auto& el   = pt->GetElement<lbcrypto::DCRTPoly>();
		el.SetFormat(Format::COEFFICIENT);
		for (size_t i = 0; i < el.GetNumOfElements(); ++i) {
			auto v		 = el.GetElementAtIndex(i);
			const auto q = v.GetModulus();
			for (uint32_t k = 0; k < N; ++k)
				v[k] = typename lbcrypto::DCRTPoly::PolyType::Integer((ramp ? (uint64_t)k + 1 : 1ULL)) % q;
			el.SetElementAtIndex(i, std::move(v));
		}
		el.SetFormat(Format::EVALUATION);
		FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(cpu_cc, pt);
		FIDESlib::CKKS::Plaintext gp(c, raw);
		cudaDeviceSynchronize();
		FIDESlib::CKKS::RNSPoly t(*c, gp.c0.getLevel());
		t.copy(gp.c0);
		t.INTT(c->batch, true);
		cudaDeviceSynchronize();
		std::vector<std::vector<uint64_t>> st;
		t.store(st);
		cudaDeviceSynchronize();
		return std::make_pair(st[0], (uint64_t)el.GetElementAtIndex(0).GetModulus().ConvertToInt());
	};
	auto [ones, q0]	 = stored(false);
	auto [ramp, q0b] = stored(true);
	uint64_t f = ones[0];
	size_t nonconst = 0;
	for (uint32_t i = 0; i < N; ++i) nonconst += (ones[i] != f);
	auto mulmod = [&](uint64_t x, uint64_t y) { return (uint64_t)((__uint128_t)x * y % q0); };
	auto powmod = [&](uint64_t b, uint64_t e) { uint64_t r = 1; while (e) { if (e & 1) r = mulmod(r, b); b = mulmod(b, b); e >>= 1; } return r; };
	const uint64_t finv = powmod(f % q0, q0 - 2);
	std::vector<uint32_t> pi(N, UINT32_MAX);
	size_t bad = 0;
	for (uint32_t i = 0; i < N; ++i) {
		const uint64_t k1 = mulmod(ramp[i], finv);  // = k + 1
		if (k1 >= 1 && k1 <= N) pi[i] = (uint32_t)(k1 - 1); else ++bad;
	}
	std::cerr << "[coef-order] N=" << N << " q0=" << q0 << " factor=" << f << " non-constant=" << nonconst
			  << " unmatched=" << bad << " pi[0..7]=";
	for (int i = 0; i < 8; ++i) std::cerr << pi[i] << " ";
	std::cerr << "\n";
	return pi;
}

void CryptoContextImpl<DCRTPoly>::LoadDiagSecret(const PrivateKey<DCRTPoly>& secretKey) {
	if (!this->loaded)
		throw std::runtime_error("LoadDiagSecret: LoadContext first");
	auto& c		 = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);
	const auto& el = skImpl->GetPrivateElement();
	std::vector<std::vector<uint64_t>> limbs;
	std::vector<uint64_t> moduli;
	for (size_t i = 0; i < el.GetNumOfElements(); ++i) {
		const auto& v = el.GetElementAtIndex(i);
		moduli.push_back(v.GetModulus().ConvertToInt());
		limbs.emplace_back(v.GetLength());
		for (size_t n = 0; n < v.GetLength(); ++n)
			limbs.back()[n] = v[n].ConvertToInt();
	}
	FIDESlib::CKKS::loadDiagSecret(*c, limbs, moduli);
}

static void extractEvalLimbs(const lbcrypto::DCRTPoly& el, std::vector<std::vector<uint64_t>>& limbs,
							 std::vector<uint64_t>& moduli) {
	limbs.clear();
	moduli.clear();
	for (size_t i = 0; i < el.GetNumOfElements(); ++i) {
		const auto& v = el.GetElementAtIndex(i);
		moduli.push_back(v.GetModulus().ConvertToInt());
		limbs.emplace_back(v.GetLength());
		for (size_t n = 0; n < v.GetLength(); ++n)
			limbs.back()[n] = v[n].ConvertToInt();
	}
}

void CryptoContextImpl<DCRTPoly>::RegenerateEncapsulationKeys(const PrivateKey<DCRTPoly>& secretKey) {
	if (this->loaded)
		throw std::runtime_error("RegenerateEncapsulationKeys: call before LoadContext");
	auto& cpu_cc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);
	if (this->keyDist != fideslib::SPARSE_ENCAPSULATED)
		throw std::runtime_error("RegenerateEncapsulationKeys: not a SPARSE_ENCAPSULATED context");
	// same construction as the fork's EvalBootstrapKeyGen (ckksrns-fhe.cpp), composite chains: hybrid keys both ways
	lbcrypto::DCRTPoly::TugType tug;
	lbcrypto::DCRTPoly sNew(tug, skImpl->GetCryptoParameters()->GetElementParams(), Format::EVALUATION, 32);
	extractEvalLimbs(sNew, aks_sparse_limbs_, aks_sparse_moduli_);
	auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cpu_cc);
	skNew->SetPrivateElement(std::move(sNew));
	auto algo		= cpu_cc->GetScheme();
	const auto M	= cpu_cc->GetCyclotomicOrder();
	auto& evalKeys	= lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMap(skImpl->GetKeyTag());
	// FIDESlib's own GPU-path convention (RawCiphertext.cu GenBootstrapKeys / createContextSwitchingKeys):
	// M-2 = dense->sparse (KeySwitchGen(s, s~)), M-4 = sparse->dense (KeySwitchGen(s~, s)) — the fork's CPU keygen
	// uses the opposite slots and is NOT what the GPU path loads.
	evalKeys[M - 2] = algo->KeySwitchGen(skImpl, skNew);
	evalKeys[M - 4] = algo->KeySwitchGen(skNew, skImpl);
	std::cerr << "[aks] encapsulation keys regenerated from a held sparse secret (h=32)\n";
}

void CryptoContextImpl<DCRTPoly>::LoadAksKeys(const PrivateKey<DCRTPoly>& secretKey) {
	if (!this->loaded)
		throw std::runtime_error("LoadAksKeys: LoadContext first");
	auto& c		 = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& cpu_cc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);
	const auto& el = skImpl->GetPrivateElement();  // EVAL form over Q
	std::vector<std::vector<uint64_t>> limbs;
	std::vector<uint64_t> moduli;
	for (size_t i = 0; i < el.GetNumOfElements(); ++i) {
		const auto& v = el.GetElementAtIndex(i);
		moduli.push_back(v.GetModulus().ConvertToInt());
		limbs.emplace_back(v.GetLength());
		for (size_t n = 0; n < v.GetLength(); ++n)
			limbs.back()[n] = v[n].ConvertToInt();
	}
	auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cpu_cc->GetScheme()->m_FHE);
	const uint64_t seed = [] {
		const char* e = std::getenv("FIDESLIB_AKS_SEED");
		return e ? (uint64_t)std::strtoull(e, nullptr, 10) : 0x5eedull;
	}();
	for (const auto& [slots, _] : fhe->m_bootPrecomMap) {
		if (!c->HasBootPrecomputation((int)slots))
			continue;
		auto& pre = c->GetBootPrecomputation((int)slots);
		if (pre.CtS.empty() || pre.cts0_const <= 0 || (int)slots != c->N / 2)  // AKS: the dense, shifted precomputation only
			continue;
		if (aks_sparse_limbs_.empty())
			throw std::runtime_error("LoadAksKeys: RegenerateEncapsulationKeys was not called before LoadContext");
		FIDESlib::CKKS::GenerateAksStage0(c, pre, limbs, moduli, aks_sparse_limbs_, seed ^ slots);
	}
}

void CryptoContextImpl<DCRTPoly>::LoadContext(const PublicKey<DCRTPoly>& publicKey) {
	if (this->loaded || this->devices.empty())
		return;

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	FIDESlib::CKKS::Parameters params{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

	// Determine the boot configuration based on the secret key distribution.
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	FIDESlib::BOOT_CONFIG bootConfig;
	switch (this->keyDist) {
	case fideslib::UNIFORM_TERNARY: bootConfig = FIDESlib::UNIFORM; break;
	case fideslib::SPARSE_TERNARY: bootConfig = FIDESlib::SPARSE; break;
	case fideslib::SPARSE_ENCAPSULATED: bootConfig = FIDESlib::ENCAPS; break;
	default: bootConfig = FIDESlib::UNIFORM; break;
	}

	FIDESlib::CKKS::RawParams rawParams = FIDESlib::CKKS::GetRawParams(context, bootConfig);
	params								= params.adaptTo(rawParams);
	FIDESlib::CKKS::Context c			= FIDESlib::CKKS::GenCryptoContextGPU(params, this->devices);

	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);

	// Multiplicative key switching key.
	auto& keyMap = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(pkImpl->GetKeyTag()) != keyMap.end()) {
		auto raw_eval_ksk = FIDESlib::CKKS::GetEvalKeySwitchKey(pkImpl);
		FIDESlib::CKKS::KeySwitchingKey eval_ksk(c);
		eval_ksk.Initialize(raw_eval_ksk);
		c->AddEvalKey(std::move(eval_ksk));
	}
	// Rotational key switching keys. Steps in deferred_rotation_indexes are skipped
	// here (their OpenFHE eval keys still exist) and GPU-loaded later via
	// LoadRotationKeys() — used to keep decode keys off the device during prefill.
	std::set<int> deferred(this->deferred_rotation_indexes.begin(), this->deferred_rotation_indexes.end());
	// FIDESLIB_ROT_KEY_BAND=<chain position>: band MODEL rotation keys to the
	// data segment (rotation-key limb pruning). Bootstrap/eval/conjugation keys
	// load elsewhere and stay full.
	const int rot_band = [] {
		const char* e = std::getenv("FIDESLIB_ROT_KEY_BAND");
		return (e && *e) ? std::atoi(e) : -1;
	}();
	// Bootstrap-internal rotations (Accumulate/CtS/StC) run at full level and
	// AddRotationKeys skips already-present indexes, so any index the bootstrap
	// needs must stay FULL here.
	std::set<int> full_keep;
	// rotation indexes are stored normalized to [0, N/2); negative model steps
	// (e.g. -k) land on the SAME slots as bootstrap indexes N/2-k, so the
	// exclusion lookup must compare in normalized space
	const int half_ring = static_cast<int>(context->GetRingDimension() / 2);
	auto norm_idx = [half_ring](int v) {
		v %= half_ring;
		if (v < 0)
			v += half_ring;
		return v;
	};
	if (rot_band >= 0) {
		auto fhe_pre = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE);
		if (fhe_pre) {
			for (const auto& [slots_pre, _] : fhe_pre->m_bootPrecomMap) {
				auto idx = FIDESlib::CKKS::GetBootstrapIndexes(context, static_cast<int>(slots_pre), nullptr);
				for (int v : idx)
					full_keep.insert(norm_idx(v));
			}
		}
		std::cerr << "[rot_band] band=" << rot_band << " full_keep=" << full_keep.size()
				  << " (precom entries=" << (fhe_pre ? fhe_pre->m_bootPrecomMap.size() : 0) << ")\n";
	}
	int n_banded = 0, n_full = 0;
	for (const auto& step : this->rotation_indexes) {
		if (deferred.count(step)) continue;
		auto raw_rot_ksk = FIDESlib::CKKS::GetRotationKeySwitchKey(pkImpl, step);
		FIDESlib::CKKS::KeySwitchingKey rot_ksk(c);
		const bool full = rot_band < 0 || full_keep.count(norm_idx(step)) > 0;
		(full ? n_full : n_banded)++;
		rot_ksk.Initialize(raw_rot_ksk, full ? -1 : rot_band);
		c->AddRotationKey(step, std::move(rot_ksk));
	}
	if (rot_band >= 0)
		std::cerr << "[rot_band] model keys: banded=" << n_banded << " full=" << n_full << "\n";

	// Bootstrapping precomputations.
	auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE);
	if (fhe) {
		auto precom = fhe->m_bootPrecomMap;
		if (!precom.empty()) {
			for (const auto& [slots, _] : precom) {
				FIDESlib::CKKS::AddBootstrapPrecomputation(pkImpl, static_cast<int32_t>(slots), c);
			}
			// per-site level-aware ModRaise variants (SetBootstrapRaiseDrops / FIDESLIB_BTS_RAISE_DROPS)
			std::set<int> drops(this->bts_raise_drops.begin(), this->bts_raise_drops.end());
			if (const char* e = std::getenv("FIDESLIB_BTS_RAISE_DROPS"); e && *e) {
				std::stringstream ss{std::string(e)};
				std::string tok;
				while (std::getline(ss, tok, ','))
					if (!tok.empty() && std::atoi(tok.c_str()) > 0) drops.insert(std::atoi(tok.c_str()));
			}
			for (const auto& [slots, _] : precom)
				for (int k : drops)
					FIDESlib::CKKS::AddBootstrapRaiseVariant(context, static_cast<int32_t>(slots), c, k);
		}
	}

	this->gpu	 = std::make_any<FIDESlib::CKKS::Context>(std::move(c));
	this->loaded = true;
}

size_t CryptoContextImpl<DCRTPoly>::FreeRotationKeys(const std::vector<int>& steps,
                                                     const PublicKey<DCRTPoly>& publicKey) {
	if (!this->loaded || this->devices.empty())
		return 0;

	auto& c      = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);
	const std::string keyID = pkImpl->GetKeyTag();
	const int half = c->N / 2;
	auto norm = [half](int i) { i %= half; if (i < 0) i += half; return i; };

	// Protect the bootstrap DFT rotation keys: they share the rot_keys map, so
	// removing one would break Bootstrap(). Gather every bootstrap automorphism
	// index for every set-up slot count.
	std::set<int> protect;
	auto cpu_cc = pkImpl->GetCryptoContext();
	auto fhe    = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cpu_cc->GetScheme()->m_FHE);
	if (fhe) {
		for (const auto& [slots, _] : fhe->m_bootPrecomMap)
			for (int b : FIDESlib::CKKS::GetBootstrapIndexes(cpu_cc, static_cast<int>(slots), nullptr))
				protect.insert(norm(b));
	}

	size_t freed = 0;
	for (int s : steps) {
		const int n = norm(s);
		if (n == 0 || protect.count(n))
			continue;
		if (c->RemoveRotationKey(s, keyID))
			++freed;
	}
	return freed;
}

void CryptoContextImpl<DCRTPoly>::LoadRotationKeys(const std::vector<int>& steps,
                                                   const PublicKey<DCRTPoly>& publicKey) {
	if (!this->loaded || this->devices.empty() || steps.empty())
		return;
	auto& c      = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);
	// AddRotationKeys dedups against already-resident keys (HasRotationKey) and only
	// transfers the missing ones; the OpenFHE eval keys were generated at setup.
	FIDESlib::CKKS::AddRotationKeys(pkImpl, c, steps);
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt) {
	// Delegate to the stash-aware overload: a plaintext staged by a residency worker
	// (ExtractRawPlaintext) must be found by EVERY load path — under FHE_STAGE_RELEASE_CPU its
	// OpenFHE-side payload is gone and an inline re-extract would throw. Value-identical:
	// with no staged entry this extracts inline exactly as before.
	this->LoadPlaintext(pt, nullptr);
}

namespace {

const size_t kStageArenaBytes = [] {
	const char* e	  = std::getenv("FHE_STAGE_ARENA_GB");
	const int	gb	  = (e && *e) ? std::atoi(e) : 3;
	return size_t(gb > 0 ? gb : 3) << 30;
}();

struct PinnedArena {
	uint8_t* base = nullptr;
	size_t	 cap  = 0;
	// Atomic so stage_into can be called from an OMP team (worker-side block staging):
	// offsets are reserved with fetch_add, the memcpys then run lock-free in parallel.
	std::atomic<size_t> used{0};
};
std::atomic<uint64_t> g_stage_overflow_pts{0};   // pts that fell back pageable since the last flip
// Prefill chunk-weight cache: while set, ExtractRawPlaintext keeps the
// OpenFHE-side payload even under FHE_STAGE_RELEASE_CPU, so a later chunk can RE-STAGE
// the same plaintexts instead of re-encoding them (~2 GB/block host for coeff pts).
std::atomic<bool> g_stage_release_suppressed{false};

struct StagedEntry {
	FIDESlib::CKKS::RawPlainText meta;   // arena!=null ⇒ sub_0 cleared (data in arena); else sub_0 kept
	const uint8_t*				 arena = nullptr;
	std::vector<size_t>			 off;    // per-limb byte offset into arena
	std::vector<size_t>			 len;    // per-limb byte length
	// COEFF-mode (MarkCoeffStaged): the single staged limb is a q0 EVAL limb; the load expands
	// it to target_limbs on the GPU instead of uploading pre-built limbs.
	bool						 coeff		  = false;
	int							 target_limbs = 0;
	int							 prescale_log2 = 0;   // un-prescale ×2^k applied by the GPU lift
	// Arena GENERATION this entry was staged under (509990c, multi-consume invalidation):
	// the entry dies when ITS arena is recycled, i.e. kStageSlots flips later. A flip-time
	// blanket clear is wrong under threaded prefetch, where the next block stages into the
	// other half while the current block is still loading.
	uint64_t					 gen           = 0;
};
// TWO halves are sufficient: a half holds HOST staging bytes, which are dead the instant the H2D
// retires (compute reads DEVICE memory), so the release point is H2D retirement, not end of compute.
constexpr int    kStageSlots = 2;
PinnedArena      g_stage_arena[kStageSlots];
std::atomic<int> g_stage_cur{0};
std::mutex		 g_stage_mutex;
// Bumped on every flip, under g_stage_mutex. Read lock-free by the multi-consume sweep.
std::atomic<uint64_t> g_stage_gen{0};

// A snapshot of the live staging arena, taken under g_stage_mutex.
// stage_arena_begin writes g_stage_cur AND can cudaFreeHost/cudaMallocHost the half's base,
// all under the mutex. An unlocked reader of `g_stage_arena[g_stage_cur]` can therefore
// observe a base pointer the flip has already freed, and memcpy several MB through it. Nothing in the block pipeline forces the flip and the staging
// onto the same thread (diagonal_linear flips on the MAIN thread and stages on the worker;
// stage_plaintexts fans the same read out to an OMP team), so this is reachable, and its
// signature is heap corruption surfacing far from here.
// Snapshotting under the lock is sufficient: a half's base/cap are assigned once (the
// `cap < kStageArenaBytes` branch) and `used` is atomic, so only the flip itself needs
// serialising against the read.
struct StageArenaView {
	PinnedArena* a	  = nullptr;
	uint8_t*	 base = nullptr;
	size_t		 cap  = 0;
	uint64_t	 gen  = 0;   // generation of THIS snapshot; stamped onto entries staged into it
};
StageArenaView view_of(PinnedArena& a) { return {&a, a.base, a.cap}; }

// Start staging a new block: ping-pong to the other arena, (lazily, once) allocate it, reset bump.
void stage_arena_begin();   // defined below; the owned variant wraps it

// ── per-slot monitors for the staging arena ──────────────────────────────────────────────
// The pipeline is deeper than the arena's halves, so a half could be recycled while the previous
// owner's ASYNC H2D still reads it (silently wrong weights). A host mutex cannot express "intact
// until the copy retires", so each half carries the id of the block that owns it: the producer
// blocks until the half it is about to flip into is free, and the consumer releases it after that
// block's compute. slot_owner[h] = owning block id, or -1 for free; owner < 0 = legacy unconditional flip.
std::mutex				 g_slot_mutex;
std::condition_variable	 g_slot_cv;
int              g_slot_owner[kStageSlots] = {-1, -1};

void stage_arena_release(int owner) {
	if (owner < 0) return;
	{
		std::lock_guard<std::mutex> g(g_slot_mutex);
		for (int h = 0; h < kStageSlots; ++h)
			if (g_slot_owner[h] == owner) g_slot_owner[h] = -1;
	}
	g_slot_cv.notify_all();
}

void stage_arena_begin_owned(int owner) {
	if (owner >= 0) {
		std::unique_lock<std::mutex> lk(g_slot_mutex);
		// IDEMPOTENT: a block owns at most one half. cpu_extract_block claims BEFORE the
		// per-plaintext `pt->loaded` early-out, so a block already staged (by the cross-token
		// prefetch) would otherwise claim a SECOND half next token — four claims against three
		// slots, and the ring deadlocks with xwait pinned and token 1 never arriving. Re-claiming
		// is wrong on its own terms too: the staged bytes live in the half this owner already
		// holds, so flipping would strand them.
		for (int h = 0; h < kStageSlots; ++h)
			if (g_slot_owner[h] == owner) return;
		// Block until the half we are about to flip INTO is free.
		const int next = (g_stage_cur.load(std::memory_order_relaxed) + 1) % kStageSlots;
		g_slot_cv.wait(lk, [&] { return g_slot_owner[next] < 0; });
		g_slot_owner[next] = owner;
	}
	stage_arena_begin();
}

void stage_arena_begin() {
	std::lock_guard<std::mutex> g(g_stage_mutex);
	g_stage_gen.fetch_add(1, std::memory_order_relaxed);
	g_stage_cur.store((g_stage_cur.load(std::memory_order_relaxed) + 1) % kStageSlots,
					  std::memory_order_relaxed);
	PinnedArena& a = g_stage_arena[g_stage_cur.load(std::memory_order_relaxed)];
	const uint64_t ov = g_stage_overflow_pts.exchange(0);
	if (ov > 0)
		std::fprintf(stderr, "[stage] arena flip: resetting used=%.2f GB cap=%.2f GB overflow_pts=%llu%s\n",
					 a.used.load() / 1e9, a.cap / 1e9, static_cast<unsigned long long>(ov),
					 ov > 0 ? " (raise FHE_STAGE_ARENA_GB)" : "");
	if (a.cap < kStageArenaBytes) {
		if (a.base)
			cudaFreeHost(a.base);
		void* p = nullptr;
		cudaMallocHost(&p, kStageArenaBytes);
		a.base = static_cast<uint8_t*>(p);
		a.cap  = a.base ? kStageArenaBytes : 0;
	}
	a.used = 0;
}

// Copy raw's limbs into arena `a` (host memcpy, no CUDA). On success sub_0 is cleared and
// arena/off/len set; on overflow/no-arena it falls back (sub_0 kept, arena=null) → pageable upload.
// Thread-safe: the plaintext's total bytes are reserved with ONE atomic fetch_add, so an OMP team
// can stage a block's plaintexts concurrently; the memcpys run lock-free into disjoint ranges.
StagedEntry stage_into(StageArenaView a, FIDESlib::CKKS::RawPlainText&& raw, bool narrow_ok = true) {
	// NATIVE-WIDTH staging (B12-plan step 1): a limb whose modulus fits 32 bits holds
	// residues < 2^32, so on u32 chains the arena stores 4 bytes/coefficient instead of
	// the historical 8 — half the pinned footprint and half the H2D bytes. The loader
	// discriminates by entry length (4N vs 8N). Chain-agnostic: on n64 every modulus is
	// > 2^32 and the layout is byte-identical to before. `narrow_ok=false` keeps the
	// 8-byte slot for COEFF-staged entries, whose GPU lift reads u64 lanes by contract.
	StagedEntry e;
	e.gen = a.gen;   // the flip this entry's bytes belong to (multi-consume sweep)
	auto limb_bytes = [&](size_t i) {
		const bool narrow = narrow_ok && i < raw.moduli.size() && raw.moduli[i] < (1ull << 32);
		return raw.sub_0[i].size() * (narrow ? sizeof(uint32_t) : sizeof(uint64_t));
	};
	size_t total = 0;
	for (size_t i = 0; i < raw.sub_0.size(); ++i)
		total += limb_bytes(i);
	bool ok = (a.base != nullptr) && total > 0;
	if (ok) {
		const size_t base_off = a.a->used.fetch_add(total, std::memory_order_relaxed);
		if (base_off + total > a.cap) {
			ok = false;   // reservation lost until the next flip resets the bump — arena is per block
			g_stage_overflow_pts.fetch_add(1, std::memory_order_relaxed);
		} else {
			e.off.reserve(raw.sub_0.size());
			e.len.reserve(raw.sub_0.size());
			size_t cur = base_off;
			for (size_t i = 0; i < raw.sub_0.size(); ++i) {
				auto& limb		   = raw.sub_0[i];
				const size_t bytes = limb_bytes(i);
				if (bytes == limb.size() * sizeof(uint32_t)) {
					auto* dst = reinterpret_cast<uint32_t*>(a.base + cur);
					for (size_t k = 0; k < limb.size(); ++k)
						dst[k] = (uint32_t)limb[k];
				} else {
					std::memcpy(a.base + cur, limb.data(), bytes);
				}
				e.off.push_back(cur);
				e.len.push_back(bytes);
				cur += bytes;
			}
		}
	}
	if (ok) {
		e.arena = a.base;
		raw.sub_0.clear();   // data now lives in the arena
	} else {
		e.arena = nullptr;   // fallback: keep sub_0 for a pageable load
		e.off.clear();
		e.len.clear();
	}
	e.meta = std::move(raw);
	return e;
}
// The live staging half, snapshotted under g_stage_mutex — the ONLY sanctioned way to reach it.
StageArenaView current_stage_arena() {
	std::lock_guard<std::mutex> g(g_stage_mutex);
	StageArenaView v = view_of(g_stage_arena[g_stage_cur.load(std::memory_order_relaxed)]);
	v.gen            = g_stage_gen.load(std::memory_order_relaxed);
	return v;
}

// ---- Persistent staging: for CONSTANT weights reloaded every token (lm_head tiles). Stage each
// once into a grow-once arena (never reset/ping-ponged), async-load every token (no re-extract).
// Gated by g_stage_persistent (set around such loads). ~one weight set, bounded. Entries are keyed
// by PlaintextImpl ADDRESS and erased from ~PlaintextImpl (PersistStagingForget): the allocator
// recycles addresses, so a weights_at()-releveled tile landing on a dead tile's address would
// silently reuse the stale staged limbs (post-handoff eager decode tok>=2 garbage, 48856712).
// Erasure keeps the arena bytes reserved (grow-once; overflow already falls back pageable).
// Map+mutex are intentionally immortal: plaintexts destroyed during static teardown still forget.
constexpr size_t kPersistArenaBytes = size_t(4) << 30;
PinnedArena		 g_persist_arena;
auto&		g_persist_staged = *new std::unordered_map<const void*, StagedEntry>();
auto&		g_persist_mutex	 = *new std::mutex();
// Atomic: written on the MAIN thread (gpt2_lm_head brackets its run_ops with
// SetPersistentStaging) and read on the residency worker inside ExtractRawPlaintext, which
// picks a different arena and a different map depending on it. A torn/stale read sends the
// two threads down different staging paths for the same plaintext.
std::atomic<bool>							 g_stage_persistent{false};
// Same thread pattern as g_stage_persistent above (driver sets it on the MAIN thread, the
// residency worker reads it inside LoadPlaintext), so it is atomic for the same reason.
std::atomic<bool>							 g_stage_multi_consume{false};

// Stage `raw` for plaintext `key` persistently (idempotent: no-op if already staged). Returns the
// stored entry. Caller holds g_persist_mutex.
StagedEntry& persist_stage_locked(const void* key, FIDESlib::CKKS::RawPlainText&& raw) {
	auto it = g_persist_staged.find(key);
	if (it != g_persist_staged.end())
		return it->second;   // already staged (constant weight) — reuse
	if (g_persist_arena.base == nullptr) {
		void* p = nullptr;
		cudaMallocHost(&p, kPersistArenaBytes);
		g_persist_arena.base = static_cast<uint8_t*>(p);
		g_persist_arena.cap	 = g_persist_arena.base ? kPersistArenaBytes : 0;
		g_persist_arena.used = 0;
	}
	// The persist arena is grow-once and never ping-ponged, and every caller holds
	// g_persist_mutex, so a plain view of it is stable for the duration of the stage.
	return g_persist_staged.emplace(key, stage_into(view_of(g_persist_arena), std::move(raw)))
		.first->second;
}

// ---- Async KV-cache offload arena (pinned, position-keyed, reused) ----
// The V cache is d_head (=64) cts/block and the K cache 1 ct/block → ~780 cts (~9 GB) per token,
// held host-side between tokens. A single REUSED pinned arena (memlock is unlimited) lets the
// per-block offload (D2H) and reload (H2D) run genuinely async and overlap compute — the K1/K2 path.
// Slots are keyed by a STABLE cache position (block+lane), allocated once on first offload and
// overwritten in place every token: a block's reload (at its compute) precedes its offload (at
// release), so in-place overwrite never races the consumer.
// env KV_ARENA_GB (default 12 GB): decode KV ≈ 9 GB + slack; prefill also stages cf.stg entries
// here, so max-staging (T>=128, chunk-4 ~221 entries) needs a larger arena (set KV_ARENA_GB=24).
static const size_t kKvArenaBytes = [] {
	const char* e = std::getenv("KV_ARENA_GB");
	const size_t gb = (e && *e && std::atoi(e) > 0) ? static_cast<size_t>(std::atoi(e)) : 12;
	return gb << 30;
}();
struct KvSlot {
	size_t						 off = 0;   // byte offset into g_kv_arena.base
	size_t						 cap = 0;   // reserved bytes (stable after token 0)
	FIDESlib::CKKS::StagedCtMeta meta;      // per-limb layout + ct metadata (refreshed each offload)
};
PinnedArena								g_kv_arena;
std::unordered_map<std::string, KvSlot> g_kv_slots;
std::mutex								g_kv_mutex;
std::future<void>						g_kv_prewarm;   // background 12GB cudaMallocHost (hides tok0's ~3.5s)

// Reserve (first sight) or reuse a stable pinned slot for a cache position. Caller holds g_kv_mutex.
KvSlot& kv_slot_ensure(const std::string& pos_key, size_t bytes) {
	if (g_kv_arena.base == nullptr) {
		void* p = nullptr;
		cudaMallocHost(&p, kKvArenaBytes);
		g_kv_arena.base = static_cast<uint8_t*>(p);
		g_kv_arena.cap	= g_kv_arena.base ? kKvArenaBytes : 0;
		g_kv_arena.used = 0;
	}
	KvSlot& s = g_kv_slots[pos_key];
	if (s.cap < bytes) {   // first sight or grew (runs once/position — sizes stable after token 0)
		if (g_kv_arena.used + bytes > g_kv_arena.cap)
			OPENFHE_THROW("KV pinned arena exhausted (raise kKvArenaBytes)");
		s.off = g_kv_arena.used;
		s.cap = bytes;
		g_kv_arena.used += bytes;
	}
	return s;
}
}   // namespace

// Called from ~PlaintextImpl: drop the address-keyed persist-staged entry with its object.
void PersistStagingForget(const void* key) {
	std::lock_guard<std::mutex> g(g_persist_mutex);
	g_persist_staged.erase(key);
}

// Kick off the pinned stage-arena allocations (2 × FHE_STAGE_ARENA_GB) on a background thread,
// once — worker-side block staging (ViT/prefill) uses block-sized arenas whose cudaMallocHost
// costs seconds; prewarming at driver init hides it under context setup. stage_arena_begin's
// lazy-alloc branch stays as the fallback and both run under g_stage_mutex, so whichever side
// wins the race publishes the arena and the loser's allocation is dropped.
namespace {
std::future<void> g_stage_prewarm;
}
void PrewarmStageArenas() {
	static std::once_flag once;
	std::call_once(once, [] {
		g_stage_prewarm = std::async(std::launch::async, [] {
			for (int i = 0; i < kStageSlots; ++i) {
				void* p = nullptr;
				cudaMallocHost(&p, kStageArenaBytes);
				std::lock_guard<std::mutex> g(g_stage_mutex);
				PinnedArena& a = g_stage_arena[i];
				if (a.base == nullptr && p) {
					a.base = static_cast<uint8_t*>(p);
					a.cap  = kStageArenaBytes;
					a.used = 0;
				} else if (p) {
					cudaFreeHost(p);   // lazy-alloc won the race — drop ours
				}
			}
		});
	});
}

void CryptoContextImpl<DCRTPoly>::SuppressStageReleaseCpu(bool suppress) {
	g_stage_release_suppressed.store(suppress, std::memory_order_relaxed);
}

void CryptoContextImpl<DCRTPoly>::BeginStageBlock() {
	{
		stage_arena_begin();
		// Multi-consume mode keeps staged entries across loads. An entry must die
		// exactly when ITS arena is recycled — that is THIS flip for entries staged
		// two generations ago (same ping-pong parity). A blanket clear here is
		// wrong under threaded prefetch: the next block stages while the current
		// one still loads, and clearing would orphan the current block's entries
		// (a coeff "consumed twice" throw). Stale survivors past their generation
		// would alias recycled bytes, hence the sweep below.
		if (g_stage_multi_consume && prefetched_raw_mutex) {
			const uint64_t G = g_stage_gen.load(std::memory_order_relaxed);
			prefetched_raw_mutex->lock();
			for (auto it = prefetched_raw.begin(); it != prefetched_raw.end();) {
				if (it->second.type() == typeid(StagedEntry) &&
					std::any_cast<const StagedEntry&>(it->second).gen + kStageSlots <= G)
					it = prefetched_raw.erase(it);
				else
					++it;
			}
			prefetched_raw_mutex->unlock();
		}
	}
}

// owner >= 0 arms the monitor: claim a half for this block and block until it is free.
void CryptoContextImpl<DCRTPoly>::BeginStageBlockOwned(int owner) {
	stage_arena_begin_owned(owner);
}

// Call once the block's compute is done — this is what makes the half reusable, and it is
// deliberately NOT at enqueue time, which is where a mutex would have released it.
void CryptoContextImpl<DCRTPoly>::ReleaseStageBlock(int owner) {
	stage_arena_release(owner);
}

void CryptoContextImpl<DCRTPoly>::SetPersistentStaging(bool on) { g_stage_persistent = on; }

// Multi-chunk forwards (BERT seq-128 = 4 packed chunks) load the SAME weight
// plaintext once per chunk within one block; erase-on-load makes reload 2..C
// miss ("consumed twice" throw for coeff pts). Multi-consume keeps entries in
// the map until the block flip (arena bytes live exactly that long anyway).
void CryptoContextImpl<DCRTPoly>::SetStageMultiConsume(bool on) { g_stage_multi_consume = on; }

double CryptoContextImpl<DCRTPoly>::ScalingFactorReal(uint32_t level) const {
	const auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto	cp		= std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	if (!cp)
		OPENFHE_THROW("ScalingFactorReal: not a CKKS-RNS context");
	return cp->GetScalingFactorReal(level);
}

// Total q-limbs (PRIMES) in this context. `multiplicative_depth` counts CKKS LEVELS, and a
// composite-scaling chain carries `d` primes per level, so `multiplicative_depth + 1` is the
// prime count ONLY at d == 1. A coeff-staged plaintext's target_limbs is a PRIME count and
// must therefore be derived from the element params, which are unit-correct on any chain.
// Mixing the two made target_limbs go NEGATIVE on d=2 (26 + 1 - 34).
// Composite degree of this context (1 on classic chains). The coeff lift stages exactly this
// many source limbs — see MarkCoeffStaged.
static uint32_t composite_degree_of(const std::any& cpu_ctx) {
	const auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cpu_ctx);
	const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	return (cp && cp->GetCompositeDegree() > 0) ? cp->GetCompositeDegree() : 1u;
}
static size_t total_q_limbs(const std::any& cpu_ctx) {
	const auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cpu_ctx);
	return context->GetCryptoParameters()->GetElementParams()->GetParams().size();
}

void CryptoContextImpl<DCRTPoly>::MarkCoeffStaged(Plaintext& pt, uint32_t target_level, double target_scale,
                                                  int prescale_log2) {
	auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
	if (pt->device_encode) {   // the coeff path stages a host encode
		ptImpl->Encode();
		pt->device_encode = false;
	}
	// The coeff lift reconstructs from the first `d` primes (Garner), so the host encode must
	// leave exactly d limbs — one on a classic chain, the whole first-mod group on a composite
	// one. A single 28-bit prime cannot carry a 2^54-scaled coefficient, so d=1 on a composite
	// chain is not merely suboptimal, it is unrepresentable.
	const uint32_t _d = composite_degree_of(this->cpu);
	if (ptImpl->GetElement<lbcrypto::DCRTPoly>().GetAllElements().size() != _d)
		OPENFHE_THROW("MarkCoeffStaged: expected a " + std::to_string(_d) +
		              "-limb host encode (composite degree), got " +
		              std::to_string(ptImpl->GetElement<lbcrypto::DCRTPoly>().GetAllElements().size()));
	ptImpl->SetLevel(target_level);
	ptImpl->SetScalingFactor(target_scale);
	pt->coeff_staged = true;
	pt->coeff_prescale_log2 = prescale_log2;
}

// Called from ~PlaintextImpl: drop a worker-staged entry that was never consumed. Without this,
// the allocator can recycle the address into a NEW plaintext, whose first load would silently
// upload the dead object's staged limbs (the g_persist_staged 48856712 bug class).
void CryptoContextImpl<DCRTPoly>::ForgetPrefetchedRaw(const void* key) {
	if (!prefetched_raw_mutex)
		return;
	// Move the entry out and let it die AFTER the lock. A StagedEntry owns a RawPlainText,
	// which owns an lbcrypto::Plaintext reference — dropping the last one runs a destructor
	// that can re-enter this function, and prefetched_raw_mutex is a NON-recursive
	// shared_mutex. RAII rather than raw lock/unlock so a throw cannot strand it either.
	std::any dead;
	{
		std::unique_lock<std::shared_mutex> lk(*prefetched_raw_mutex);
		auto it = prefetched_raw.find(key);
		if (it == prefetched_raw.end())
			return;
		dead = std::move(it->second);
		prefetched_raw.erase(it);
	}
}

void CryptoContextImpl<DCRTPoly>::PrewarmKvArena() {
	// Allocate the 12GB pinned KV-offload arena on a background thread ONCE, at decode init, so the
	// ~3.5s cudaMallocHost overlaps token-0's block compute instead of stalling the first offload
	// (REL(0)). kv_slot_ensure publishes the arena under g_kv_mutex; whichever finishes first wins
	// and the loser frees its allocation, so there is never a double-resident 24GB.
	static std::once_flag once;
	std::call_once(once, [] {
		g_kv_prewarm = std::async(std::launch::async, [] {
			void* p = nullptr;
			cudaMallocHost(&p, kKvArenaBytes);   // ~3.5s pinning 12GB (pinned pages commit at alloc)
			std::lock_guard<std::mutex> g(g_kv_mutex);
			if (g_kv_arena.base == nullptr) {
				g_kv_arena.base = static_cast<uint8_t*>(p);
				g_kv_arena.cap	= p ? kKvArenaBytes : 0;
				g_kv_arena.used = 0;
			} else if (p) {
				cudaFreeHost(p);   // kv_slot_ensure won the race — drop ours
			}
		});
	});
}

bool CryptoContextImpl<DCRTPoly>::KvStoreStaged(Ciphertext<DCRTPoly>& ct, const std::string& pos_key,
												cudaStream_t stream) {
	if (!ct->loaded)
		return false;
	if (this->devices.empty() || !this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	auto		 ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	const size_t bytes	= ct_gpu->staged_bytes();

	std::lock_guard<std::mutex> g(g_kv_mutex);
	KvSlot&						slot = kv_slot_ensure(pos_key, bytes);
	ct_gpu->storeStaged(g_kv_arena.base + slot.off, slot.meta, stream);   // async D->H, no sync

	// storeStaged omits the per-limb moduli (load needs them) — fill from the modulus chain. A ct at
	// numRes residues uses the first numRes ciphertext (Q) primes.
	auto&		context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& chain	= context->GetCryptoParameters()->GetElementParams()->GetParams();
	slot.meta.moduli.clear();
	slot.meta.moduli.reserve(slot.meta.numRes);
	for (int i = 0; i < slot.meta.numRes; ++i)
		slot.meta.moduli.push_back(chain[i]->GetModulus().ConvertToInt());
	// Device copy is STILL valid (D->H only reads it) until KvEvict — leave ct->loaded=true so a
	// stray reader is correct; the caller evicts after the offload stream is synchronised.
	return true;
}

void CryptoContextImpl<DCRTPoly>::KvEvict(Ciphertext<DCRTPoly>& ct) {
	if (!ct->loaded)
		return;
	std::lock_guard<std::mutex> g(g_kv_mutex);
	this->EvictDeviceCiphertext(ct->gpu);   // free device; pinned slot holds the data
	ct->loaded = false;
}

void CryptoContextImpl<DCRTPoly>::KvLoadStaged(Ciphertext<DCRTPoly>& ct, const std::string& pos_key,
											   cudaStream_t stream) {
	if (ct->loaded || this->devices.empty())
		return;
	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	std::lock_guard<std::mutex> g(g_kv_mutex);
	auto						it = g_kv_slots.find(pos_key);
	if (it == g_kv_slots.end()) {
		OPENFHE_THROW("KvLoadStaged: no pinned slot for cache position " + pos_key);
	}
	KvSlot&	 slot		 = it->second;
	auto&	 context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto	 gpu_ct		 = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu);   // empty shell
	gpu_ct->loadStaged(g_kv_arena.base + slot.off, slot.meta, stream);                  // async H->D
	uint32_t handle	   = this->RegisterDeviceCiphertext(std::move(gpu_ct));
	ct->gpu			   = handle;
	ct->loaded		   = true;
	ct->original_level = this->multiplicative_depth - ct->GetLevel();
}


// Kept for API compatibility: the LoadPlaintext timing probe was removed, so this is a no-op.
extern "C" void AcqProbeReport(int /*tok*/) {}

// FIDESLIB_GPU_ENCODE: a device-encoded plaintext's limbs, dumped in the device layout to pinned host memory at its
// first load, so an evicted plaintext reloads by one async H2D per limb (no host encode, no OpenFHE repacking).
// Blocks come from 1 GiB pinned slabs and are recycled by size; ~PlaintextImpl releases them (DeviceDumpForget).
namespace {
struct DeviceDump {
	FIDESlib::CKKS::RawPlainText meta;
	uint8_t*					 base = nullptr;
	size_t						 bytes = 0;
	std::vector<size_t>			 off, len;
	int							 limbs = 0;   // the plaintext's; the dump keeps the first composite group
};
std::mutex										 g_dump_mutex;
std::unordered_map<const void*, DeviceDump>		 g_dumps;
std::unordered_map<size_t, std::vector<uint8_t*>> g_dump_free;
// device handle -> (plaintext, pinned max |coefficient| of its encode, or null for degree > 1): dumped at eviction
std::unordered_map<uint32_t, std::pair<const void*, double*>> g_undumped;
std::vector<double*> g_cmax_free;   // pinned slots, recycled
size_t g_dump_bytes = 0, g_full_bytes = 0;
uint8_t*										 g_slab	  = nullptr;
size_t											 g_slab_left = 0;

uint8_t* dump_alloc_locked(size_t bytes) {
	auto& fl = g_dump_free[bytes];
	if (!fl.empty()) {
		uint8_t* p = fl.back();
		fl.pop_back();
		return p;
	}
	if (g_slab_left < bytes) {
		const size_t sz = std::max(bytes, size_t(1) << 30);
		if (cudaMallocHost(reinterpret_cast<void**>(&g_slab), sz) != cudaSuccess)
			OPENFHE_THROW("device dump: cudaMallocHost failed");
		g_slab_left = sz;
	}
	uint8_t* p = g_slab;
	g_slab += bytes;
	g_slab_left -= bytes;
	return p;
}
}   // namespace

void DeviceDumpForget(const void* key) {
	std::lock_guard<std::mutex> g(g_dump_mutex);
	for (auto u = g_undumped.begin(); u != g_undumped.end();) {
		if (u->second.first != key) {
			++u;
			continue;
		}
		if (u->second.second)
			g_cmax_free.push_back(u->second.second);
		u = g_undumped.erase(u);
	}
	auto it = g_dumps.find(key);
	if (it == g_dumps.end())
		return;
	g_dump_free[it->second.bytes].push_back(it->second.base);
	g_dumps.erase(it);
}

// The dump keeps only the first composite group (d limbs): a weight's coefficients are integers below that group's
// product, so the reload's Garner lift (loadCoeffExpand) rebuilds every other limb exactly.
static void dump_device_plaintext_locked(const void* key, FIDESlib::CKKS::Plaintext& dev, const double* cmax) {
	DeviceDump d;
	cudaStreamSynchronize(dev.c0.GPU.at(0).s.ptr());   // the encode (and its max |coefficient| copy) is done
	d.limbs		= dev.c0.getLevel() + 1;
	// compact when every integer coefficient fits a quarter of the first composite group's product
	double group = 1;
	for (int l = 0; l < dev.cc.compositeDegree(); ++l)
		group *= (double)dev.cc.prime[l].p;
	const bool compact = cmax && *cmax < 0.25 * group;
	const int  src	   = compact ? std::min(d.limbs, dev.cc.compositeDegree()) : d.limbs;
	d.bytes = (size_t)src * dev.cc.N * sizeof(uint32_t);
	for (int l = 0; l < src; ++l)
		d.meta.moduli.push_back(dev.cc.prime[l].p);
	d.meta.numRes	  = src;
	d.meta.N		  = dev.cc.N;
	d.meta.format	  = Format::EVALUATION;
	d.meta.Noise	  = dev.NoiseFactor;
	d.meta.NoiseLevel = dev.NoiseLevel;
	d.meta.slots	  = dev.slots;
	d.base		  = dump_alloc_locked(d.bytes);
	size_t cursor = 0;
	const cudaStream_t ps = dev.c0.GPU.at(0).s.ptr();
	dev.c0.storeStaged(d.base, cursor, d.off, d.len, ps, src);
	cudaStreamSynchronize(ps);
	g_dump_bytes += d.bytes;
	g_full_bytes += (size_t)d.limbs * dev.cc.N * sizeof(uint32_t);
	if (cursor != d.bytes)
		OPENFHE_THROW("device dump: " + std::to_string(cursor) + " bytes stored, " + std::to_string(d.bytes) + " expected");
	g_dumps[key] = std::move(d);
}

// FIDESLIB_GPU_ENCODE_CHECK=1: a reloaded plaintext against a fresh device encode of its values, limb for limb.
static long g_check_n = 0, g_check_bad = 0, g_check_compact = 0;
static void check_reload_locked(FIDESlib::CKKS::Plaintext& got, const lbcrypto::Plaintext& cpu_pt, size_t total_limbs,
                                bool compact) {
	FIDESlib::CKKS::Plaintext ref(got.cc_);
	cudaStreamSynchronize(FIDESlib::CKKS::encodeOnDevice(ref, cpu_pt->GetCKKSPackedValue(),
	                                                    static_cast<int>(total_limbs - cpu_pt->GetLevel()),
	                                                    cpu_pt->GetScalingFactor(),
	                                                    static_cast<int>(cpu_pt->GetNoiseScaleDeg())));
	cudaDeviceSynchronize();
	std::vector<std::vector<uint64_t>> a, b;
	got.c0.store(a);
	ref.c0.store(b);
	cudaDeviceSynchronize();
	long bad = a.size() != b.size() || got.NoiseFactor != ref.NoiseFactor;
	for (size_t l = 0; l < a.size() && l < b.size(); ++l)
		for (size_t k = 0; k < a[l].size(); ++k)
			bad += a[l][k] != b[l][k];
	++g_check_n;
	g_check_compact += compact;
	g_check_bad += bad;
	if (g_check_n % 256 == 0 || bad)
		std::cerr << "[gpu_encode_check] reloads=" << g_check_n << " compact=" << g_check_compact
		          << " mismatches=" << g_check_bad << " dump_MB=" << (g_dump_bytes >> 20) << " full_MB=" << (g_full_bytes >> 20)
		          << std::endl;
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt, cudaStream_t stream_override) {
	if (pt->loaded || this->devices.empty())
		return;

	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt;
	gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu);
	const cudaStream_t load_stream = ResolvePlaintextLoadStream(stream_override);

	if (pt->device_encode) {
		const void* dkey = static_cast<const void*>(pt.get());
		bool		reloaded = false;
		{
			std::lock_guard<std::mutex> g(g_dump_mutex);
			auto it = g_dumps.find(dkey);
			if (it != g_dumps.end()) {
				const DeviceDump& d = it->second;
				if ((int)d.off.size() < d.limbs)
					gpu_pt->loadCoeffExpand(d.meta, d.base, d.off, d.len, (int)d.off.size(), d.limbs, load_stream, 0, true);
				else
					gpu_pt->loadStaged(d.meta, d.base, d.off, d.len, load_stream);
				reloaded = true;
				if (std::getenv("FIDESLIB_GPU_ENCODE_CHECK"))
					check_reload_locked(*gpu_pt, std::any_cast<const lbcrypto::Plaintext&>(pt->cpu), total_q_limbs(this->cpu),
					                    (int)d.off.size() < d.limbs);
			}
		}
		cudaStream_t ready = load_stream;
		double*		 cmax = nullptr;
		if (!reloaded) {
			const auto& cpu_pt = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
			if (cpu_pt->GetNoiseScaleDeg() == 1) {
				std::lock_guard<std::mutex> g(g_dump_mutex);
				if (g_cmax_free.empty()) {
					double* block = nullptr;
					cudaMallocHost(reinterpret_cast<void**>(&block), 4096 * sizeof(double));
					for (int i = 0; i < 4096; ++i)
						g_cmax_free.push_back(block + i);
				}
				cmax = g_cmax_free.back();
				g_cmax_free.pop_back();
			}
			ready = FIDESlib::CKKS::encodeOnDevice(*gpu_pt, cpu_pt->GetCKKSPackedValue(),
			                                       static_cast<int>(total_q_limbs(this->cpu) - cpu_pt->GetLevel()),
			                                       cpu_pt->GetScalingFactor(),
			                                       static_cast<int>(cpu_pt->GetNoiseScaleDeg()), cmax);
			if (!plaintext_streams_enabled)
				cudaStreamSynchronize(ready);
		}
		uint32_t handle = this->RegisterDevicePlaintext(std::move(gpu_pt));
		pt->gpu			= handle;
		pt->loaded		= true;
		if (!reloaded) {
			std::lock_guard<std::mutex> g(g_dump_mutex);
			g_undumped[handle] = {dkey, cmax};
		}
		RecordPlaintextReady(handle, ready);
		return;
	}

	const void* key = static_cast<const void*>(pt.get());

	// Persistent staging (constant lm_head tiles): async-load from the persistent arena; stage on the
	// first miss (tok0). The entry is never erased (constant weight) so the arena pointer is stable.
	if (g_stage_persistent) {
		std::lock_guard<std::mutex> g(g_persist_mutex);
		auto		 it = g_persist_staged.find(key);
		StagedEntry* e	= nullptr;
		if (it != g_persist_staged.end()) {
			e = &it->second;
		} else {
			auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
			const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
			e = &persist_stage_locked(key, FIDESlib::CKKS::GetRawPlainText(context, ptImpl));
			if (pt->coeff_staged) {
				if (e->arena == nullptr)
					OPENFHE_THROW("LoadPlaintext: coeff-staged plaintext overflowed the persistent arena");
				e->coeff		= true;
				e->target_limbs = static_cast<int>(total_q_limbs(this->cpu) - pt->GetLevel());
				e->prescale_log2 = pt->coeff_prescale_log2;
			}
		}
		if (e->coeff) {
			gpu_pt->loadCoeffExpand(e->meta, e->arena, e->off, e->len,
			                        (int)composite_degree_of(this->cpu), e->target_limbs, load_stream,
			                        e->prescale_log2);
		} else if (e->arena != nullptr) {
			gpu_pt->loadStaged(e->meta, e->arena, e->off, e->len, load_stream);
		} else if (load_stream != nullptr) {
			gpu_pt->load(e->meta, load_stream);   // overflow fallback: pageable from the kept sub_0
		} else {
			gpu_pt->load(e->meta);
		}
		uint32_t handle = this->RegisterDevicePlaintext(std::move(gpu_pt));
		pt->gpu			= handle;
		pt->loaded		= true;
		RecordPlaintextReady(handle, load_stream);
		return;
	}

	// Consume the worker-pre-extracted entry if present: either a StagedEntry (FHE_PIN_STAGE — async
	// H2D from the pinned arena) or a plain RawPlainText (FHE_CPU_PREFETCH). Else extract inline.
	StagedEntry				   staged;
	bool					   have_staged = false;
	FIDESlib::CKKS::RawPlainText raw_pt;
	bool					   from_stash = false;
	if (prefetched_raw_mutex) {
		// RAII: the any_casts below can throw (a std::any holding neither type), and the raw
		// lock/unlock pair this replaces would then have stranded the mutex for the process.
		std::unique_lock<std::shared_mutex> lk(*prefetched_raw_mutex);
		auto it = prefetched_raw.find(key);
		if (it != prefetched_raw.end()) {
			if (it->second.type() == typeid(StagedEntry)) {
				if (g_stage_multi_consume) {
					staged = std::any_cast<const StagedEntry&>(it->second);   // copy: reload per chunk
				} else {
					staged = std::move(std::any_cast<StagedEntry&>(it->second));
				}
				have_staged = true;
			} else {
				if (g_stage_multi_consume) {
					raw_pt = std::any_cast<const FIDESlib::CKKS::RawPlainText&>(it->second);
				} else {
					raw_pt = std::move(std::any_cast<FIDESlib::CKKS::RawPlainText&>(it->second));
				}
				from_stash = true;
			}
			if (!g_stage_multi_consume)
				prefetched_raw.erase(it);   // single-consumption (proven default)
		}
	}

	if (have_staged && staged.coeff) {
		gpu_pt->loadCoeffExpand(staged.meta, staged.arena, staged.off, staged.len,
								(int)composite_degree_of(this->cpu), staged.target_limbs, load_stream,
								staged.prescale_log2);
	} else if (have_staged && staged.arena != nullptr) {
		gpu_pt->loadStaged(staged.meta, staged.arena, staged.off, staged.len, load_stream);
	} else {
		// A coeff-marked plaintext carries only its first d limbs — it MUST come through the
		// staged path (a plain short upload at a claimed deeper level would be silently wrong).
		if (pt->coeff_staged)
			OPENFHE_THROW("LoadPlaintext: coeff-staged plaintext has no staged entry (consumed "
						  "twice, or staged before the arena was armed) — this is a bug");
		FIDESlib::CKKS::RawPlainText* src = have_staged ? &staged.meta : &raw_pt;   // staged-fallback keeps sub_0
		if (!have_staged && !from_stash) {
			auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
			const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
			raw_pt			   = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
		}
		if (load_stream != nullptr) {
			gpu_pt->load(*src, load_stream);
		} else {
			gpu_pt->load(*src);
		}
	}
	uint32_t handle = this->RegisterDevicePlaintext(std::move(gpu_pt));
	pt->gpu			= handle;
	pt->loaded		= true;
	RecordPlaintextReady(handle, load_stream);
}

void CryptoContextImpl<DCRTPoly>::ExtractRawPlaintext(Plaintext& pt) {
	// CPU-only; no CUDA, no device state. Builds the host RawPlainText (GetRawPlainText = the heavy
	// limb-copy + bit-reverse) so a later LoadPlaintext uploads it without re-extracting. Safe on a
	// residency worker thread during compute. No-op if already on device or not loadable yet.
	if (pt->loaded || pt->device_encode || this->devices.empty() || !this->loaded || !prefetched_raw_mutex)
		return;
	const void* key = static_cast<const void*>(pt.get());

	// Persistent staging (constant lm_head tiles): stage once, idempotent — the check is against
	// g_persist_staged BEFORE the expensive GetRawPlainText so tok1+ is a true no-op.
	if (g_stage_persistent) {
		std::lock_guard<std::mutex> g(g_persist_mutex);
		if (g_persist_staged.find(key) != g_persist_staged.end())
			return;
		auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
		persist_stage_locked(key, FIDESlib::CKKS::GetRawPlainText(context, ptImpl));
		return;
	}

	{   // idempotent: skip if already extracted (shared lock, no double GetRawPlainText)
		std::shared_lock<std::shared_mutex> lk(*prefetched_raw_mutex);
		if (prefetched_raw.find(key) != prefetched_raw.end())
			return;
	}
	// This probe is only a fast path — it is NOT a claim on `key`. Everything below runs
	// unlocked for milliseconds (GetRawPlainText copies several MB, stage_into copies them
	// again), and in that window ~PlaintextImpl can call ForgetPrefetchedRaw(key) or
	// LoadPlaintext can consume-and-erase. The insert at the end therefore re-checks under
	// the exclusive lock instead of assuming the probe still holds; see there.
	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
	if (pt->coeff_staged && raw.numRes != (int)composite_degree_of(this->cpu))
		OPENFHE_THROW("ExtractRawPlaintext: coeff-staged plaintext must carry exactly composite_degree limbs");
	// Stage into the pinned arena (host memcpy on this worker — overlapped, no CUDA call) when
	// FHE_PIN_STAGE; else stash the raw for a pageable upload. Coeff-staged entries keep the
	// 8-byte slot (their GPU lift reads u64 lanes by contract — see stage_into narrow_ok).
	const bool narrow_ok = !pt->coeff_staged;
	std::any entry = std::any(stage_into(current_stage_arena(), std::move(raw), narrow_ok));
	if (pt->coeff_staged) {
		if (entry.type() != typeid(StagedEntry) || std::any_cast<const StagedEntry&>(entry).arena == nullptr)
			OPENFHE_THROW("ExtractRawPlaintext: coeff-staged plaintext requires the pinned arena "
						  "(FHE_PIN_STAGE on, arena not overflowed) — raise FHE_STAGE_ARENA_GB");
		StagedEntry& se = std::any_cast<StagedEntry&>(entry);
		se.coeff		= true;
		// pt->GetLevel() reports the TARGET level (MarkCoeffStaged); limbs = depth+1 - level.
		se.target_limbs = static_cast<int>(total_q_limbs(this->cpu) - pt->GetLevel());
		se.prescale_log2 = pt->coeff_prescale_log2;
	}
	// FHE_STAGE_RELEASE_CPU: once the limbs live in the pinned arena, the OpenFHE-side DCRTPoly is
	// redundant (~4-6 MB/pt; a ViT block is ~65 GB) — drop it so staged blocks don't double-hold
	// host RAM. Only on a successful arena stage (the pageable fallback still reads sub_0/meta).
	// A second extraction of a released plaintext throws LOUDLY in GetRawPlainText — never silent.
	// The encoding's value vector survives (GetRealPackedValue/GetLevel read metadata, not the poly).
	static const bool release_cpu = [] {
		const char* e = std::getenv("FHE_STAGE_RELEASE_CPU");
		return e && *e && std::atoi(e) != 0;
	}();
	if (release_cpu && !g_stage_release_suppressed.load(std::memory_order_relaxed) &&
		entry.type() == typeid(StagedEntry) &&
		std::any_cast<const StagedEntry&>(entry).arena != nullptr) {
		auto& pt_nc = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		pt_nc->GetElement<lbcrypto::DCRTPoly>() = lbcrypto::DCRTPoly();
	}
	// Publish. `operator[] = std::move(entry)` was wrong twice over: it DESTROYS whatever the
	// key already held while the exclusive lock is held (that destructor can drop the last
	// lbcrypto::Plaintext reference and re-enter ForgetPrefetchedRaw on this same
	// non-recursive mutex), and it silently overwrites an entry that appeared while we were
	// working — including one belonging to a DIFFERENT plaintext that the allocator has since
	// placed at this address, which is precisely the address-recycling bug class
	// ForgetPrefetchedRaw exists to prevent. try_emplace instead: first writer wins, and our
	// loser copy is destroyed after the lock is released.
	{
		std::any loser;
		{
			std::unique_lock<std::shared_mutex> lk(*prefetched_raw_mutex);
			// try_emplace leaves `entry` untouched when it does not insert, by contract.
			if (!prefetched_raw.try_emplace(key, std::move(entry)).second)
				loser = std::move(entry);   // someone published first — keep theirs, drop ours
		}
	}
}

void CryptoContextImpl<DCRTPoly>::LoadCiphertext(Ciphertext<DCRTPoly>& ct) {
	if (ct->loaded || this->devices.empty())
		return;

	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	std::shared_ptr<FIDESlib::CKKS::Ciphertext> gpu_ct;
	// Guard the host stash: the async KV pipeline reloads (this erase) on a worker
	// thread concurrently with an offload (insert) on the main thread.
	if (offloaded_ciphertexts_mutex)
		offloaded_ciphertexts_mutex->lock();
	auto off = this->offloaded_ciphertexts.find(ct->gpu);
	if (off != this->offloaded_ciphertexts.end()) {
		// Re-upload a ciphertext previously offloaded by StoreDeviceCiphertext.
		// store()/load() are the device-native pair (no bit-reversal, unlike the
		// OpenFHE import path), so this round-trips exactly. ct->cpu is untouched.
		auto& raw_ct = std::any_cast<FIDESlib::CKKS::RawCipherText&>(off->second);
		gpu_ct		 = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
		this->offloaded_ciphertexts.erase(off);
		if (offloaded_ciphertexts_mutex)
			offloaded_ciphertexts_mutex->unlock();
	} else {
		if (offloaded_ciphertexts_mutex)
			offloaded_ciphertexts_mutex->unlock();
		auto& context						 = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		const auto& ctImpl					 = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		// The only way to re-upload is from the CPU shadow. A metadata-only shadow
		// (FIDESLIB_LAZY_CPU_SHADOW) has no limbs to upload — a ciphertext that was
		// evicted without being stashed by StoreDeviceCiphertext is unrecoverable.
		// Throw instead of uploading zeros: loud, never silently wrong.
		if (ctImpl->GetElements().empty()) {
			OPENFHE_THROW("LoadCiphertext: ciphertext has a metadata-only CPU shadow and is not in the "
						  "offload stash — cannot re-upload (FIDESLIB_LAZY_CPU_SHADOW)");
		}
		FIDESlib::CKKS::RawCipherText raw_ct = FIDESlib::CKKS::GetRawCipherText(context, ctImpl);
		gpu_ct								 = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
	}
	uint32_t handle	   = this->RegisterDeviceCiphertext(std::move(gpu_ct));
	ct->gpu			   = handle;
	ct->loaded		   = true;
	ct->original_level = this->multiplicative_depth - ct->GetLevel();
}

bool CryptoContextImpl<DCRTPoly>::StoreDeviceCiphertext(Ciphertext<DCRTPoly>& ct) {
	if (!ct->loaded)
		return false;
	if (this->devices.empty() || !this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct_gpu->store(raw_ct);	// device -> host: numRes / sub_0 / sub_1 / NoiseLevel / Noise / keyid / slots

	// store() omits the per-limb moduli (only the OpenFHE import path fills them),
	// but load() needs them — fill from the context modulus chain. A ct at numRes
	// residues uses the first numRes ciphertext (Q) primes.
	auto& context	  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& chain = context->GetCryptoParameters()->GetElementParams()->GetParams();
	raw_ct.moduli.clear();
	raw_ct.moduli.reserve(raw_ct.numRes);
	for (int i = 0; i < raw_ct.numRes; ++i)
		raw_ct.moduli.push_back(chain[i]->GetModulus().ConvertToInt());

	// Stash host-side keyed by the (monotonic, never-reused) handle, free device,
	// and KEEP ct->gpu as that key + ct->cpu as its original OpenFHE shell — so a
	// cpu-reading op (clone/metadata) still sees a valid ct while offloaded, and
	// LoadCiphertext reconstructs from offloaded_ciphertexts[ct->gpu].
	const uint32_t key = ct->gpu;
	if (!this->EvictDeviceCiphertext(key)) {
		OPENFHE_THROW("StoreDeviceCiphertext: could not evict ciphertext from device");
	}
	if (offloaded_ciphertexts_mutex)
		offloaded_ciphertexts_mutex->lock();
	this->offloaded_ciphertexts[key] = std::move(raw_ct);
	if (offloaded_ciphertexts_mutex)
		offloaded_ciphertexts_mutex->unlock();
	ct->loaded = false;
	return true;
}

// Drain-free offload (K0): identical to StoreDeviceCiphertext(ct) but the D->H
// store skips the two whole-device cudaDeviceSynchronize() (per-limb syncs still
// guarantee completion). Used by the async KV pipeline so a residency worker
// thread can offload block i while the main thread computes block i+1.
bool CryptoContextImpl<DCRTPoly>::StoreDeviceCiphertext(Ciphertext<DCRTPoly>& ct, cudaStream_t stream) {
	if (!ct->loaded)
		return false;
	if (this->devices.empty() || !this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct_gpu->store(raw_ct, stream);	// drain-free device -> host

	auto& context	  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& chain = context->GetCryptoParameters()->GetElementParams()->GetParams();
	raw_ct.moduli.clear();
	raw_ct.moduli.reserve(raw_ct.numRes);
	for (int i = 0; i < raw_ct.numRes; ++i)
		raw_ct.moduli.push_back(chain[i]->GetModulus().ConvertToInt());

	const uint32_t key = ct->gpu;
	if (!this->EvictDeviceCiphertext(key)) {
		OPENFHE_THROW("StoreDeviceCiphertext: could not evict ciphertext from device");
	}
	if (offloaded_ciphertexts_mutex)
		offloaded_ciphertexts_mutex->lock();
	this->offloaded_ciphertexts[key] = std::move(raw_ct);
	if (offloaded_ciphertexts_mutex)
		offloaded_ciphertexts_mutex->unlock();
	ct->loaded = false;
	return true;
}

// ---- Key Generation ----

KeyPair<DCRTPoly> CryptoContextImpl<DCRTPoly>::KeyGen() {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto keys	  = context->KeyGen();

	KeyPair<DCRTPoly> keypair;
	keypair.publicKey = std::make_shared<PublicKeyImpl<DCRTPoly>>();
	keypair.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();

	keypair.publicKey->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(keys.publicKey);
	keypair.secretKey->pimpl = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(keys.secretKey);

	return keypair;
}

void CryptoContextImpl<DCRTPoly>::EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalMultKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalMultKeyGen(skImpl);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalRotateKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalRotateKeyGen(skImpl, steps);
	this->rotation_indexes.insert(this->rotation_indexes.end(), steps.begin(), steps.end());
}

// ---- Bootstrapping ----

void CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor) {

	// Only before loading one must compute the bootstrapping auxiliary data.
	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalBootstrapSetup must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

	std::vector<double> coeffchebyshev;
	int doubleAngleIts = 3;

	if (this->keyDist == fideslib::SPARSE_ENCAPSULATED) {
		// r=5 / degree-14 refit; must stay in lockstep with GetRawParams' ENCAPS branch (this copy only
		// feeds `modall`, i.e. the level plan).
		{
			coeffchebyshev = { -5.73829476916553172e-01, 2.63718536771466207e-02, -9.15574236933522245e-01,
				-3.08975417541565676e-02, 2.85601052580403802e-01, 4.83129148738224799e-03,
				-2.74350673185783482e-02, -3.16919293037672828e-04, 1.31295181914509542e-03,
				1.15825536926191591e-05, -3.79010152666429525e-05, -2.71035793019274615e-07,
				7.33952552563662122e-07, 4.41686895092293209e-09, -1.03169742989480305e-08 };
			doubleAngleIts = 5;
		}
	} else if (this->keyDist == fideslib::SPARSE_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (this->keyDist == fideslib::UNIFORM_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
	} else {
		OPENFHE_THROW("Unsupported key distribution");
	}


	// FIDESLIB_SPARSE_ARCSINE = dual-slots mode: the arcsine correction's 3 levels are reserved ONLY on
	// sparse-slot precomps (slots < N/2); the full-slot precomp stays vanilla. Must match ApproxModEval.cu.
	int arcsineLvls = 0;
	if (const char* e = std::getenv("FIDESLIB_SPARSE_ARCSINE"); e && *e && *e != '0') {
		if (slots < context->GetRingDimension() / 2)
			arcsineLvls = 3;
	}
	int32_t modall = static_cast<int>(lbcrypto::GetMultiplicativeDepthByCoeffVector(coeffchebyshev, false)) + doubleAngleIts + arcsineLvls;

	if (this->devices.empty()) {
		context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, true);
		return;
	}

	context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, true, modall);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& sk, uint32_t slots) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalBootstrapKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);

	if (this->devices.empty()) {
		context->EvalBootstrapKeyGen(skImpl, slots);
	} else {
		FIDESlib::CKKS::GenBootstrapKeys(skImpl, static_cast<int>(slots));
	}
}

// ---- Serialization ----

bool CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(std::ostream& ser, const fideslib::SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case fideslib::SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::SerializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case fideslib::SerType::JSON:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::SerializeEvalMultKey(
		  ser, lbcrypto::SerType::JSON, keyTag);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::SerializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::SerializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::JSON, keyTag);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Deserialization ----

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("DeserializeEvalMultKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::DeserializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::DeserializeEvalMultKey(
		  ser, lbcrypto::SerType::JSON);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("DeserializeEvalAutomorphismKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<FIDES_DCRTPOLY_FULL>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::JSON);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Encoding ----

// FIDESLIB_GPU_ENCODE=1: a full-slot packed plaintext is left unencoded on the host and encoded on the device at load.
static bool gpu_encode_on() {
	const char* e = std::getenv("FIDESLIB_GPU_ENCODE");
	return e && std::atoi(e) > 0;
}

// MakeCKKSPackedPlaintextInternal without its Encode(); nullptr where the device encoder does not apply.
static lbcrypto::Plaintext unencoded_ckks(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                          const std::vector<std::complex<double>>& value, size_t nsd, uint32_t level,
                                          uint32_t slots) {
	const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(cc->GetCryptoParameters());
	if (!cp || value.empty() || (cp->GetScalingTechnique() == lbcrypto::FLEXIBLEAUTOEXT && level == 0))
		return nullptr;
	auto params = cp->GetElementParams();
	if (level >= params->GetParams().size() || value.size() > params->GetRingDimension() / 2)
		return nullptr;
	if (level != 0) {
		lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> p = *params;
		for (uint32_t i = 0; i < level; i++)
			p.PopLastParam();
		params = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>(p);
	}
	auto pt = std::make_shared<lbcrypto::CKKSPackedEncoding>(params, cc->GetEncodingParams(), value, nsd, level,
	                                                         cp->GetScalingFactorReal(level), slots,
	                                                         cc->GetCKKSDataType());
	if (pt->GetSlots() != params->GetRingDimension() / 2)
		return nullptr;
	return pt;
}

// MakeCKKSPackedPlaintext's OpenFHE plaintext: under FIDESLIB_GPU_ENCODE left unencoded for the device.
template <class V>
static lbcrypto::Plaintext packed_plaintext(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, bool devices,
                                            const V& value, size_t nsd, uint32_t level, uint32_t slots,
                                            bool& device_encode) {
	device_encode = false;
	if (devices && gpu_encode_on()) {
		auto pt = unencoded_ckks(cc, std::vector<std::complex<double>>(value.begin(), value.end()), nsd, level, slots);
		if (pt) {
			device_encode = true;
			return pt;
		}
	}
	return cc->MakeCKKSPackedPlaintext(value, nsd, level, nullptr, slots);
}

// A device-encode plaintext's host polynomial, encoded on its first host use (Encrypt, the host Eval paths).
static void host_encode(Plaintext& pt) {
	if (!pt->device_encode)
		return;
	std::any_cast<lbcrypto::Plaintext&>(pt->cpu)->Encode();
	pt->device_encode = false;
	DeviceDumpForget(pt.get());
}

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
  size_t noiseScaleDeg,
  uint32_t level,
  const std::shared_ptr<void> params,
  uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	bool device_encode;
	auto pt = packed_plaintext(context, !this->devices.empty(), value, noiseScaleDeg, level, slots, device_encode);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;
	plaintext->device_encode = device_encode;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
  size_t noiseScaleDeg,
  uint32_t level,
  const std::shared_ptr<void> params,
  uint32_t slots,
  cudaStream_t stream_override) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	bool device_encode;
	auto pt = packed_plaintext(context, !this->devices.empty(), value, noiseScaleDeg, level, slots, device_encode);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu      = std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded   = false;
	plaintext->device_encode = device_encode;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext, stream_override);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	bool device_encode;
	auto pt = packed_plaintext(context, !this->devices.empty(), value, noiseScaleDeg, level, slots, device_encode);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;
	plaintext->device_encode = device_encode;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg,
                                                     uint32_t level, const std::shared_ptr<void> params,
                                                     uint32_t slots, cudaStream_t stream_override) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	bool device_encode;
	auto pt = packed_plaintext(context, !this->devices.empty(), value, noiseScaleDeg, level, slots, device_encode);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu      = std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded   = false;
	plaintext->device_encode = device_encode;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext, stream_override);

	return plaintext;
}

// ---- Encryption ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
	host_encode(pt);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

	auto ct							= context->Encrypt(pkImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (this->devices.empty() || !this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt) {
	return Encrypt(pt, pk);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	host_encode(pt);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

	auto ct							= context->Encrypt(skImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (this->devices.empty() || !this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt) {
	return Encrypt(pt, sk);
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt) {

	if (pt == nullptr) {
		OPENFHE_THROW("Plaintext pointer is null");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& ct_cpu  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

	// Copy ciphertext to CPU if needed.
	if (ct->loaded) {
		auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
		FIDESlib::CKKS::RawCipherText raw_ct;
		ct_gpu->store(raw_ct);

		// Check if CPU ciphertext has enough levels to hold GPU ciphertext.
		// A metadata-only shadow (FIDESLIB_LAZY_CPU_SHADOW) reports 0 and takes the
		// re-encrypt branch below, which builds a correctly-sized container; the level
		// GetOpenFHECipherText then derives is (totalPrimes - numRes) either way,
		// because level + numTowers is invariant for any consistent OpenFHE ciphertext.
		size_t cpu_levels = ct_cpu->GetElements().empty()
								? 0
								: ct_cpu->GetElements()[0].GetAllElements().size();
		size_t gpu_levels = raw_ct.numRes;

		if (cpu_levels < gpu_levels) {
			// Create a fresh ciphertext at the top level with enough space
			std::vector<double> dummy(1, 0.0);
			// OpenFHE's encode `level` param counts primes dropped (composite-safe): cc.L - limb.
			auto pt_dummy = context->MakeCKKSPackedPlaintext(dummy, 1, ct_gpu->cc.L - ct_gpu->getLevel());
			auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
			ct_cpu		  = context->Encrypt(skImpl, pt_dummy);
		}

		// Overwrite cpu ct with the data from GPU.
		FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
	}

	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext ptImpl;
	auto res = context->Decrypt(skImpl, ct_cpu, &ptImpl);

	if (pt->get() != nullptr) {

		if ((*pt)->loaded && !this->devices.empty()) {
			OPENFHE_THROW("Inconsistent state: Plaintext is marked as loaded but no devices are available");
		}
		if ((*pt)->loaded && !this->EvictDevicePlaintext((*pt)->gpu)) {
			OPENFHE_THROW("Plaintext eviction error: could not evict Plaintext from device");
		}

		(*pt)->cpu	  = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		(*pt)->loaded = false;
		(*pt)->gpu	  = 0;
	} else {
		*pt			  = std::make_shared<PlaintextImpl>();
		(*pt)->cpu	  = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		(*pt)->loaded = false;
		(*pt)->gpu	  = 0;
	}

	DecryptResult result{};
	result.isValid		 = res.isValid;
	result.messageLength = res.messageLength;
	return result;
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt) {
	return Decrypt(ct, sk, pt);
}

namespace {
// Pinned snapshot ring for async magnitude capture. Ciphertext::store() drains the
// whole DEVICE twice per call (~11 ms/node measured — THE capture wall); storeStaged
// is a genuinely async D2H into pinned memory. Fixed-size slots + freelist; the
// producer only enqueues copies and records an event, the decrypt worker waits the
// event and reconstructs the RawCipherText host-side.
struct MagSnap {
	FIDESlib::CKKS::StagedCtMeta meta;
	size_t		slot_off = 0;
	int			slot_idx = -1;
	cudaEvent_t ev		 = nullptr;
	bool		cpu_only = false;
	FIDESlib::CKKS::RawCipherText cpu_raw;   // fallback for non-GPU cts
};
struct MagRing {
	uint8_t*		 base = nullptr;
	size_t			 slot_bytes = 0;
	int				 nslots		= 0;
	std::vector<int> freelist;
	std::mutex		 mtx;
	std::condition_variable cv;
	cudaStream_t	 stream = nullptr;
	void init(size_t slot_bytes_) {
		if (base) return;
		const char* e  = std::getenv("FHE_MAG_RING_GB");
		const size_t gb = (e && *e && std::atoi(e) > 0) ? (size_t)std::atoi(e) : 6;
		slot_bytes		= slot_bytes_;
		nslots			= (int)std::max<size_t>(4, (gb << 30) / slot_bytes);
		cudaMallocHost(&base, (size_t)nslots * slot_bytes);
		for (int i = nslots - 1; i >= 0; --i)
			freelist.push_back(i);
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	}
	int acquire() {
		std::unique_lock<std::mutex> lk(mtx);
		cv.wait(lk, [&] { return !freelist.empty(); });
		int s = freelist.back();
		freelist.pop_back();
		return s;
	}
	void release(int s) {
		{
			std::lock_guard<std::mutex> lk(mtx);
			freelist.push_back(s);
		}
		cv.notify_one();
	}
};
MagRing g_mag_ring;
}	// namespace

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::StoreRaw(const Ciphertext<DCRTPoly>& ct) {
	auto snap = std::shared_ptr<MagSnap>(new MagSnap(), [](MagSnap* s) {
		if (s->ev) cudaEventDestroy(s->ev);
		if (s->slot_idx >= 0) g_mag_ring.release(s->slot_idx);
		delete s;
	});
	if (ct->loaded) {
		auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
		const int keep = -1;   // full snapshot
		const size_t slot_limbs = (size_t)(ct_gpu->cc.L + 1 + ct_gpu->cc.K + 1);
		g_mag_ring.init((size_t)2 * slot_limbs * ct_gpu->cc.N * sizeof(uint64_t));
		snap->slot_idx = g_mag_ring.acquire();
		snap->slot_off = (size_t)snap->slot_idx * g_mag_ring.slot_bytes;
		ct_gpu->storeStagedOrdered(g_mag_ring.base + snap->slot_off, snap->meta, g_mag_ring.stream, keep);
		cudaEventCreateWithFlags(&snap->ev, cudaEventDisableTiming);
		cudaEventRecord(snap->ev, g_mag_ring.stream);
	} else {
		snap->cpu_only	  = true;
		snap->cpu_raw.numRes = 0;   // DecryptStoredRaw refuses
	}
	return snap;
}

void CryptoContextImpl<DCRTPoly>::DecryptStoredRaw(const std::shared_ptr<void>& raw_in,
												   const PrivateKey<DCRTPoly>& sk, Plaintext* pt) {
	auto snap = std::static_pointer_cast<MagSnap>(raw_in);
	if (!snap || snap->cpu_only) {
		OPENFHE_THROW("DecryptStoredRaw: empty snapshot (ct was not GPU-resident at StoreRaw)");
	}
	// Wait for the async D2H (worker blocks; the producer never did), then rebuild the
	// RawCipherText host-side: the staged bytes are each limb's device words verbatim
	// (native width — widen u32 lanes), the same content Ciphertext::store() emits.
	cudaEventSynchronize(snap->ev);
	auto raw	= std::make_shared<FIDESlib::CKKS::RawCipherText>();
	raw->numRes = snap->meta.numRes;
	raw->N		= snap->meta.N;
	raw->Noise	= snap->meta.Noise;
	raw->NoiseLevel = snap->meta.NoiseLevel;
	raw->keyid	= snap->meta.keyid;
	raw->slots	= snap->meta.slots;
	auto widen = [&](const std::vector<size_t>& off, const std::vector<size_t>& len,
					 std::vector<std::vector<uint64_t>>& sub) {
		sub.resize(off.size());
		for (size_t i = 0; i < off.size(); ++i) {
			const uint8_t* src = g_mag_ring.base + snap->slot_off + off[i];
			if (len[i] == (size_t)snap->meta.N * sizeof(uint32_t)) {
				const auto* s32 = reinterpret_cast<const uint32_t*>(src);
				sub[i].resize(snap->meta.N);
				for (int k = 0; k < snap->meta.N; ++k)
					sub[i][k] = s32[k];
			} else {
				sub[i].assign(reinterpret_cast<const uint64_t*>(src),
							  reinterpret_cast<const uint64_t*>(src) + len[i] / sizeof(uint64_t));
			}
		}
	};
	widen(snap->meta.off0, snap->meta.len0, raw->sub_0);
	widen(snap->meta.off1, snap->meta.len1, raw->sub_1);

	// (The magnitude-only tower truncation happens at STORE time, deg-aware — see
	// StoreRaw. The snapshot already holds exactly the towers to decode.)
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);

	// Container prototypes per limb count (GetOpenFHECipherText truncates raw to the
	// container size). Encrypt once per distinct count, Clone per call — all CPU.
	static std::mutex proto_mtx;
	static std::map<int, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> protos;
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> holder;
	{
		std::lock_guard<std::mutex> lk(proto_mtx);
		auto it = protos.find(raw->numRes);
		if (it == protos.end()) {
			std::vector<double> dummy(1, 0.0);
			// OpenFHE's encode `level` counts primes dropped: total - numRes.
			const int total = (int)context->GetCryptoParameters()->GetElementParams()->GetParams().size();
			auto pt_dummy = context->MakeCKKSPackedPlaintext(dummy, 1, std::max(0, total - (int)raw->numRes));
			it = protos.emplace(raw->numRes, context->Encrypt(skImpl, pt_dummy)).first;
		}
		holder = it->second->Clone();
	}
	FIDESlib::CKKS::GetOpenFHECipherText(holder, *raw);

	lbcrypto::Plaintext ptImpl;
	context->Decrypt(skImpl, holder, &ptImpl);

	*pt			  = std::make_shared<PlaintextImpl>();
	(*pt)->cpu	  = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
	(*pt)->loaded = false;
	(*pt)->gpu	  = 0;
}

// ---- Operations ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalNegate(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalNegate(ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(-1.0);
	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalNegateInPlace(Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		context->EvalNegateInPlace(ctImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->multScalar(-1.0);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalAdd(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->add(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		host_encode(pt);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalAdd(ctImpl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);
	this->WaitPlaintextReady(pt->gpu);
	this->WaitPlaintextReady(pt->gpu);

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->addPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalAdd(ctImpl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->addScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalAddInPlace(ct1Impl, ct2Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->add(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		host_encode(pt);
		auto& ptImpl  = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		context->EvalAddInPlace(ct1Impl, ptImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadPlaintext(pt);
	this->WaitPlaintextReady(pt->gpu);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto pt_gpu	 = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->addPt(*pt_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalAddInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->addScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalAdd(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalAddInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {

	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
	}

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
		ctImpls.reserve(ciphertexts.size());
		for (const auto& ct : ciphertexts) {
			ctImpls.push_back(std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu));
		}
		auto ct							= context->EvalAddMany(ctImpls);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.

	for (const auto& ct : ciphertexts) {
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	// Initialize result with the first ciphertext.
	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ciphertexts[0]);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	const size_t inSize = ciphertexts.size();
	const size_t lim	= inSize * 2 - 2;
	std::vector<Ciphertext<DCRTPoly>> ciphertextSumVec;
	ciphertextSumVec.resize(inSize - 1);
	size_t ctrIndex = 0;

	for (size_t i = 0; i < lim; i = i + 2) {
		ciphertextSumVec[ctrIndex++] =
		  this->EvalAdd(i < inSize ? ciphertexts[i] : ciphertextSumVec[i - inSize], i + 1 < inSize ? ciphertexts[i + 1] : ciphertextSumVec[i + 1 - inSize]);
	}

	return ciphertextSumVec.back();
}

void CryptoContextImpl<DCRTPoly>::EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {

	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
		ctImpls.reserve(ciphertexts.size());
		for (const auto& ct : ciphertexts) {
			ctImpls.push_back(std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu));
		}
		context->EvalAddManyInPlace(ctImpls);
		ciphertexts[0]->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ctImpls[0]);
		return;
	}

	// GPU path.

	for (const auto& ct : ciphertexts) {
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	for (size_t j = 1; j < ciphertexts.size(); j = j * 2) {
		for (size_t i = 0; i < ciphertexts.size(); i = i + 2 * j) {
			if ((i + j) < ciphertexts.size()) {
				if (ciphertexts[i] != nullptr && ciphertexts[i + j] != nullptr) {
					this->EvalAddInPlace(ciphertexts[i], ciphertexts[i + j]);
				} else if (ciphertexts[i] == nullptr && ciphertexts[i + j] != nullptr) {
					ciphertexts[i] = ciphertexts[i + j];
				}
			}
		}
	}
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalSub(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->sub(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		host_encode(pt);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalSub(ctImpl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);
	this->WaitPlaintextReady(pt->gpu);

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->subPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		host_encode(pt);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalSub(ptImpl, ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSub(ctImpl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->addScalar(-scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSub(scalar, ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalSubInPlace(ct1Impl, ct2Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->sub(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalSubInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->addScalar(-scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalSubInPlace(scalar, ct1Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalSub(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalSub(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalSub(pt, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalSubInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalMult(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->mult(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ptImpl					= std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
		auto ct							= context->EvalMult(ct1Impl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadPlaintext(pt);
	this->WaitPlaintextReady(pt->gpu);

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, pt);
}

std::vector<Ciphertext<DCRTPoly>> CryptoContextImpl<DCRTPoly>::EvalMultPtBatch(const Ciphertext<DCRTPoly>& ct1,
                                                                               std::vector<Plaintext>& pts) {

	std::vector<Ciphertext<DCRTPoly>> results;
	results.reserve(pts.size());

	// Fall back to CPU (and to the serial path on empty input).
	if (this->devices.empty()) {
		for (auto& pt : pts)
			results.push_back(EvalMult(ct1, pt));
		return results;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	for (auto& pt : pts) {
		this->LoadPlaintext(pt);
		this->WaitPlaintextReady(pt->gpu);
	}

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));

	std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> results_gpu;
	results_gpu.reserve(pts.size());
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());

	for (auto& pt : pts) {
		Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
		results_gpu.push_back(
			std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu)));
		pts_gpu.push_back(std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu)).get());
		results.push_back(std::move(result));
	}

	FIDESlib::CKKS::MultPtBatch(results_gpu, *ct_gpu, pts_gpu);

	return results;
}

void CryptoContextImpl<DCRTPoly>::EvalMultCtAccumBatch(Ciphertext<DCRTPoly>& acc,
                                                       const std::vector<Ciphertext<DCRTPoly>>& as,
                                                       const std::vector<Ciphertext<DCRTPoly>>& bs) {

	// GPU-only: the CPU fallback would be the serial loop, which the wrapper keeps anyway.
	if (this->devices.empty()) {
		OPENFHE_THROW("EvalMultCtAccumBatch: GPU-only (serial fallback lives in the caller)");
	}

	this->LoadCiphertext(acc);
	for (auto& ct : as)
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	for (auto& ct : bs)
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	auto acc_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(acc->gpu));

	std::vector<const FIDESlib::CKKS::Ciphertext*> as_gpu, bs_gpu;
	as_gpu.reserve(as.size());
	bs_gpu.reserve(bs.size());
	std::vector<std::shared_ptr<FIDESlib::CKKS::Ciphertext>> keepalive;
	keepalive.reserve(as.size() + bs.size());
	for (auto& ct : as) {
		auto g = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
		as_gpu.push_back(g.get());
		keepalive.push_back(std::move(g));
	}
	for (auto& ct : bs) {
		auto g = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
		bs_gpu.push_back(g.get());
		keepalive.push_back(std::move(g));
	}

	acc_gpu->multAccumulateBatch(as_gpu, bs_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto ct							= context->EvalMult(ct1Impl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ptImpl  = std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
		auto res	  = context->EvalMult(ct1Impl, ptImpl);
		ct1->cpu	  = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadPlaintext(pt);
	this->WaitPlaintextReady(pt->gpu);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto pt_gpu	 = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multPt(*pt_gpu);
}

bool CryptoContextImpl<DCRTPoly>::LazyCpuShadowEnabled() {
	// Always on: fresh GPU-op outputs carry a metadata-only CPU shadow (CloneEmpty) with
	// loud-throw re-upload guards, never a deep copy.
	return true;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::MakeGpuResultLike(const Ciphertext<DCRTPoly>& src) {
	return std::make_shared<CiphertextImpl<DCRTPoly>>(*src, LazyCpuShadowEnabled());
}

void CryptoContextImpl<DCRTPoly>::CopyCiphertextDevice(Ciphertext<DCRTPoly>& dst,
													   const Ciphertext<DCRTPoly>& src) {
	if (this->devices.empty()) {
		OPENFHE_THROW("CopyCiphertextDevice: GPU-only (no devices configured)");
	}
	if (dst.get() == src.get()) {
		return;
	}

	this->LoadCiphertext(dst);
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(src));

	auto dst_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(dst->gpu));
	auto src_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(src->gpu));
	dst_gpu->copy(*src_gpu);   // c0/c1 device limbs + copyMetadata (level, noise, scale)
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalMultInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->multScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalMultInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalMult(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalMult(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalMultMutableInPlace(ct1Impl, ct2Impl);
		return;
	}

	this->LoadCiphertext(ct1);
	this->LoadCiphertext(ct2);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->mult(*ct2_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquare(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSquare(ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->square();

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSquareInPlace(Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		context->EvalSquareInPlace(ctImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->square();
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquareMutable(Ciphertext<DCRTPoly>& ct) {
	return EvalSquare(ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalRotate(ctImpl, index);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->rotate(index);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context	= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl	= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct			= context->EvalRotate(ctImpl, index);
		ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ciphertext);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	ct_gpu->rotate(index);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalConjugate(const Ciphertext<DCRTPoly>& ciphertext) {

	if (this->devices.empty()) {
		OPENFHE_THROW("EvalConjugate: CPU fallback not implemented (GPU contexts only)");
	}

	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ciphertext);
	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto src_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	res_gpu->conjugate(*src_gpu);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalMultMonomialInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t power) {
	if (this->devices.empty()) {
		OPENFHE_THROW("EvalMultMonomialInPlace: CPU fallback not implemented (GPU contexts only)");
	}
	this->LoadCiphertext(ciphertext);
	auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	gpu->multMonomial(static_cast<int>(power));
}

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		return context->EvalFastRotationPrecompute(ctImpl);
	}

	// GPU path not needed.

	return nullptr;
}

#include <core/math/hal/bigintdyn/ubintdyn.h>

Ciphertext<DCRTPoly>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<FIDES_DCRTPOLY_FULL>>(precomp);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
		return result;
	}

	// GPU path. Inefficient for only one rotation.

	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->rotate_hoisted({ (int)index }, { res_gpu.get() }, false);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<FIDES_DCRTPOLY_FULL>>(digits);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
		return result;
	}

	// GPU path. Inefficient for only one rotation.

	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->rotate_hoisted({ (int)index }, { res_gpu.get() }, true);

	return result;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) {

	std::vector<Ciphertext<DCRTPoly>> results;

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<FIDES_DCRTPOLY_FULL>>(precomp);

		for (const auto& index : indices) {
			Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
			result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
			results.push_back(result);
		}
		return results;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	// Create result ciphertexts.
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, false);
	return results;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {

	std::vector<Ciphertext<DCRTPoly>> results;

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<FIDES_DCRTPOLY_FULL>>(digits);

		for (const auto& index : indices) {
			Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
			result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
			results.push_back(result);
		}
		return results;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, true);
	return results;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct						= context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto res	  = context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
		ct->cpu		  = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
}

std::vector<double> CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree) {
	return FIDESlib::CKKS::get_chebyshev_coefficients(func, a, b, degree);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Rescale(const Ciphertext<DCRTPoly>& ciphertext) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->Rescale(ctImpl);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->rescale();

	return result;
}

void CryptoContextImpl<DCRTPoly>::RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context	= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl	= std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct			= context->Rescale(ctImpl);
		ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ciphertext);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	res_gpu->rescale();
}

void CryptoContextImpl<DCRTPoly>::DropToLevel(Ciphertext<DCRTPoly>& ciphertext, uint32_t level) {

	// Fall back to CPU (OpenFHE LevelReduce: value-preserving tower drop).
	if (this->devices.empty()) {
		auto& context	   = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl	   = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		const uint32_t cur = ctImpl->GetLevel();
		if (level > cur) {
			auto ct			= context->LevelReduce(ctImpl, nullptr, level - cur);
			ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		}
		return;
	}

	// GPU path: same tower-drop FLEXIBLEAUTO uses to align operands. The device level is
	// REMAINING depth (device=mult_depth-host_level), so convert the OpenFHE target level.
	this->LoadCiphertext(ciphertext);
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	// level counts primes dropped (OpenFHE convention, composite-safe): target limb = cc.L - level.
	ct_gpu->dropToLevel(static_cast<int>(ct_gpu->cc.L) - static_cast<int>(level));
}

void CryptoContextImpl<DCRTPoly>::SetLevel(Ciphertext<DCRTPoly>& ct, size_t level) {
	ct->SetLevel(level);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {

	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalBootstrap(ctImpl, numIterations, precision);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalBootstrap(ctImpl, numIterations, precision);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		ciphertext					= result;
		return;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));

	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride) {

	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ctImpl);

		for (int i = 0; i < log2(slots); i++) {
			int rot_idx = stride * (1 << i);
			auto tmp	= context->EvalRotate(result_ct, rot_idx);
			context->EvalAddInPlace(result_ct, tmp);
		}

		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(result_ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = this->MakeGpuResultLike(ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);

	return result;
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride) {

	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

		for (int i = 0; i < log2(slots); i++) {
			int rot_idx = stride * (1 << i);
			auto tmp	= context->EvalRotate(ctImpl, rot_idx);
			context->EvalAddInPlace(ctImpl, tmp);
		}

		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);
}

void CryptoContextImpl<DCRTPoly>::ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {

	if (this->devices.empty()) {
		OPENFHE_THROW("Not implemented for CPU path");
	}

	// GPU path.
	this->LoadCiphertext(ct);
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		this->LoadPlaintext(const_cast<Plaintext&>(pt));
		this->WaitPlaintextReady(pt->gpu);
		auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
		pts_gpu.push_back(pt_gpu.get());
	}

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::ConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, indexes, gStep);
}

void CryptoContextImpl<DCRTPoly>::SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {

	if (this->devices.empty()) {
		OPENFHE_THROW("Not implemented for CPU path");
	}

	// GPU path.
	this->LoadCiphertext(ct);
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		this->LoadPlaintext(const_cast<Plaintext&>(pt));
		this->WaitPlaintextReady(pt->gpu);
		auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
		pts_gpu.push_back(pt_gpu.get());
	}

	// Load mask
	this->LoadPlaintext(mask);
	this->WaitPlaintextReady(mask->gpu);
	auto mask_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(mask->gpu));

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::SpecialConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, *mask_gpu, stride, maskRotationStride, indexes, gStep);
}

// ---- Copy helpers ----

uint32_t CryptoContextImpl<DCRTPoly>::CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.loaded) {
		OPENFHE_THROW("Ciphertext not loaded to any device");
	}

	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto ct_gpu		  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct.gpu));
	auto new_ct		  = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu);
	new_ct->copy(*ct_gpu);
	uint32_t handle = this->RegisterDeviceCiphertext(std::move(new_ct));
	return handle;
}

// ---- Map Handling ----

uint32_t CryptoContextImpl<DCRTPoly>::next_handle() {
	// Process-wide: serialises handle allocation across the two registry mutexes so the
	// residency worker (KV reload / weight load) and the main compute thread can't collide.
	static std::mutex handle_mutex;
	std::lock_guard<std::mutex> g(handle_mutex);
	return next_gpu_handle++;
}

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDevicePlaintext(std::shared_ptr<void>&& p) {
	if (this->devices.empty()) {
		OPENFHE_THROW("No devices available to register plaintext");
	}
	device_plaintexts_mutex->lock();
	uint32_t handle = next_handle();
	device_plaintexts.emplace(handle, std::move(p));
	device_plaintexts_mutex->unlock();
	return handle;
}

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDeviceCiphertext(std::shared_ptr<void>&& c) {
	if (this->devices.empty()) {
		OPENFHE_THROW("No devices available to register ciphertext");
	}
	device_ciphertexts_mutex->lock();
	uint32_t handle = next_handle();
	device_ciphertexts.emplace(handle, std::move(c));
	device_ciphertexts_mutex->unlock();
	return handle;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDevicePlaintext(uint32_t handle) {
	device_plaintexts_mutex->lock_shared();
	auto& it = device_plaintexts.at(handle);
	device_plaintexts_mutex->unlock_shared();
	return it;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDeviceCiphertext(uint32_t handle) {
	device_ciphertexts_mutex->lock_shared();
	auto& it = device_ciphertexts.at(handle);
	device_ciphertexts_mutex->unlock_shared();
	return it;
}

bool CryptoContextImpl<DCRTPoly>::EvictDevicePlaintext(uint32_t handle) {
	{   // FIDESLIB_GPU_ENCODE: a device encode is dumped at its first eviction, so the next load reloads it
		std::lock_guard<std::mutex> g(g_dump_mutex);
		auto u = g_undumped.find(handle);
		if (u != g_undumped.end()) {
			device_plaintexts_mutex->lock();
			auto d = device_plaintexts.find(handle);
			std::shared_ptr<void> dev = d != device_plaintexts.end() ? d->second : nullptr;
			device_plaintexts_mutex->unlock();
			if (dev)
				dump_device_plaintext_locked(u->second.first, *std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(dev),
				                             u->second.second);
			if (u->second.second)
				g_cmax_free.push_back(u->second.second);
			g_undumped.erase(u);
		}
	}
	ClearPlaintextReady(handle);
	device_plaintexts_mutex->lock();
	auto result = device_plaintexts.erase(handle) > 0;
	device_plaintexts_mutex->unlock();
	return result;
}

bool CryptoContextImpl<DCRTPoly>::EvictDeviceCiphertext(uint32_t handle) {
	device_ciphertexts_mutex->lock();
	auto result = device_ciphertexts.erase(handle) > 0;
	device_ciphertexts_mutex->unlock();
	return result;
}

void CryptoContextImpl<DCRTPoly>::Synchronize() const {
	if (this->devices.empty() || !this->loaded) {
		return;
	}
	for (const auto& device : this->devices) {
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		CudaCheckErrorModNoSync;
	}
}

cudaStream_t CryptoContextImpl<DCRTPoly>::ResolvePlaintextLoadStream(cudaStream_t stream_override) const {
	if (stream_override != nullptr) {
		return stream_override;
	}
	if (!plaintext_streams_enabled) {
		return nullptr;
	}
	return plaintext_load_stream;
}

cudaStream_t CryptoContextImpl<DCRTPoly>::ResolvePlaintextComputeStream() const {
	if (!plaintext_streams_enabled) {
		return nullptr;
	}
	if (plaintext_compute_stream != nullptr) {
		return plaintext_compute_stream;
	}
	return 0;
}

void CryptoContextImpl<DCRTPoly>::RecordPlaintextReady(uint32_t handle, cudaStream_t stream) {
	if (!plaintext_streams_enabled || stream == nullptr) {
		return;
	}
	if (!plaintext_ready_events_mutex) {
		return;
	}
	plaintext_ready_events_mutex->lock();
	auto it = plaintext_ready_events.find(handle);
	if (it == plaintext_ready_events.end()) {
		cudaEvent_t ev = nullptr;
		cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
		plaintext_ready_events.emplace(handle, ev);
		it = plaintext_ready_events.find(handle);
	}
	cudaEventRecord(it->second, stream);
	plaintext_ready_events_mutex->unlock();
}

void CryptoContextImpl<DCRTPoly>::WaitPlaintextReady(uint32_t handle) {
	if (!plaintext_streams_enabled) {
		return;
	}
	if (!plaintext_ready_events_mutex) {
		return;
	}
	const cudaStream_t compute_stream = ResolvePlaintextComputeStream();
	if (compute_stream == nullptr) {
		return;
	}
	plaintext_ready_events_mutex->lock_shared();
	auto it = plaintext_ready_events.find(handle);
	if (it == plaintext_ready_events.end()) {
		plaintext_ready_events_mutex->unlock_shared();
		return;
	}
	cudaEvent_t ev = it->second;
	plaintext_ready_events_mutex->unlock_shared();
	cudaStreamWaitEvent(compute_stream, ev, 0);
}

void CryptoContextImpl<DCRTPoly>::ClearPlaintextReady(uint32_t handle) {
	if (!plaintext_ready_events_mutex) {
		return;
	}
	plaintext_ready_events_mutex->lock();
	auto it = plaintext_ready_events.find(handle);
	if (it != plaintext_ready_events.end()) {
		if (it->second != nullptr) {
			cudaEventDestroy(it->second);
		}
		plaintext_ready_events.erase(it);
	}
	plaintext_ready_events_mutex->unlock();
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	return FIDESlib::CKKS::GetConvolutionTransformRotationIndices(rowSize, bStep, stride, gStep);
}

} // namespace fideslib