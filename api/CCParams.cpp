#include "CCParams.hpp"
#include "lattice/constants-lattice.h"

#include <openfhe.h>

#include <cassert>

namespace fideslib {

CCParams<CryptoContextCKKSRNS>::CCParams() {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
	this->cpu = std::make_any<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>>(std::move(params));
}

// ---- CKKS Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetMultiplicativeDepth(uint32_t depth) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetMultiplicativeDepth(depth);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetScalingModSize(size);
}

namespace detail {
template <typename P, typename = void> struct has_per_level_sizes : std::false_type {};
template <typename P>
struct has_per_level_sizes<P, std::void_t<decltype(std::declval<P&>().SetScalingModSizePerLevel(
								  std::declval<std::vector<uint32_t>>()))>> : std::true_type {};
} // namespace detail

void CCParams<CryptoContextCKKSRNS>::SetCKKSDataTypeComplex() {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetCKKSDataType(lbcrypto::COMPLEX);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingModSizePerLevel(std::vector<uint32_t> sizes) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	if constexpr (detail::has_per_level_sizes<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>>::value) {
		params.SetScalingModSizePerLevel(std::move(sizes));
	} else {
		if (!sizes.empty())
			throw std::runtime_error("SetScalingModSizePerLevel: installed OpenFHE lacks mixed-chain support");
	}
}

void CCParams<CryptoContextCKKSRNS>::SetBatchSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetBatchSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetRingDim(uint32_t dim) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetRingDim(dim);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingTechnique(ScalingTechnique tech) {
	auto& params	   = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	auto scale_openfhe = static_cast<lbcrypto::ScalingTechnique>(tech);
	assert((int)scale_openfhe == (int)tech);
	params.SetScalingTechnique(scale_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetCompositeDegree(uint32_t degree) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetCompositeDegree(degree);
}

void CCParams<CryptoContextCKKSRNS>::SetRegisterWordSize(uint32_t bits) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetRegisterWordSize(bits);
}

void CCParams<CryptoContextCKKSRNS>::SetNumLargeDigits(uint32_t numDigits) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetNumLargeDigits(numDigits);
}

void CCParams<CryptoContextCKKSRNS>::SetFirstModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetFirstModSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetDigitSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	params.SetDigitSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetKeySwitchTechnique(KeySwitchTechnique tech) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	auto ks_openfhe = static_cast<lbcrypto::KeySwitchTechnique>(tech);
	assert((int)ks_openfhe == (int)tech);
	params.SetKeySwitchTechnique(ks_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetSecretKeyDist(SecretKeyDist dist) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	
	if (this->devices.empty()) {
		if (dist == SecretKeyDist::SPARSE_TERNARY) {
			params.SetSecretKeyDist(lbcrypto::SPARSE_TERNARY);
		}
		else {
			params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
		}
	}
	else {
		params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	}
	keyDist = dist;
}

void CCParams<CryptoContextCKKSRNS>::SetSecurityLevel(SecurityLevel level) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	auto sl_openfhe = static_cast<lbcrypto::SecurityLevel>(level);
	assert((int)sl_openfhe == (int)level);
	params.SetSecurityLevel(sl_openfhe);
}

// ---- Getters ----

SecretKeyDist CCParams<CryptoContextCKKSRNS>::GetSecretKeyDist() const {
	auto& params	 = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	auto skd_openfhe = params.GetSecretKeyDist();
	return static_cast<SecretKeyDist>(skd_openfhe);
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetMultiplicativeDepth() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	return params.GetMultiplicativeDepth();
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetBatchSize() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(cpu);
	return params.GetBatchSize();
}

// ---- Device Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetDevices(std::vector<int>&& devices) {
	this->devices = std::move(devices);
}

void CCParams<CryptoContextCKKSRNS>::SetPlaintextAutoload(bool autoload) {
	this->plaintextAutoload = autoload;
}

void CCParams<CryptoContextCKKSRNS>::SetCiphertextAutoload(bool autoload) {
	this->ciphertextAutoload = autoload;
}

} // namespace fideslib