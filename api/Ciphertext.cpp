#include "Ciphertext.hpp"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"  // full ContextData: the level<->limb conversions read cc.L
#include "Definitions.hpp"

#include <iostream>
#include <openfhe.h>

namespace fideslib {

CiphertextImpl<DCRTPoly>::~CiphertextImpl() {
	if (this->loaded && this->gpu != 0 && this->parent_context) {
		this->parent_context->EvictDeviceCiphertext(this->gpu);
		this->gpu = 0;
	}
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

// ---- Copy ----

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CiphertextImpl<DCRTPoly>& other)
	: CiphertextImpl<DCRTPoly>(other, /*lazy_cpu_shadow=*/false) {
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CiphertextImpl<DCRTPoly>& other, bool lazy_cpu_shadow) {

	auto const& other_cpu = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(other.cpu);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpu_copy =
		lazy_cpu_shadow ? other_cpu->CloneEmpty()
						: std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*other_cpu);
	this->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(cpu_copy));

	// Copy underlying GPU Ciphertext if loaded.
	this->loaded = other.loaded;
	if (this->loaded) {
		this->gpu = other.parent_context->CopyDeviceCiphertext(other);
	} else {
		this->gpu = 0;
	}

	// Copy parent context.
	this->parent_context = other.parent_context;
	this->original_level = other.original_level;
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const Ciphertext<DCRTPoly>& other) : CiphertextImpl<DCRTPoly>(static_cast<const CiphertextImpl<DCRTPoly>&>(other)) {
}

// ---- Clone ----

Ciphertext<DCRTPoly> CiphertextImpl<DCRTPoly>::Clone() const {
	// GPU-resident ciphertexts clone with the metadata-only CPU shadow (the BAKED-ON
	// lazy-shadow discipline; CloneEmpty carries all scalar value-metadata — slots,
	// level, noiseScaleDeg, scalingFactor — so only the ELEMENTS stay lazy). The old
	// deep host copy made clone ~40x its device cost (788 us wrapper vs 18 us lazy;
	// the delta was host DCRTPoly memcpy). A ciphertext that is NOT loaded lives on
	// the CPU: the deep copy remains the only correct behavior there.
	Ciphertext<DCRTPoly> clone = std::make_shared<CiphertextImpl<DCRTPoly>>(*this, /*lazy_cpu_shadow=*/this->loaded);
	return clone;
}

// ---- Getters ----

size_t CiphertextImpl<DCRTPoly>::GetLevel() const {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		return ct->GetLevel();
	}

	// GPU path. FIDESlib's level is the TOP LIMB INDEX; OpenFHE's ciphertext level counts
	// PRIMES DROPPED (also under COMPOSITESCALING, where one CKKS level = d primes). The
	// exact conversion at every composite degree is against the chain's top limb index
	// cc.L, not multiplicative_depth (they coincide only on classic FLEXIBLEAUTO chains).
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	return static_cast<size_t>(ct_gpu->cc.L - ct_gpu->getLevel());
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDeg() const {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		return ct->GetNoiseScaleDeg();
	}

	// GPU path.
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	return ct_gpu->NoiseLevel;
}

// ---- Setters ----

void CiphertextImpl<DCRTPoly>::SetSlots(size_t slots) {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		ct->SetSlots(slots);
		return;
	}
	// GPU path.
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	ct_gpu->slots = static_cast<int>(slots);
}

void CiphertextImpl<DCRTPoly>::SetLevel(size_t level) {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);

		// A lazy CPU shadow (see the lazy_cpu_shadow ctor) has no elements to drop.
		// Fail loudly rather than index an empty vector: this can only be reached if a
		// lazily-shadowed ciphertext was evicted from the device while still live.
		if (ct->GetElements().empty()) {
			OPENFHE_THROW("SetLevel: ciphertext is unloaded and has a metadata-only CPU shadow");
		}

		size_t currentTowers = ct->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = ct->GetLevel();

		size_t totalPrimes	= currentTowers + currentLevel;
		size_t targetTowers = totalPrimes - level;

		if (currentTowers > targetTowers) {
			// Need to drop towers
			size_t towersToDrop = currentTowers - targetTowers;

			auto& elements = ct->GetElements();
			for (auto& elem : elements) {
				elem.DropLastElements(towersToDrop);
			}
		}

		ct->SetLevel(level);

		return;
	}

	// GPU path. level counts primes dropped (OpenFHE convention, composite-safe);
	// target limb index = cc.L - level.
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	ct_gpu->dropToLevel(static_cast<int>(ct_gpu->cc.L) - static_cast<int>(level));
}

// ---- Operators ----

Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs) {
	if (lhs->parent_context.get() != rhs->parent_context.get()) {
		OPENFHE_THROW("Cannot add ciphertexts from different contexts");
	}

	return lhs->parent_context->EvalAdd(lhs, rhs);
}

} // namespace fideslib
