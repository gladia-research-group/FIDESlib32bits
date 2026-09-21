#ifndef API_PLAINTEXT_HPP
#define API_PLAINTEXT_HPP

#include <any>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl {
  public:
	PlaintextImpl()	 = default;
	~PlaintextImpl();

	PlaintextImpl(const CryptoContext<DCRTPoly>&& context);

	// ---- Copy ----

	PlaintextImpl(const PlaintextImpl&);
	PlaintextImpl(const Plaintext&);
	PlaintextImpl& operator=(const PlaintextImpl&) = delete;
	PlaintextImpl& operator=(const Plaintext&)	   = delete;

	// ---- Move ----

	PlaintextImpl(PlaintextImpl&&)			  = delete;
	PlaintextImpl(Plaintext&&)				  = delete;
	PlaintextImpl& operator=(PlaintextImpl&&) = delete;
	PlaintextImpl& operator=(Plaintext&&)	  = delete;

	// ---- Functions ----

	void SetLength(size_t length);
	void SetSlots(uint32_t slots);

	double GetLogPrecision() const;
	uint32_t GetLevel() const;
	std::vector<std::complex<double>> GetCKKSPackedValue() const;
	std::vector<double> GetRealPackedValue() const;

	// ---- Friend Operators ----

	friend std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt);

	// ---- Internal State ----

	std::any cpu;
	uint32_t gpu = 0;
	/// @brief Flag indicating whether the plaintext is loaded to the devices.
	bool loaded = false;
	/// @brief COEFF-mode weight plaintext (MarkCoeffStaged): the CPU side holds a single q0 EVAL
	/// limb; device loads expand it to the target level's limbs on the GPU (loadCoeffExpand).
	/// Must be consumed through the staged path — a plain load of a marked plaintext throws.
	bool coeff_staged = false;
	/// @brief COEFF-mode prescale (MarkCoeffStaged): host values were encoded ÷2^k so the
	/// centered lift bound holds; the GPU expand multiplies the lifted limbs back by
	/// (2^k mod q_i) — exact integer un-prescale, no scale/level change. 0 = none.
	int coeff_prescale_log2 = 0;
	/// @brief Parent context for device management.
	CryptoContext<DCRTPoly> parent_context;
};

// ---- Override Operators ----

std::ostream& operator<<(std::ostream& os, const Plaintext& pt);
} // namespace fideslib

#endif // API_PLAINTEXT_HPP