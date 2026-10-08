//
// Created by carlosad on 12/11/24.
//

#include <immintrin.h>
#include <atomic>
#include "CKKS/Lockstep.cuh"
#include <exception>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "CKKS/Discard.cuh"  // fusedRescaleFlag (lever A)
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"
#include <atomic>
#include <cstdlib>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

constexpr bool PRINT = false;

using namespace FIDESlib::CKKS;

#include "CKKS/Bootstrap.cuh"
// Stage-divergence harness: stash checkpoints INSIDE the EvalMod segment (between post-CtS and
// pre-StC, which the outer probes cannot attribute). Inert unless a caller installed the stash.
static void emStageProbe(const char* stage, FIDESlib::CKKS::Ciphertext& ctxt) {
    if (FIDESlib::CKKS::g_btsStageStash) {
        cudaDeviceSynchronize();
        auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
        c->copy(ctxt);
        FIDESlib::CKKS::g_btsStageStash->emplace_back(stage, std::move(c));
    }
}

void evalChebyshevSeries(Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, std::vector<double>& coefficients, double lower_bound, double upper_bound);
void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval);

// Arcsine correction: u = asin(2*pi*y)/(2*pi) ~= y*(1 + (2*pi)^2/6 * y^2) cancels the EvalMod
// sine-linearization cubic. Off by default; setArcsineOverride(1/0) forces it on/off per region
// (-1 restores the default). The context must have the 3 levels reserved (FIDESLIB_SPARSE_ARCSINE).
static std::atomic<int>& arcsineOverride() {
	static std::atomic<int> v{ -1 };
	return v;
}
void FIDESlib::CKKS::setArcsineOverride(int v) {
	arcsineOverride().store(v, std::memory_order_relaxed);
}
static bool arcsineEnabled() {
	const int o = arcsineOverride().load(std::memory_order_relaxed);
	return o > 0;
}

// FIDESLIB_SPARSE_ARCSINE = dual-slots mode: arcsine reservation+correction live ONLY on
// sparse-slot precomps (approxModReductionSparse); the dense path never applies it. A reserve
// without consume (or the reverse) is fatal, so the two must agree.
static bool sparseArcsineMode() {
	static const bool v = [] {
		const char* e = std::getenv("FIDESLIB_SPARSE_ARCSINE");
		return e && *e && *e != '0';
	}();
	return v;
}


static double arcsineC3() {
	return (2.0 * M_PI) * (2.0 * M_PI) / 6.0;
}

static void applyArcsineCorrection(Ciphertext& y) {
	const int lvl_in = static_cast<int>(y.getLevel());
	Ciphertext t(y.cc_);
	t.square(y, false);
	t.multScalar(arcsineC3(), false);
	t.addScalar(1.0);
	y.mult(t, false);
	static const bool logged = [&] {
		std::cout << "[arcsine] engaged C3=" << arcsineC3() << " level " << lvl_in << "->" << y.getLevel()
				  << std::endl;
		return true;
	}();
	(void)logged;
}

// ---- FIDESLIB_EVALMOD_LOCKSTEP (CKKS/Lockstep.cuh) -------------------------------------------------------------------
namespace {
struct Lockstep {
	// spin handoff: a condition-variable wake costs ~20 us per turn (52 turns per bootstrap), more than the L2 gain
	std::atomic<int> turn{0};  // 0 = the Re thread may issue, 1 = the Im thread
	std::atomic<bool> done[2] = {false, false};
	void pass(int to) { turn.store(to, std::memory_order_release); }
	void wait(int me) {
		while (turn.load(std::memory_order_acquire) != me && !done[1 - me].load(std::memory_order_acquire))
			_mm_pause();
	}
	void finish(int me) {
		done[me].store(true, std::memory_order_release);
		turn.store(1 - me, std::memory_order_release);
	}
};
thread_local Lockstep* tlLock = nullptr;
thread_local int tlSide = 0;
bool lockstepEnabled() {  // read per call: a test toggles it in-process for the bit-exact gate
	const char* e = std::getenv("FIDESLIB_EVALMOD_LOCKSTEP");
	return e && std::atoi(e) > 0;
}
}  // namespace
namespace {
thread_local int tlDepth = 0;  // nested relinearizing ops (mult -> adjust -> mult) count once
}
FIDESlib::CKKS::LockstepGuard::LockstepGuard() {
	if (tlDepth++ == 0)
		lockstepBeforeKeySwitch();
}
FIDESlib::CKKS::LockstepGuard::~LockstepGuard() {
	if (--tlDepth == 0)
		lockstepAfterKeySwitch();
}
// Re thread: hand the turn to Im before each key switch and wait until Im has issued the same one.
void FIDESlib::CKKS::lockstepBeforeKeySwitch() {
	if (tlLock && tlSide == 0) {
		tlLock->pass(1);
		tlLock->wait(0);
	}
}
// Im thread: right after issuing a key switch, hand the turn back so Re issues the same key switch next.
void FIDESlib::CKKS::lockstepAfterKeySwitch() {
	if (tlLock && tlSide == 1) {
		tlLock->pass(0);
		tlLock->wait(1);
	}
}

void FIDESlib::CKKS::approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });

	// cudaDeviceSynchronize();
	if constexpr (PRINT)
		std::cout << "Approx mod red start " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;

	bool constexpr COMPLEX = true;
	ContextData& cc		   = ctxtEnc.cc;

	emStageProbe("EM-in-Re", ctxtEnc);
	emStageProbe("EM-in-Im", ctxtEncI);
	if (COMPLEX && lockstepEnabled() && !FIDESlib::CKKS::g_btsStageStash && !tlLock) {
		// both halves (Chebyshev + double angle) in lockstep; identical op sequences, so the key switches pair up
		Lockstep L;
		const int dev = ctxtEnc.c0.GPU.at(0).device;
		std::exception_ptr err;
		std::thread im([&] {
			cudaSetDevice(dev);
			tlLock = &L;
			tlSide = 1;
			L.wait(1);
			try {
				evalChebyshevSeries(ctxtEncI, cc.GetCoeffsChebyshev(), -1.0, 1.0);
				applyDoubleAngleIterations(ctxtEncI, cc.GetDoubleAngleIts(), keySwitchingKey);
			} catch (...) {
				err = std::current_exception();
			}
			tlLock = nullptr;
			L.finish(1);
		});
		tlLock = &L;
		tlSide = 0;
		evalChebyshevSeries(ctxtEnc, cc.GetCoeffsChebyshev(), -1.0, 1.0);
		applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
		tlLock = nullptr;
		L.finish(0);
		im.join();
		if (err)
			std::rethrow_exception(err);
		goto after_da;
	}
	if constexpr (COMPLEX)
		evalChebyshevSeries(ctxtEncI, cc.GetCoeffsChebyshev(), -1.0, 1.0);
	evalChebyshevSeries(ctxtEnc, cc.GetCoeffsChebyshev(), -1.0, 1.0);
	emStageProbe("EM-cheby-Re", ctxtEnc);
	emStageProbe("EM-cheby-Im", ctxtEncI);
	if constexpr (PRINT) {
		std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;

		std::cout << "ctxtEncI res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
		for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}

	applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
	if constexpr (COMPLEX)
		applyDoubleAngleIterations(ctxtEncI, cc.GetDoubleAngleIts(), keySwitchingKey);
after_da:
	emStageProbe("EM-DA-Re", ctxtEnc);
	emStageProbe("EM-DA-Im", ctxtEncI);
	if (!sparseArcsineMode() && arcsineEnabled()) {
		applyArcsineCorrection(ctxtEnc);
		if constexpr (COMPLEX)
			applyArcsineCorrection(ctxtEncI);
	}
	if constexpr (PRINT) {
		std::cout << "ctxtEnc DA res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;

		std::cout << "ctxtEncI DA res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
		for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	// cudaDeviceSynchronize();
	if constexpr (COMPLEX)
		ctxtEncI.multMonomial(cc.N / 2);
	emStageProbe("EM-mono-Im", ctxtEncI);
	// cudaDeviceSynchronize();
	if constexpr (COMPLEX)
		ctxtEnc.add(ctxtEncI);
	emStageProbe("EM-comb", ctxtEnc);
	if constexpr (!COMPLEX)
		ctxtEnc.add(ctxtEnc);
	// cudaDeviceSynchronize();
	if (post != 1)
		multIntScalar(ctxtEnc, post);
	if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
		ctxtEnc.rescale();
	// cudaDeviceSynchronize();
	if constexpr (PRINT) {
		std::cout << "ctxtEnc final res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
}

void FIDESlib::CKKS::approxModReductionReal(Ciphertext& ctxtEnc, const KeySwitchingKey& keySwitchingKey, uint64_t post) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	ContextData& cc = ctxtEnc.cc;
	evalChebyshevSeries(ctxtEnc, cc.GetCoeffsChebyshev(), -1.0, 1.0);
	applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
	if (!sparseArcsineMode() && arcsineEnabled())
		applyArcsineCorrection(ctxtEnc);
	if (post != 1)
		multIntScalar(ctxtEnc, post);
	if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
		ctxtEnc.rescale();
}

void FIDESlib::CKKS::approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post) {
	approxModReductionSparse(ctxtEnc, post, ctxtEnc.cc.GetCoeffsChebyshev(), ctxtEnc.cc.GetDoubleAngleIts());
}

void FIDESlib::CKKS::approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post, std::vector<double>& coefficients,
                                               int daIts) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	ContextData& cc = ctxtEnc.cc;

	KeySwitchingKey& keySwitchingKey = cc.GetEvalKey(ctxtEnc.keyID);

	evalChebyshevSeries(ctxtEnc, coefficients, (double)-1.0, (double)1.0);

	if constexpr (PRINT) {
		std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	applyDoubleAngleIterations(ctxtEnc, daIts, keySwitchingKey);
	// dual-slots mode: this precomp carries the +3 reservation — the
	// correction MUST run unconditionally (reserve-without-consume is fatal)
	if (sparseArcsineMode() || arcsineEnabled())
		applyArcsineCorrection(ctxtEnc);
	if constexpr (PRINT) {
		std::cout << "ctxtEnc DA " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	if (post != 1)
		multIntScalar(ctxtEnc, post);
	if constexpr (PRINT) {
		std::cout << "ctxtEnc final " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
		for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
			cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	// cudaDeviceSynchronize();
}

void FIDESlib::CKKS::multIntScalar(Ciphertext& ctxt, uint64_t op) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	// n32: the scalar is consumed PER PRIME by the mult kernels, which require operands
	// < prime. On 27-28-bit chains a raw 64-bit scalar (e.g. the bootstrap corFactor)
	// can exceed that — reduce per prime instead of broadcasting the raw value.
	std::vector<uint64_t> op_(ctxt.getLevel() + 1);
	for (size_t i = 0; i < op_.size(); ++i)
		op_[i] = op % ctxt.cc.prime[i].p;
	ctxt.c0.multScalar(op_);
	ctxt.c1.multScalar(op_);
}

void innerEvalChebyshevPS(const Ciphertext& ctxt,
  Ciphertext& out,
  const std::vector<double>& coefficients,
  const uint32_t k,
  uint32_t m,
  const std::vector<Ciphertext*>& T,
  const std::vector<Ciphertext*>& T2,
  int level_offset = 0,
  int max_m		   = 1000) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
	/*
Ciphertext<DCRTPoly> AdvancedSHECKKSRNS::InnerEvalChebyshevPS(ConstCiphertext<DCRTPoly> x,
															  const std::vector<double>& coefficients, uint32_t k,
															  uint32_t m, std::vector<Ciphertext<DCRTPoly>>& T,
															  std::vector<Ciphertext<DCRTPoly>>& T2) const {
*/
	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;

	/// Left AS IS ///
	// Compute k*2^{m-1}-k because we use it a lot
	uint32_t k2m2k = k * (1 << (m - 1)) - k;
	// Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
	auto f2 = coefficients;
	f2.resize(2 * k2m2k + k + 1, 0.0);
	if (f2.size() > coefficients.size())
		f2.back() = 1;
	// Divide f2 by T^{k*2^{m-1}}
	std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
	Tkm.back() = 1;
	auto divqr = lbcrypto::LongDivisionChebyshev(f2, Tkm);

	// Subtract x^{k(2^{m-1} - 1)} from r
	std::vector<double> r2 = divqr->r;
	if (int32_t(k2m2k - lbcrypto::Degree(divqr->r)) <= 0) {
		r2[int32_t(k2m2k)] -= 1;
		r2.resize(lbcrypto::Degree(r2) + 1);
	} else {
		r2.resize(int32_t(k2m2k + 1), 0.0);
		r2.back() = -1;
	}

	// Divide r2 by q
	auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

	// Add x^{k(2^{m-1} - 1)} to s
	std::vector<double> s2 = divcs->r;
	s2.resize(int32_t(k2m2k + 1), 0.0);
	s2.back() = 1;

	/// Left AS IS ///

	if constexpr (true) {
		// Evaluate c at u
		Ciphertext& cu = out;
		uint32_t dc	   = lbcrypto::Degree(divcs->q);
		bool flag_c	   = false;
		if (dc >= 1) {
			if (dc == 1) {
				if (divcs->q[1] != 1) {
					cu.multScalar(*T[0], divcs->q[1], true);
				} else {
					cu.copy(*T[0]);
				}
			} else {
				// std::vector<Ciphertext*>& ctxs = T;
				std::vector<double> weights(dc);

				for (uint32_t i = 0; i < dc; i++) {
					weights[i] = divcs->q[i + 1];
				}

				{
					// composite: deg-alignment (+1 level) and the PS recursion offset are
					// LEVEL counts -> d limbs each.
					const int d_ = T2[m - 1]->cc.compositeDegree();
					cu.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - level_offset * d_);
					cu.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - level_offset * d_);
				}

				cu.evalLinearWSumMutable(dc, T, weights);
			}

			cu.addScalar(divcs->q.front() / 2);

			if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL) {
				cu.rescale();
			}
			// cu.dropToLevel(T2[m - 1]->getLevel() + cu.NoiseLevel - 1);

			flag_c = true;
		}

		Ciphertext qu(cc_);
		// Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
		if (lbcrypto::Degree(divqr->q) > k) {
			assert(m > 2);
			innerEvalChebyshevPS(ctxt, qu, divqr->q, k, m - 1, T, T2, level_offset, max_m);

			if (qu.NoiseLevel == 2)
				qu.rescale();

		} else {
			// dq = k from construction
			// perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
			auto qcopy = divqr->q;
			qcopy.resize(k);
			if (lbcrypto::Degree(qcopy) > 0) {
				std::vector<double> weights; //(/*lbcrypto::Degree(qcopy)*/ k - 1);

				std::vector<Ciphertext*> ctxs; // T;
				for (uint32_t i = 0; i < divqr->q.size() - 1 /*lbcrypto::Degree(qcopy)*/; i++) {
					if (divqr->q[i + 1] != 0) {
						weights.push_back(divqr->q[i + 1]);
						ctxs.push_back(T[i]);
					}
				}

				{
					const int d_ = T2[m - 1]->cc.compositeDegree();
					qu.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - level_offset * d_);
					qu.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - level_offset * d_);
				}
				// qu.growToLevel(T[k - 1]->getLevel() + (T[k - 1]->NoiseLevel == 1 ? 1 : 0));

				qu.evalLinearWSumMutable(/*bcrypto::Degree(qcopy)*/ ctxs.size(), ctxs, weights);
				// the highest order coefficient will always be 2 after one division because of the Chebyshev division rule

				qu.addScalar(divqr->q.front() / 2);

				if (T[k - 1]->NoiseLevel == 1)
					qu.rescale();
				/*
			sum.add(T[k - 1], T[k - 1]);
			qu.add(sum);
			 */
				/*
				if (divqr->q.back() == 2.0) {
					qu.add(*T[k - 1]);
					qu.add(*T[k - 1]);
				} else {
					Ciphertext sum(cc_);
					sum.copy(*T[k - 1]);
					if (divqr->q.back() > 0 && divqr->q.back() - round(divqr->q.back()) == 0.0) {
						multIntScalar(sum, (uint64_t)divqr->q.back());
					} else {
						__builtin_unreachable();
					}
					// adds the free term (at x^0)
					qu.add(sum);
				}
				*/

				if (T[k - 1]->NoiseLevel == 2)
					qu.rescale();

			} else {
				qu.copy(*T[k - 1]);

				if (divqr->q.back() > 0 && divqr->q.back() - round(divqr->q.back()) == 0.0) {
					multIntScalar(qu, (uint64_t)divqr->q.back());
				} else {
					__builtin_unreachable();
				}
				// adds the free term (at x^0)
				qu.addScalar(divqr->q.front() / 2);
				if (qu.NoiseLevel == 2)
					qu.rescale();
			}

			// adds the free term (at x^0)

			// The number of levels of qu is the same as the number of levels of T[k-1] + 1.
			// Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.
		}
		Ciphertext su(cc_);
		if (lbcrypto::Degree(s2) > k) {
			assert(m > 2);
			innerEvalChebyshevPS(ctxt, su, s2, k, m - 1, T, T2, level_offset + 1, max_m);
		} else {
			// ds = k from construction
			// perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
			auto scopy = s2;
			scopy.resize(k);
			if (lbcrypto::Degree(scopy) > 0) {
				std::vector<Ciphertext*> ctxs;
				std::vector<double> weights; //(lbcrypto::Degree(scopy));

				for (uint32_t i = 0; i < /*lbcrypto::Degree(scopy)*/ s2.size() - 1; i++) {
					if (s2[i + 1] != 0) {
						ctxs.emplace_back(T[i]);
						weights.push_back(s2[i + 1]);
					}
				}

				{
					const int d_ = T2[m - 1]->cc.compositeDegree();
					su.growToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - d_ - level_offset * d_);
					su.dropToLevel(T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1 ? d_ : 0) - d_ - level_offset * d_);
				}

				su.evalLinearWSumMutable(/*lbcrypto::Degree(scopy)*/ ctxs.size(), T, weights);
				// adds the free term (at x^0)
				su.addScalar(s2.front() / 2);

				// if (T[k - 1]->NoiseLevel == 1)
				//     su.rescale();
				//  the highest order coefficient will always be 1 because s2 is monic.
				assert(s2.back() == 1.0);
				// su.add(*T[k - 1]);

			} else {
				su.copy(*T[k - 1]);
				// adds the free term (at x^0)
				su.addScalar(s2.front() / 2);
			}

			// The number of levels of su is the same as the number of levels of T[k-1] + 1.
			// Will only get here when m = 2, so need to reduce the number of levels by 1.
			// if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL && su.NoiseLevel == 2)
			//    su.dropToLevel(su.getLevel() - 1);
		}

		if (flag_c) {
			if (max_m - m <= 1)
				T2[m - 1]->adjustForAddOrSub(
				  cu); // For m > 3, the required levels for the recursive cu component are not strictly decreasing, caching is needed, for which the benefit is uncertain
			if (T2[m - 1]->NoiseLevel == 1 && cu.NoiseLevel == 2)
				cu.rescale();
			cu.add(*T2[m - 1]);
		} else {
			cu.addScalar(*T2[m - 1], divcs->q.front() / 2);
		}
		if (cc.rescaleTechnique == FIXEDMANUAL && out.NoiseLevel == 2)
			cu.rescale();
		cu.mult(qu, false);
		cu.add(su); // cu aliases out
	}
}

/**
 * Adaptation of OpenFHE's implementation.
 */
void FIDESlib::CKKS::evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound, double upper_bound) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
	/*
	Ciphertext<DCRTPoly> AdvancedSHECKKSRNS::EvalChebyshevSeriesPS(ConstCiphertext<DCRTPoly> x,
const std::vector<double>& coefficients, double a, double b) const {
	*/
	constexpr bool sync = false;

	uint32_t n			   = lbcrypto::Degree(coefficients);
	std::vector<double> f2 = coefficients;
	f2.resize(n + 1);
	/*
	 uint32_t n = Degree(coefficients);
	std::vector<double> f2 = coefficients;
	// Make sure the coefficients do not have the zero dominant terms
	if (coefficients[coefficients.size() - 1] == 0)
		f2.resize(n + 1);
	*/

	std::vector<uint32_t> degs = lbcrypto::ComputeDegreesPS(n);
	uint32_t k				   = degs[0];
	uint32_t m				   = degs[1];
	if (false) {
		if (n <= 36) {
			k = 12;
			m = 2;
		} else if (n <= 92) {
			k = 13;
			m = 3;
		}
	}
	static const bool ps_logged = [&] {
		std::cout << "[cheb_ps] n=" << n << " k=" << k << " m=" << m << std::endl;
		return true;
	}();
	(void)ps_logged;
	/*
	std::vector<uint32_t> degs = ComputeDegreesPS(n);
	uint32_t k                 = degs[0];
	uint32_t m                 = degs[1];
	// std::cerr << "\n Degree: n = " << n << ", k = " << k << ", m = " << m << std::endl;
	*/
	// assert((lower_bound - std::round(lower_bound) < 1e-10) && (upper_bound - std::round(upper_bound) < 1e-10) &&
	//        (std::round(lower_bound) == -1) && (std::round(upper_bound) == 1));

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;
	/*
	std::vector<Ciphertext> T_;
	T_.emplace_back(cc);
	for (uint32_t i = 1; i < k; ++i)
		T_.emplace_back(cc);
	std::vector<Ciphertext> T2_;
	T2_.emplace_back(cc);
	for (uint32_t i = 1; i < m; i++) {
		T2_.emplace_back(cc);
	}
*/
	std::vector<Ciphertext> aux;
	for (size_t i = aux.size(); i < k + m; i++) {
		aux.emplace_back(cc_);
	}

	std::vector<Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i] = &aux[i];
	std::vector<Ciphertext*> T2(m);
	for (uint32_t i = 0; i < m; i++)
		T2[i] = &aux[i + k];
	/*
	std::vector<Ciphertext*> T(k);
	for (uint32_t i = 0; i < k; ++i)
		T[i] = &T_[i];
	std::vector<Ciphertext*> T2(m);
	for (uint32_t i = 0; i < m; i++)
		T2[i] = &T2_[i];
*/
	T[0]->copy(ctxt);
	/*
	// computes linear transformation y = -1 + 2 (x-a)/(b-a)
	// consumes one level when a <> -1 && b <> 1
	auto cc = x->GetCryptoContext();
	std::vector<Ciphertext<DCRTPoly>> T(k);
	if ((a - std::round(a) < 1e-10) && (b - std::round(b) < 1e-10) && (std::round(a) == -1) && (std::round(b) == 1)) {
		// no linear transformation is needed if a = -1, b = 1
		// T_1(y) = y
		T[0] = x->Clone();
	}
	else {
		// linear transformation is needed
		double alpha = 2 / (b - a);
		double beta  = 2 * a / (b - a);

		T[0] = cc->EvalMult(x, alpha);
		cc->ModReduceInPlace(T[0]);
		cc->EvalAddInPlace(T[0], -1.0 - beta);
	}
	*/
	// Ciphertext y(cc);
	// y.copy(T[0]);

	// if (ctxt.NoiseLevel == 1)
	//     ctxt.multScalar(1.0);
	if (T[0]->NoiseLevel == 2)
		T[0]->rescale();
	for (uint32_t i = 2; i <= k; i++) {
		// if i is a power of two
		if constexpr (sync)
			cudaDeviceSynchronize();

		if (i % 2 == 1) {
			// if i is odd
			// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
			T[i / 2]->adjustForMult(*T[i / 2 - 1]);
			T[i / 2 - 1]->adjustForMult(*T[i / 2]);
			T[i - 1]->mult(*T[i / 2 - 1], *T[i / 2], fusedRescaleFlag() != 0);
			T[i - 1]->add(*T[i - 1]);
			ctxt.adjustForAddOrSub(*T[i - 1]);
			if (ctxt.NoiseLevel == 1 && T[i - 1]->NoiseLevel == 2)
				T[i - 1]->rescale();
			T[i - 1]->sub(ctxt);
			// if (ctxt.NoiseLevel == 2)
			//     T[i - 1]->rescale();
		} else {
			// i is even
			// compute T_{2i}(y) = 2*T_i(y)^2 - 1
			T[i / 2 - 1]->adjustForMult(*T[i / 2 - 1]);
			T[i - 1]->square(*T[i / 2 - 1], fusedRescaleFlag() != 0);
			T[i - 1]->add(*T[i - 1]);
			T[i - 1]->addScalar(-1.0);
			// T[i - 1]->rescale();
		}
	}

	if constexpr (PRINT) {
		for (size_t j = 0; j < k; ++j) {
			std::cout << "T[" << j << "]: " << std::endl;
			for (auto& i : T[j]->c0.GPU.at(0).limb) {
				cudaSetDevice(T[j]->c0.GPU.at(0).device);
				SWITCH(i, printThisLimb(1));
			}
			std::cout << std::endl;
		}
	}
	// cudaDeviceSynchronize();
	/*
	Ciphertext<DCRTPoly> y = T[0]->Clone();

	// Computes Chebyshev polynomials up to degree k
	// for y: T_1(y) = y, T_2(y), ... , T_k(y)
	// uses binary tree multiplication
	for (uint32_t i = 2; i <= k; i++) {
		// if i is a power of two
		if (!(i & (i - 1))) {
			// compute T_{2i}(y) = 2*T_i(y)^2 - 1
			auto square = cc->EvalSquare(T[i / 2 - 1]);
			T[i - 1] = cc->EvalAdd(square, square);
			cc->ModReduceInPlace(T[i - 1]);
			cc->EvalAddInPlace(T[i - 1], -1.0);
		} else {
			// non-power of 2
			if (i % 2 == 1) {
				// if i is odd
				// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
				auto prod = cc->EvalMult(T[i / 2 - 1], T[i / 2]);
				T[i - 1] = cc->EvalAdd(prod, prod);

				cc->ModReduceInPlace(T[i - 1]);
				cc->EvalSubInPlace(T[i - 1], y);
			} else {
				// i is even but not power of 2
				// compute T_{2i}(y) = 2*T_i(y)^2 - 1
				auto square = cc->EvalSquare(T[i / 2 - 1]);
				T[i - 1] = cc->EvalAdd(square, square);
				cc->ModReduceInPlace(T[i - 1]);
				cc->EvalAddInPlace(T[i - 1], -1.0);
			}
		}
	}
	*/
	for (size_t i = 1; i <= k; i++) {
		if (T[i - 1]->NoiseLevel == 2)
			T[i - 1]->rescale();
	}

	if (cc.rescaleTechnique == CKKS::FIXEDMANUAL) {

		for (size_t i = 1; i < k; i++) {
			T[i - 1]->dropToLevel(T[k - 1]->getLevel());
		}
	} else {

		/*
		assert(k >= 2);
		assert(T[k - 1]->getLevel() - T[k - 1]->NoiseLevel == T[k - 2]->getLevel() - T[k - 2]->NoiseLevel - 1);
		for (size_t i = 1; i < k - 1; i++) {
			if (!T[i - 1]->adjustForMult(*T[k - 2])) {
				//if (!T[k - 1]->adjustForAddOrSub(*T[i - 1])) {
				assert("false");
				//  std::cerr << "PANIC" << std::endl;
				//}
			}
			//algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
		}
		*/
	}
	/*
	const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(T[k - 1]->GetCryptoParameters());

	auto algo = cc->GetScheme();

	if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
		// brings all powers of x to the same level
		for (size_t i = 1; i < k; i++) {
			usint levelDiff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
			cc->LevelReduceInPlace(T[i - 1], nullptr, levelDiff);
		}
	} else {
		for (size_t i = 1; i < k; i++) {
			algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
		}
	}
	 */

	// Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
	T2[0]->copy(*T.back());
	for (uint32_t i = 1; i < m; i++) {
		if (cc.rescaleTechnique == FIXEDMANUAL && T2[i - 1]->NoiseLevel == 2)
			T2[i - 1]->rescale();
		T2[i]->square(*T2[i - 1], fusedRescaleFlag() != 0);
		T2[i]->add(*T2[i]);
		T2[i]->addScalar(-1.0);

		// T2[i]->rescale();
		if (cc.rescaleTechnique == FIXEDMANUAL && T2[i]->NoiseLevel == 2)
			T2[i]->rescale();
	}

	if constexpr (PRINT) {
		for (size_t j = 0; j < m; ++j) {
			std::cout << "T2[" << j << "]: " << std::endl;

			for (auto& i : T2[j]->c0.GPU.at(0).limb) {
				cudaSetDevice(T2[j]->c0.GPU.at(0).device);
				SWITCH(i, printThisLimb(1));
			}
			std::cout << std::endl;
		}
	}
	if constexpr (sync)
		cudaDeviceSynchronize();
	/*
	std::vector<Ciphertext<DCRTPoly>> T2(m);
	// Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
	// T2[0] is used as a placeholder
	T2.front() = T.back();
	for (uint32_t i = 1; i < m; i++) {
		auto square = cc->EvalSquare(T2[i - 1]);
		T2[i] = cc->EvalAdd(square, square);
		cc->ModReduceInPlace(T2[i]);
		cc->EvalAddInPlace(T2[i], -1.0);
	}
	*/

	// computes T_{k(2*m - 1)}(y)
	Ciphertext T2km1(cc_);

	T2km1.copy(*T2[0]);
	if (cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		T2km1.dropToLevel(T2[1]->getLevel());

	} else {
		// if (!T2km1.adjustForMult(*T2[1])) {
		//     assert(false);
		// }
	}

	for (uint32_t i = 1; i < m; i++) {
		// compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
		T2km1.mult(*T2[i], fusedRescaleFlag() != 0);
		T2km1.add(T2km1);
		// T2km1.rescale();
		T2[0]->adjustForAddOrSub(T2km1);
		if (T2[0]->NoiseLevel == 1 && T2km1.NoiseLevel == 2)
			T2km1.rescale();
		T2km1.sub(*T2[0]);
		if (T2[0]->NoiseLevel == 2 && i < m - 1 && T2km1.NoiseLevel == 2)
			T2km1.rescale();

		if constexpr (sync)
			cudaDeviceSynchronize();
	}
	if constexpr (PRINT) {
		std::cout << "T2kmi cheby " << T2km1.getLevel() << " " << T2km1.NoiseLevel << std::endl;
		for (auto& i : T2km1.c0.GPU.at(0).limb) {
			cudaSetDevice(T2km1.c0.GPU.at(0).device);
			SWITCH(i, printThisLimb(1));
		}
		std::cout << std::endl;
	}
	/*
	// computes T_{k(2*m - 1)}(y)
	auto T2km1 = T2.front();
	for (uint32_t i = 1; i < m; i++) {
		// compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
		auto prod = cc->EvalMult(T2km1, T2[i]);
		T2km1 = cc->EvalAdd(prod, prod);
		cc->ModReduceInPlace(T2km1);
		cc->EvalSubInPlace(T2km1, T2.front());
	}

	// We also need to reduce the number of levels of T[k-1] and of T2[0] by another level.
	//  cc->LevelReduceInPlace(T[k-1], nullptr);
	//  cc->LevelReduceInPlace(T2.front(), nullptr);
	*/

	if constexpr (true) {
		Ciphertext out(cc_);
		innerEvalChebyshevPS(ctxt, ctxt, f2, k, m, T, T2, 0, m);
		// ctxt.copy(out);
	}

	ctxt.sub(T2km1);
	/*

	Ciphertext<DCRTPoly> result;

	if (flag_c) {
		result = cc->EvalAdd(T2[m - 1], cu);
	} else {
		result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);
	}

	result = cc->EvalMult(result, qu);
	cc->ModReduceInPlace(result);

	cc->EvalAddInPlace(result, su);
	cc->EvalSubInPlace(result, T2km1);

	return result;
	*/
	if constexpr (sync)
		cudaDeviceSynchronize();
}

void FIDESlib::CKKS::evalHornerSeries(Ciphertext& ctxt, const std::vector<double>& coefficients) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });

	int n = (int)coefficients.size() - 1;
	if (n < 0) return;

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;

	if (n == 0) {
		ctxt.multScalar(0.0);
		ctxt.addScalar(coefficients[0]);
		return;
	}

	// acc = c_n * x + c_{n-1}  (scalar ops, 0 levels consumed)
	Ciphertext acc(cc_);
	acc.copy(ctxt);
	if (acc.NoiseLevel == 2) acc.rescale();
	acc.multScalar(coefficients[n]);
	acc.addScalar(coefficients[n - 1]);

	// acc = acc * x + c_i  (1 ctxt-ctxt mult per step)
	for (int i = n - 2; i >= 0; i--) {
		Ciphertext x_tmp(cc_);
		x_tmp.copy(ctxt);
		acc.adjustForMult(x_tmp);
		x_tmp.adjustForMult(acc);
		acc.mult(acc, x_tmp, false);
		if (cc.rescaleTechnique == FIXEDMANUAL && acc.NoiseLevel == 2)
			acc.rescale();
		acc.addScalar(coefficients[i]);
	}

	ctxt.copy(acc);
}

void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval) {
	FIDESlib::CudaNvtxRange r_(std::string{ sc::current().function_name() });
	ContextData& cc = ctxt.cc;
	int32_t r		= its;
	// std::cout << "Its: " << its << std::endl;
	// Stage stash: distinguish the Re/Im calls in the stash names
	static std::atomic<int> daCall{0};
	const int callId = FIDESlib::CKKS::g_btsStageStash ? daCall++ : 0;
	for (int32_t j = 1; j < r + 1; j++) {
		if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
			ctxt.rescale();
		ctxt.square(fusedRescaleFlag() != 0);
		if (FIDESlib::CKKS::g_btsStageStash) {
			cudaDeviceSynchronize();
			auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
			c->copy(ctxt);
			FIDESlib::CKKS::g_btsStageStash->emplace_back(
			    "EM-DAc" + std::to_string(callId) + "-sq" + std::to_string(j), std::move(c));
		}
		double scalar = -1.0 / std::pow((2.0 * M_PI), std::pow(2.0, j - r));
		ctxt.add(ctxt);
		ctxt.addScalar(scalar);
		if (FIDESlib::CKKS::g_btsStageStash) {
			cudaDeviceSynchronize();
			auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
			c->copy(ctxt);
			FIDESlib::CKKS::g_btsStageStash->emplace_back(
			    "EM-DAc" + std::to_string(callId) + "-it" + std::to_string(j), std::move(c));
		}

		// cudaDeviceSynchronize();
	}
}