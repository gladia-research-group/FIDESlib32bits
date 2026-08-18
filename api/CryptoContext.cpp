#include "CryptoContext.hpp"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
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
#include <cmath>
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
#include <mutex>
#include <future>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

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
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalMultKeys();
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalAutomorphismKeys();
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

bool fhe_pin_stage() {
	static const bool v = [] {
		const char* e = std::getenv("FHE_PIN_STAGE");
		return !(e && *e && std::atoi(e) == 0);
	}();
	return v;
}

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

bool stage_stats_enabled() {
	static const bool v = [] {
		const char* e = std::getenv("FHE_STAGE_STATS");
		return e && *e && std::atoi(e) != 0;
	}();
	return v;
}
struct StagedEntry {
	FIDESlib::CKKS::RawPlainText meta;   // arena!=null ⇒ sub_0 cleared (data in arena); else sub_0 kept
	const uint8_t*				 arena = nullptr;
	std::vector<size_t>			 off;    // per-limb byte offset into arena
	std::vector<size_t>			 len;    // per-limb byte length
	// COEFF-mode (MarkCoeffStaged): the single staged limb is a q0 EVAL limb; the load expands
	// it to target_limbs on the GPU instead of uploading pre-built limbs.
	bool						 coeff		  = false;
	int							 target_limbs = 0;
	// Arena generation this entry was staged under (multi-consume invalidation:
	// the entry dies when ITS arena is recycled, i.e. two flips later — a flip-time
	// clear is wrong under threaded prefetch, where the next block stages into the
	// other arena while the current block still loads).
	uint64_t					 gen = 0;
};
PinnedArena g_stage_arena[2];
int			g_stage_cur = 0;
std::mutex	g_stage_mutex;
std::atomic<uint64_t> g_stage_gen{0};

// Start staging a new block: ping-pong to the other arena, (lazily, once) allocate it, reset bump.
void stage_arena_begin() {
	std::lock_guard<std::mutex> g(g_stage_mutex);
	g_stage_gen.fetch_add(1, std::memory_order_relaxed);
	g_stage_cur	   = (g_stage_cur + 1) & 1;
	PinnedArena& a = g_stage_arena[g_stage_cur];
	const uint64_t ov = g_stage_overflow_pts.exchange(0);
	if (ov > 0 || stage_stats_enabled())
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
StagedEntry stage_into(PinnedArena& a, FIDESlib::CKKS::RawPlainText&& raw) {
	StagedEntry e;
	size_t		total = 0;
	for (const auto& limb : raw.sub_0)
		total += limb.size() * sizeof(uint64_t);
	bool ok = (a.base != nullptr) && total > 0;
	if (ok) {
		const size_t base_off = a.used.fetch_add(total, std::memory_order_relaxed);
		if (base_off + total > a.cap) {
			ok = false;   // reservation lost until the next flip resets the bump — arena is per block
			g_stage_overflow_pts.fetch_add(1, std::memory_order_relaxed);
		} else {
			e.off.reserve(raw.sub_0.size());
			e.len.reserve(raw.sub_0.size());
			size_t cur = base_off;
			for (auto& limb : raw.sub_0) {
				const size_t bytes = limb.size() * sizeof(uint64_t);
				std::memcpy(a.base + cur, limb.data(), bytes);
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
StagedEntry stage_raw(FIDESlib::CKKS::RawPlainText&& raw) {
	// Lock-free like the original: the loader flips BEFORE staging a block, and no
	// staging of the previous block overlaps its flip — cur/gen are stable here.
	StagedEntry e = stage_into(g_stage_arena[g_stage_cur], std::move(raw));
	e.gen		  = g_stage_gen.load(std::memory_order_relaxed);
	return e;
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
bool										 g_stage_persistent = false;
bool										 g_stage_multi_consume = false;

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
	return g_persist_staged.emplace(key, stage_into(g_persist_arena, std::move(raw))).first->second;
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
	if (!fhe_pin_stage())
		return;
	static std::once_flag once;
	std::call_once(once, [] {
		g_stage_prewarm = std::async(std::launch::async, [] {
			for (int i = 0; i < 2; ++i) {
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

void CryptoContextImpl<DCRTPoly>::BeginStageBlock() {
	if (fhe_pin_stage()) {
		stage_arena_begin();
		// Multi-consume mode keeps staged entries across loads. An entry must die
		// exactly when ITS arena is recycled — that is THIS flip for entries staged
		// two generations ago (same ping-pong parity). A blanket clear here is
		// wrong under threaded prefetch: the next block stages while the current
		// one still loads, and clearing would orphan the current block's entries
		// (measured: coeff "consumed twice" throw, 2026-08-10). Stale survivors
		// past their generation would alias recycled bytes (the 48856712 class),
		// hence the sweep below.
		if (g_stage_multi_consume && prefetched_raw_mutex) {
			const uint64_t G = g_stage_gen.load(std::memory_order_relaxed);
			prefetched_raw_mutex->lock();
			for (auto it = prefetched_raw.begin(); it != prefetched_raw.end();) {
				if (it->second.type() == typeid(StagedEntry) &&
					std::any_cast<const StagedEntry&>(it->second).gen + 2 <= G)
					it = prefetched_raw.erase(it);
				else
					++it;
			}
			prefetched_raw_mutex->unlock();
		}
	}
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

void CryptoContextImpl<DCRTPoly>::MarkCoeffStaged(Plaintext& pt, uint32_t target_level, double target_scale) {
	auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
	if (ptImpl->GetElement<lbcrypto::DCRTPoly>().GetAllElements().size() != 1)
		OPENFHE_THROW("MarkCoeffStaged: expected a 1-limb (q0) host encode");
	ptImpl->SetLevel(target_level);
	ptImpl->SetScalingFactor(target_scale);
	pt->coeff_staged = true;
}

// Called from ~PlaintextImpl: drop a worker-staged entry that was never consumed. Without this,
// the allocator can recycle the address into a NEW plaintext, whose first load would silently
// upload the dead object's staged limbs (the g_persist_staged 48856712 bug class).
void CryptoContextImpl<DCRTPoly>::ForgetPrefetchedRaw(const void* key) {
	if (!prefetched_raw_mutex)
		return;
	prefetched_raw_mutex->lock();
	prefetched_raw.erase(key);
	prefetched_raw_mutex->unlock();
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

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt, cudaStream_t stream_override) {
	if (pt->loaded || this->devices.empty())
		return;

	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu);
	const cudaStream_t load_stream = ResolvePlaintextLoadStream(stream_override);

	const void* key = static_cast<const void*>(pt.get());

	// Persistent staging (constant lm_head tiles): async-load from the persistent arena; stage on the
	// first miss (tok0). The entry is never erased (constant weight) so the arena pointer is stable.
	if (fhe_pin_stage() && g_stage_persistent) {
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
				e->target_limbs = static_cast<int>(this->multiplicative_depth + 1 - pt->GetLevel());
			}
		}
		if (e->coeff) {
			gpu_pt->loadCoeffExpand(e->meta, e->arena + e->off[0], e->len[0], e->target_limbs, load_stream);
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
		prefetched_raw_mutex->lock();
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
		prefetched_raw_mutex->unlock();
	}

	if (have_staged && staged.coeff) {
		gpu_pt->loadCoeffExpand(staged.meta, staged.arena + staged.off[0], staged.len[0],
								staged.target_limbs, load_stream);
	} else if (have_staged && staged.arena != nullptr) {
		gpu_pt->loadStaged(staged.meta, staged.arena, staged.off, staged.len, load_stream);
	} else {
		// A coeff-marked plaintext carries only its q0 limb — it MUST come through the staged
		// path (a plain 1-limb upload at a claimed deeper level would be silently wrong).
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
	if (pt->loaded || this->devices.empty() || !this->loaded || !prefetched_raw_mutex)
		return;
	const void* key = static_cast<const void*>(pt.get());

	// Persistent staging (constant lm_head tiles): stage once, idempotent — the check is against
	// g_persist_staged BEFORE the expensive GetRawPlainText so tok1+ is a true no-op.
	if (fhe_pin_stage() && g_stage_persistent) {
		std::lock_guard<std::mutex> g(g_persist_mutex);
		if (g_persist_staged.find(key) != g_persist_staged.end())
			return;
		auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
		persist_stage_locked(key, FIDESlib::CKKS::GetRawPlainText(context, ptImpl));
		return;
	}

	{   // idempotent: skip if already extracted (shared lock, no double GetRawPlainText)
		prefetched_raw_mutex->lock_shared();
		const bool have = prefetched_raw.find(key) != prefetched_raw.end();
		prefetched_raw_mutex->unlock_shared();
		if (have)
			return;
	}
	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& ptImpl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
	FIDESlib::CKKS::RawPlainText raw = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
	if (pt->coeff_staged && raw.numRes != 1)
		OPENFHE_THROW("ExtractRawPlaintext: coeff-staged plaintext must carry exactly one (q0) limb");
	// Stage into the pinned arena (host memcpy on this worker — overlapped, no CUDA call) when
	// FHE_PIN_STAGE; else stash the raw for a pageable upload.
	std::any entry = fhe_pin_stage() ? std::any(stage_raw(std::move(raw)))
									 : std::any(std::move(raw));
	if (pt->coeff_staged) {
		if (entry.type() != typeid(StagedEntry) || std::any_cast<const StagedEntry&>(entry).arena == nullptr)
			OPENFHE_THROW("ExtractRawPlaintext: coeff-staged plaintext requires the pinned arena "
						  "(FHE_PIN_STAGE on, arena not overflowed) — raise FHE_STAGE_ARENA_GB");
		StagedEntry& se = std::any_cast<StagedEntry&>(entry);
		se.coeff		= true;
		// pt->GetLevel() reports the TARGET level (MarkCoeffStaged); limbs = depth+1 - level.
		se.target_limbs = static_cast<int>(this->multiplicative_depth + 1 - pt->GetLevel());
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
	if (release_cpu && entry.type() == typeid(StagedEntry) &&
		std::any_cast<const StagedEntry&>(entry).arena != nullptr) {
		auto& pt_nc = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		pt_nc->GetElement<lbcrypto::DCRTPoly>() = lbcrypto::DCRTPoly();
	}
	prefetched_raw_mutex->lock();
	prefetched_raw[key] = std::move(entry);
	prefetched_raw_mutex->unlock();
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

	// FHE_TIME_KV diagnostic: split the offload into store(D2H)/moduli(OpenFHE)/evict to find the
	// 2.7 s; print + reset every 24 cts (~one token's worth). Zero overhead when off.
	static const bool kvbrk = [] { const char* e = std::getenv("FHE_TIME_KV"); return e && *e && std::atoi(e) != 0; }();
	static double g_store = 0, g_mod = 0, g_evict = 0;
	static int	  g_n	  = 0;
	auto _t = std::chrono::steady_clock::now();
	auto _lap = [&] { auto n = std::chrono::steady_clock::now(); double d = std::chrono::duration<double, std::milli>(n - _t).count(); _t = n; return d; };

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct_gpu->store(raw_ct, stream);	// drain-free device -> host
	if (kvbrk) g_store += _lap();

	auto& context	  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& chain = context->GetCryptoParameters()->GetElementParams()->GetParams();
	raw_ct.moduli.clear();
	raw_ct.moduli.reserve(raw_ct.numRes);
	for (int i = 0; i < raw_ct.numRes; ++i)
		raw_ct.moduli.push_back(chain[i]->GetModulus().ConvertToInt());
	if (kvbrk) g_mod += _lap();

	const uint32_t key = ct->gpu;
	if (!this->EvictDeviceCiphertext(key)) {
		OPENFHE_THROW("StoreDeviceCiphertext: could not evict ciphertext from device");
	}
	if (kvbrk) {
		g_evict += _lap();
		if (++g_n % 24 == 0) {
			std::fprintf(stderr, "[storebrk] store=%.1f moduli=%.1f evict=%.1f ms (per ~24 cts)\n", g_store, g_mod, g_evict);
			std::fflush(stderr);
			g_store = g_mod = g_evict = 0;
		}
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
		coeffchebyshev = { 0.24554573401685137,
			-0.047919064883347899,
			0.28388702040840819,
			-0.029944538735513584,
			0.35576522619036460,
			0.015106561885073030,
			0.29532946674499999,
			0.071203602333739374,
			-0.10347347339668074,
			0.044997590512555294,
			-0.42750712431925747,
			-0.090342129729094875,
			0.36762876269324946,
			0.049318066039335348,
			-0.14535986272411980,
			-0.015106938483063579,
			0.035951935499240355,
			0.0031036582188686437,
			-0.0062644606607068463,
			-0.00046609430477154916,
			0.00082128798852385086,
			0.000053910533892372678,
			-0.000084551549768927401,
			-4.9773801787288514e-6,
			7.0466620439083618e-6,
			3.7659807574103204e-7,
			-4.8648510153626034e-7,
			-2.3830267651437146e-8,
			2.8329709716159918e-8,
			1.2817720050334158e-9,
			-1.4122220430105397e-9,
			-5.9306213139085216e-11,
			6.3298928388417848e-11 };
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (this->keyDist == fideslib::SPARSE_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (this->keyDist == fideslib::UNIFORM_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
	} else {
		OPENFHE_THROW("Unsupported key distribution");
	}

	// FIDESLIB_ARCSINE = reserve + enable everywhere (isolation probes);
	// FIDESLIB_ARCSINE_RESERVE = reserve ONLY, correction stays off until a
	// caller scopes it on via setArcsineOverride (production: cutmax argmax).
	int arcsineLvls = 0;
	const auto env_on = [](const char* n) {
		const char* e = std::getenv(n);
		return e && *e && *e != '0';
	};
	if (env_on("FIDESLIB_ARCSINE") || env_on("FIDESLIB_ARCSINE_RESERVE")) {
		arcsineLvls = 3;  // measured: applyArcsineCorrection consumes 3 levels (job 48598682)
		if (const char* al = std::getenv("FIDESLIB_ARCSINE_LEVELS"); al && *al)
			arcsineLvls = std::atoi(al);
	}
	// FIDESLIB_SPARSE_ARCSINE = dual-slots mode: the arcsine reservation rides
	// ONLY sparse-slot precomps (slots < N/2); the full-slot precomp stays
	// byte-identical vanilla (reserve-without-consume is fatal, job 48603930).
	if (env_on("FIDESLIB_SPARSE_ARCSINE")) {
		arcsineLvls = 0;
		if (slots < context->GetRingDimension() / 2) {
			arcsineLvls = 3;
			if (const char* al = std::getenv("FIDESLIB_ARCSINE_LEVELS"); al && *al)
				arcsineLvls = std::atoi(al);
		}
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
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case fideslib::SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
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
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
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
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
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
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::JSON);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Encoding ----

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
  size_t noiseScaleDeg,
  uint32_t level,
  const std::shared_ptr<void> params,
  uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;

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
	auto pt       = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu      = std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded   = false;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext, stream_override);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;

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
	auto pt       = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu      = std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded   = false;

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
			auto pt_dummy = context->MakeCKKSPackedPlaintext(dummy, 1, this->multiplicative_depth - ct_gpu->getLevel());
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
	// BAKED ON (2026-07-21, user ruling — no longer tunable): the lazy CPU shadow is
	// value-validated on both arms (ViT e2e 49902068/49902164, GPT-2 planned decode A/B
	// 49902926 top1-identical) and strictly faster (ct×pt mult 659→35 µs, ViT block
	// 45→20 s, GPT-2 decode −15 %). The old FIDESLIB_LAZY_CPU_SHADOW env is ignored;
	// the CloneEmpty shadow with the loud-throw re-upload guards is the only behavior.
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
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);
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
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);
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
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);

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
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);

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
	ct_gpu->dropToLevel(static_cast<int>(this->multiplicative_depth) - static_cast<int>(level));
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