#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "Definitions.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"

namespace fideslib {

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	void SetDevices(const std::vector<int>& devices);
	void SetPlaintextStreams(cudaStream_t load_stream, cudaStream_t compute_stream);
	void ClearPlaintextStreams();

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Free a set of (already-loaded) rotation keys to reclaim GPU memory.
	/// Bootstrap DFT rotation indexes are protected (never removed). Intended for
	/// auxiliary rotation keys that are no longer needed after a phase (e.g. prefill
	/// filling-packing keys before decode). Returns the number of keys actually freed.
	size_t FreeRotationKeys(const std::vector<int>& steps, const PublicKey<DCRTPoly>& publicKey);
	/// @brief GPU-load a set of rotation keys that were deferred at LoadContext time
	/// (their OpenFHE eval keys already exist; this is the device transfer only).
	/// Dedups against already-resident keys. Used to lazily bring in the decode
	/// (cachemir) keys after prefill so they don't occupy the device during prefill.
	void LoadRotationKeys(const std::vector<int>& steps, const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a plaintext to the devices using a custom CUDA stream.
	/// @param pt Plaintext to load.
	/// @param stream CUDA stream for async H2D copies (single-GPU contexts only).
	void LoadPlaintext(Plaintext& pt, cudaStream_t stream);
	/// @brief CPU-ONLY pre-extraction half of the acquire (no CUDA). Builds the host RawPlainText
	/// (GetRawPlainText: limb copy + bit-reverse — the ~10.5 s/tok serial cost) and stashes it in
	/// prefetched_raw, so a later LoadPlaintext uploads without re-extracting. Safe to call from a
	/// residency WORKER thread during compute: reads only this->cpu + pt->cpu (both read-only) and
	/// the mutex-guarded stash; touches no device state, no inf.w, no FHE context. No-op if loaded.
	void ExtractRawPlaintext(Plaintext& pt);
	/// @brief Drop a worker-staged (prefetched_raw) entry for a plaintext being destroyed. Called
	/// from ~PlaintextImpl so a recycled address can never resurrect a dead object's staged limbs.
	void ForgetPrefetchedRaw(const void* key);
	/// @brief The FLEXIBLEAUTO per-level scaling factor (for the coeff-mode encode prescale).
	double ScalingFactorReal(uint32_t level) const;
	/// @brief Mark a 1-limb (q0) host encode as a COEFF-mode weight plaintext at `target_level`:
	/// overwrites the plaintext's level + scaling factor to the target's and sets coeff_staged.
	/// Staged loads then expand it to the target limbs ON THE GPU (INTT → broadcastLimb0 → NTT),
	/// replacing the host-side per-limb CRT+NTT that dominates MakeCKKSPackedPlaintext (~4.6x).
	void MarkCoeffStaged(Plaintext& pt, uint32_t target_level, double target_scale);
	/// @brief Begin staging a residency block under FHE_PIN_STAGE: ping-pong to the next pinned
	/// arena and reset it. Call once before the per-plaintext ExtractRawPlaintext calls of a block
	/// (from the residency worker). No-op when FHE_PIN_STAGE is off.
	void BeginStageBlock();
	/// @brief Toggle PERSISTENT staging (FHE_PIN_STAGE): while on, LoadPlaintext/ExtractRawPlaintext
	/// stage a plaintext once into a grow-once arena keyed by identity and async-load it every token
	/// without re-extracting — for CONSTANT weights reloaded per token (lm_head tiles). Off ⇒ the
	/// ping-pong block arena. Set true before such loads, false after.
	void SetPersistentStaging(bool on);
	void SetStageMultiConsume(bool on);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load. Handles both an OpenFHE-backed ct->cpu and
	/// a ct->cpu previously stashed by StoreDeviceCiphertext (RawCipherText).
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	/// @brief Offload a device ciphertext to host: download its GPU-computed data
	/// into ct->cpu (as a RawCipherText) and free the device copy. Inverse of
	/// LoadCiphertext. Unlike EvictDeviceCiphertext — which only frees device
	/// memory and leaves ct->cpu stale — this preserves the computed value, so a
	/// later LoadCiphertext restores it exactly. Returns false if not loaded.
	bool StoreDeviceCiphertext(Ciphertext<DCRTPoly>& ct);

	/// @brief Drain-free offload for async KV eviction. Same as StoreDeviceCiphertext(ct)
	/// but routes through Ciphertext::store(raw, stream) which omits the two device-wide
	/// cudaDeviceSynchronize() (K0). Safe to call from a residency worker thread: the
	/// offloaded_ciphertexts stash and device registry are mutex-guarded.
	bool StoreDeviceCiphertext(Ciphertext<DCRTPoly>& ct, cudaStream_t stream);

	/// @brief Async KV-cache offload into a reused PINNED arena, keyed by a stable cache position
	/// (block+lane), NOT the device handle. Enqueues the D2H of every limb on `stream` (no sync)
	/// and records the staged descriptor; does NOT free the device copy — call KvEvict after the
	/// stream is synced. Returns false if `ct` is not on device. Single-GPU only.
	/// @brief Kick off the 12GB pinned KV-arena allocation on a background thread (once), so the
	/// ~3.5s cudaMallocHost overlaps token-0 compute instead of stalling the first KV offload.
	/// Call at decode init when the swap is active.
	void PrewarmKvArena();
	bool KvStoreStaged(Ciphertext<DCRTPoly>& ct, const std::string& pos_key, cudaStream_t stream);
	/// @brief Free the device copy of a ciphertext previously enqueued by KvStoreStaged. Must be
	/// called only after the offload stream has been synchronised (the pinned slot is then valid).
	void KvEvict(Ciphertext<DCRTPoly>& ct);
	/// @brief Async KV-cache reload: reconstruct the device ciphertext from its pinned slot
	/// (`pos_key`) via an H2D on `stream` (no sync — the caller syncs before use). Registers a new
	/// device handle on `ct`. No-op if already loaded.
	void KvLoadStaged(Ciphertext<DCRTPoly>& ct, const std::string& pos_key, cudaStream_t stream);

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& sk, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg, uint32_t level,
	                        std::shared_ptr<void> params, uint32_t slots, cudaStream_t stream);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level,
	                        std::shared_ptr<void> params, uint32_t slots, cudaStream_t stream);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);

	void CopyCiphertextDevice(Ciphertext<DCRTPoly>& dst, const Ciphertext<DCRTPoly>& src);

	Ciphertext<DCRTPoly> MakeGpuResultLike(const Ciphertext<DCRTPoly>& src);
	/// Always true (baked 2026-07-21): fresh GPU-op outputs carry a metadata-only CPU shadow
	/// (CloneEmpty), never a deep copy. The FIDESLIB_LAZY_CPU_SHADOW env is no longer read.
	static bool LazyCpuShadowEnabled();
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	Ciphertext<DCRTPoly> EvalConjugate(const Ciphertext<DCRTPoly>& ciphertext);
	void EvalMultMonomialInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t power);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);

	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	/// @brief Value-preserving drop to a higher (fewer-limbs) target level via the same
	/// tower-drop used to align add/sub operands. No-op if already at/below target.
	void DropToLevel(Ciphertext<DCRTPoly>& ciphertext, uint32_t level);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);

	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	std::any cpu;
	std::any gpu;
	/// @brief Whether the context has been loaded to the devices.
	bool loaded = false;
	/// @brief List of devices the context is loaded on.
	std::vector<int> devices = { 0 };
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Optional async streams for plaintext load/compute coordination.
	cudaStream_t plaintext_load_stream = nullptr;
	cudaStream_t plaintext_compute_stream = nullptr;
	bool plaintext_streams_enabled = false;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Rotation indexes whose GPU load is deferred at LoadContext time (set
	/// before LoadContext). Their OpenFHE eval keys are still generated; only the
	/// device transfer is skipped until a later LoadRotationKeys() call.
	std::vector<int32_t> deferred_rotation_indexes;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;

	// ---- Copy helpers ----

	uint32_t CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct);

	// --- Map Handling ----

	/// @brief  Registry of plaintexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_plaintexts;
	std::unique_ptr<std::shared_mutex> device_plaintexts_mutex;
	/// @brief  Registry of plaintext readiness events by GPU handle.
	std::unordered_map<uint32_t, cudaEvent_t> plaintext_ready_events;
	std::unique_ptr<std::shared_mutex> plaintext_ready_events_mutex;
	/// @brief  Registry of ciphertexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_ciphertexts;
	std::unique_ptr<std::shared_mutex> device_ciphertexts_mutex;
	/// @brief Host-resident store for ciphertexts offloaded by StoreDeviceCiphertext,
	/// keyed by the (now-freed, monotonic) device handle. The value boxes a
	/// FIDESlib::CKKS::RawCipherText in std::any so this public header stays free of
	/// CUDA/internal types. ct->cpu is left as its original OpenFHE shell so any
	/// cpu-reading op still sees a valid ciphertext while offloaded.
	std::unordered_map<uint32_t, std::any> offloaded_ciphertexts;
	/// @brief Guards offloaded_ciphertexts. The async KV pipeline (K1/K2) reloads
	/// block i+1 (erase) on a residency worker thread while it offloads block i
	/// (insert) — without this lock those concurrent find/erase/insert race.
	std::unique_ptr<std::shared_mutex> offloaded_ciphertexts_mutex;
	/// @brief Bounded (one-block-ahead) host RawPlainText stash, keyed by host plaintext identity
	/// (PlaintextImpl*). ExtractRawPlaintext (CPU-only, worker thread) fills it during compute;
	/// LoadPlaintext drains+erases it on the device upload. NON-pinned, transient — at most the
	/// next block's weights live here (the prefetch+upload are one block apart), so it self-bounds
	/// (~1.7 GB). This is the CPU-extract overlap that hides the ~10.5 s/tok GetRawPlainText.
	std::unordered_map<const void*, std::any> prefetched_raw;
	std::unique_ptr<std::shared_mutex> prefetched_raw_mutex;
	/// @brief Next available handle for GPU objects. Zero is reserved as a null handle.
	/// RegisterDevicePlaintext / RegisterDeviceCiphertext increment this under DIFFERENT
	/// mutexes, so the residency worker (KV reload / weight load) and the main compute
	/// thread would race it (handle collision -> map corruption). The increment is
	/// serialised by a process-static mutex in RegisterDevice*; kept a plain uint32_t so
	/// CryptoContextImpl stays move-constructible (it is make_shared(std::move(...))'d).
	uint32_t next_gpu_handle = 1;

	uint32_t RegisterDevicePlaintext(std::shared_ptr<void>&& p);
	uint32_t RegisterDeviceCiphertext(std::shared_ptr<void>&& c);
	/// @brief Allocate the next GPU handle, serialised across plaintext/ciphertext
	/// registration (which hold different mutexes) so concurrent residency-worker and
	/// compute-thread registrations cannot collide.
	uint32_t next_handle();
	std::shared_ptr<void>& GetDevicePlaintext(uint32_t handle);
	std::shared_ptr<void>& GetDeviceCiphertext(uint32_t handle);
	bool EvictDevicePlaintext(uint32_t handle);
	bool EvictDeviceCiphertext(uint32_t handle);
	cudaStream_t ResolvePlaintextLoadStream(cudaStream_t stream_override) const;
	cudaStream_t ResolvePlaintextComputeStream() const;
	void RecordPlaintextReady(uint32_t handle, cudaStream_t stream);
	void WaitPlaintextReady(uint32_t handle);
	void ClearPlaintextReady(uint32_t handle);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);
};

/// @brief Kick off the pinned stage-arena allocations (2 × FHE_STAGE_ARENA_GB) on a background
/// thread, once. Worker-side block staging (ViT/prefill) uses block-sized arenas whose
/// cudaMallocHost costs seconds; call at driver init so the pinning overlaps context setup.
/// No-op when FHE_PIN_STAGE=0.
void PrewarmStageArenas();

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP