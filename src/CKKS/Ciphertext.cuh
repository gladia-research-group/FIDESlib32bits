//
// Created by carlosad on 24/04/24.
//

#ifndef FIDESLIB_CKKS_CIPHERTEXT_CUH
#define FIDESLIB_CKKS_CIPHERTEXT_CUH

#include <source_location>
#include <functional>
#include "RNSPoly.cuh"
#include "forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::CKKS {

/**
 * @brief Global flag controlling whether hoisted rotation optimisation is enabled.
 *
 * This can be toggled at runtime to switch between the standard hoisted-rotation
 * implementation and the fused hoisted‑rotation variant.
 */
extern bool hoistRotateFused;

/// One-prime rescale (FIDESLIB_STC_SINGLE): while set, Ciphertext::rescale() drops one prime, not compositeDegree().
extern thread_local bool g_rescaleOnePrime;
struct RescaleOneScope {
    bool old;
    explicit RescaleOneScope(bool a) : old(g_rescaleOnePrime) { g_rescaleOnePrime = a; }
    ~RescaleOneScope() { g_rescaleOnePrime = old; }
};

/**
 * @brief Pinned-arena offload descriptor for one ciphertext (async KV-cache swap).
 *
 * storeStaged() writes c0 then c1 contiguously into a pinned arena slot and fills the per-limb
 * (offset,length) tables plus the scalar metadata; the CryptoContext fills `moduli` from the chain.
 * loadStaged() reconstructs the device ciphertext from the same slot. All copies are async (no sync).
 */
struct StagedCtMeta {
    int numRes = 0;
    int N = 0;
    int NoiseLevel = 1;
    int slots = 0;
    double Noise = 0;
    std::string keyid;
    std::vector<uint64_t> moduli;                    // numRes (filled by CryptoContext)
    std::vector<size_t> off0, len0, off1, len1;      // per-limb byte offsets/lengths within the slot
    size_t total_bytes = 0;
};

/**
 * @class Ciphertext
 * @brief Represents a ciphertext in the CKKS scheme.
 *
 * The ciphertext holds two RNSPoly components (c0 and c1) together with
 * associated metadata (key identifier, noise level, scaling factor, etc.).
 * It provides arithmetic operations, key‑switching, rotations and
 * rescaling.  Most operations maintain the level information and may
 * automatically perform rescaling or modulus‑down conversion as required
 * by the CKKS protocol.
 */
class Ciphertext {
    /** @brief Identifier string for NVTX profiling. */
    static constexpr const char* loc{"Ciphertext"};
    /** @brief NVTX range object for profiling this class' methods. */
    CudaNvtxRange my_range;

   public:
    /** @brief Identifier for the key associated with this ciphertext. */
    KeyHash keyID{""};
    /** @brief Reference to the CKKS context owning this ciphertext. */
    Context& cc_;
    /** @brief Reference to the context data (metadata) for this ciphertext. */
    ContextData& cc;
    /** @brief The two polynomial components of the ciphertext (c0 and c1). */
    RNSPoly c0, c1;
    /** @brief Accumulated noise factor for this ciphertext. */
    double NoiseFactor = 0;
    /** @brief Discrete noise level indicator. */
    int NoiseLevel = {1};
    /** @brief Number of slots (vector length) encoded in the ciphertext. */
    int slots = 0;

    /**
     * @brief Move constructor. Transfers ownership of the internal resources
     *        from another Ciphertext instance.
     *
     * @param ct_moved Rvalue reference to the source Ciphertext.
     */
    Ciphertext(Ciphertext&& ct_moved) noexcept;

    /**
     * @brief Constructs an empty Ciphertext bound to a given context.
     *
     * The created ciphertext has uninitialised polynomial components and zero noise.
     *
     * @param cc Reference to the CKKS Context the ciphertext belongs to.
     */
    explicit Ciphertext(Context& cc);

    /**
     * @brief Constructs a Ciphertext from a serialized RawCipherText.
     *
     * This loads polynomial data and metadata from the raw representation.
     *
     * @param cc   Reference to the CKKS Context.
     * @param rawct Serialized representation of a ciphertext.
     */
    Ciphertext(Context& cc, const RawCipherText& rawct);

    /** @brief Destructor. Releases any allocated resources. */
    ~Ciphertext();

    /**
     * @brief Copies only the metadata fields from another ciphertext.
     *
     * This does **not** copy polynomial data.
     *
     * @param a Source ciphertext to copy metadata from.
     */
    void copyMetadata(const Ciphertext& a);

    /**
     * @brief Merges metadata from two ciphertexts into *this*.
     *
     * Used after operations that combine two ciphertexts.
     *
     * @param a First operand.
     * @param b Second operand.
     */
    void addMetadata(const Ciphertext& a, const Ciphertext& b);

    /**
     * @brief Merges metadata from a ciphertext and a plaintext into *this*.
     *
     * @param a Ciphertext operand.
     * @param b Plaintext operand.
     */
    void addMetadata(const Ciphertext& a, const Plaintext& b);

    /**
     * @brief Merges metadata appropriate for multiplication of two ciphertexts.
     *
     * @param a First operand.
     * @param b Second operand.
     */
    void multMetadata(const Ciphertext& a, const Ciphertext& b);

    /**
     * @brief Merges metadata for multiplication of a ciphertext with a plaintext.
     *
     * @param a Ciphertext operand.
     * @param b Plaintext operand.
     */
    void multMetadata(const Ciphertext& a, const Plaintext& b);

    /**
     * @brief Normalises a rotation index to the valid range for the current slot count.
     *
     * The function makes the index positive, maps it into `[0, slots)` and, if the
     * index exceeds half the slot count, adds `N/2 - slots` to map it into the
     * underlying NTT domain.
     *
     * @param index Desired rotation index (may be negative).
     * @return Normalized index within `[0, slots)`.
     */
    int normalyzeIndex(int index) const;

    /**
     * @brief Loads ciphertext data from a RawCipherText representation.
     *
     * @param rawct Serialized ciphertext to load.
     */
    void load(const RawCipherText& rawct);

    /**
     * @brief Stores the current ciphertext into a RawCipherText representation.
     *
     * @param rawct Destination object that will receive the serialized data.
     */
    void store(RawCipherText& rawct);

    /**
     * @brief Drain-free store for async offload (KV-cache eviction).
     *
     * Identical to store(rawct) but omits the two unconditional
     * cudaDeviceSynchronize() that the plain overload issues. The per-limb
     * cudaStreamSynchronize inside Limb::store (and c0/c1.sync()) already
     * guarantee the D->H copies complete before this returns, so the two
     * whole-device drains would be redundant over-synchronisation — at ~780 K/V
     * ciphertexts/token, ~1560 device-wide serialisations avoided. The
     * `stream` argument is reserved for a future dedicated-offload-stream copy
     * path; today the per-limb copies still run on their own limb streams.
     */
    void store(RawCipherText& rawct, cudaStream_t stream);

    /** @brief Upper-bound byte size of this ciphertext's staged form ((level+1)*2 limbs, u64). */
    [[nodiscard]] size_t staged_bytes() const;

    /** @brief Async D2H of c0,c1 into a pinned arena slot `base`; fills `m` (except moduli). No sync. */
    void storeStaged(uint8_t* base, StagedCtMeta& m, cudaStream_t stream);
    /// storeStaged with producer ordering: bridges the partition stream (where the ops
    /// that produced this value ran) into `stream` via an event before copying — the
    /// async-capture snapshot primitive (no device-wide sync, unlike store()).
    /// max_limbs > 0 snapshots only the first max_limbs towers (magnitude probes).
    void storeStagedOrdered(uint8_t* base, StagedCtMeta& m, cudaStream_t stream, int max_limbs = -1);

    /** @brief Async H2D reconstruction of c0,c1 from a pinned arena slot `base` per `m`. No sync. */
    void loadStaged(const uint8_t* base, const StagedCtMeta& m, cudaStream_t stream);

    /**
     * @brief Adds another ciphertext to *this* (in‑place).
     *
     * The method ensures both ciphertexts are on the same level, performing
     * rescaling or level adjustments when required by the current
     * `rescaleTechnique`.  It may invoke `adjustForAddOrSub` and, if that
     * fails, create a temporary copy of `b` with aligned scaling before
     * retrying.  After the polynomial addition, metadata is merged via
     * `addMetadata`.  The operation count for `ADD` is incremented.
     *
     * @param b Ciphertext to add.
     */
    void add(const Ciphertext& b);

    /**
     * @brief Subtracts another ciphertext from *this* (in‑place).
     *
     * Mirrors the behaviour of `add` but uses subtraction on the polynomial
     * components.  Rescaling, level alignment and metadata handling follow the
     * same logic as in `add`.
     *
     * @param b Ciphertext to subtract.
     */
    void sub(const Ciphertext& b);

    /**
     * @brief Adds a plaintext to *this* ciphertext (in‑place).
     *
     * Handles automatic rescaling when the plaintext and ciphertext have
     * mismatched levels or noise.  For flexible rescaling techniques it may
     * create a temporary adjusted plaintext before performing the addition.
     * After the operation, metadata is merged via `addMetadata`.
     *
     * @param b Plaintext operand.
     */
    void addPt(const Plaintext& b);

    /**
     * @brief Subtracts a plaintext from *this* ciphertext (in‑place).
     *
     * The semantics are analogous to `addPt` but perform subtraction on `c0`.
     *
     * @param pt Plaintext operand.
     */
    void subPt(const Plaintext& pt);

    /**
     * @brief Adds a scalar (double) to both polynomial components.
     *
     * The scalar is first converted to the appropriate modulus representation
     * via `ElemForEvalAddOrSub`.  The sign of the scalar is handled by
     * converting the element to its complement when `c < 0`.  No metadata
     * updates are performed.
     *
     * @param c Scalar value to add.
     */
    void addScalar(const double c);

    /**
     * @brief Multiplies *this* ciphertext by a plaintext.
     *
     * If the current rescaling technique requires matching noise levels,
     * the method may rescale the ciphertext or adjust the plaintext
     * (creating a temporary copy) before multiplication.  After the
     * component‑wise multiplication, metadata is merged via `multMetadata`.
     * If `rescale` is true and the technique is `FIXEDMANUAL`, the
     * ciphertext is rescaled and the noise factor/level are updated.
     *
     * @param b Plaintext operand.
     * @param rescale If true, performs rescaling after multiplication.
     */
    void multPt(const Plaintext& b, bool rescale = false);

    /**
     * @brief Multiplies a ciphertext `c` by a plaintext and stores the result in *this*.
     *
     * Equivalent to copying `c` into *this* and then invoking `multPt(b, rescale)`.
     *
     * @param c Ciphertext operand.
     * @param b Plaintext operand.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

    /**
     * @brief Computes `*this + (c * b)` in a single fused operation.
     *
     * The method adds ciphertext `c * b` to *this* and optionally rescales.  
     * Metadata is merged via `multMetadata`.
     *
     * @param c Ciphertext to add.
     * @param b Plaintext to multiply.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

    /**
     * @brief Multiplies *this* by another ciphertext.
     *
     * Handles level alignment and possible rescaling according to the current
     * `rescaleTechnique`.  If necessary, the method calls `adjustForMult`
     * (creating a temporary copy of `b` when alignment fails).  The actual
     * multiplication is performed before a key‑switch operation.  Metadata is
     * merged via `multMetadata`.  The operation count for `MULT` is incremented.
     *
     * @param b Ciphertext multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     * @param moddown Perform modulus down‑conversion if true.
     */
    void mult(const Ciphertext& b, bool rescale = false, bool moddown = true);

    /**
     * @brief Multiplies two ciphertexts and stores the result in *this*.
     *
     * This overload forwards to the two‑argument version, using `b` and `c`
     * as the two multiplicands.  It exists for convenience when the two
     * operands are already separate objects.
     *
     * @param b First operand.
     * @param c Second operand.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void mult(const Ciphertext& b, const Ciphertext& c, bool rescale = false);

    /**
     * @brief this += Σ_j a[j]·b[j] with ONE keyswitch (FHE_LANE_BATCH phase 2).
     *
     * *this* must already hold a NoiseLevel-2 product (the lane-0 seed) at the common
     * level; every a[j]/b[j] must be NoiseLevel-1 at that level. The binomial partial
     * products of all lanes are accumulated in one fused kernel pass (d0→c0, d1→c1,
     * d2→keyswitch aux), then a single relinearization closes the sum — numerically
     * Σ relin(d2_j) == relin(Σ d2_j). No rescale, mirroring serial mult(rescale=false).
     */
    void multAccumulateBatch(const std::vector<const Ciphertext*>& a, const std::vector<const Ciphertext*>& b);

    /**
     * @brief Multiplies both polynomial components by a scalar without pre‑checks.
     *
     * The scalar is converted to the appropriate modulus element via
     * `ElemForEvalMult`.  If `rescale` is true and the technique is
     * `FIXEDMANUAL`, the ciphertext is rescaled after multiplication.
     *
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalarNoPrecheck(const double c, bool rescale = false);

    /**
     * @brief Multiplies both polynomial components by a scalar, performing necessary checks.
     *
     * For flexible rescaling techniques the method may rescale the ciphertext
     * when the noise level is 2.  The scalar is converted using
     * `ElemForEvalMult` and the multiplication is applied to both components.
     * Metadata (noise level/factor) is updated accordingly; if `rescale`
     * is true and the technique is `FIXEDMANUAL`, a rescaling step follows.
     *
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalar(const double c, bool rescale = false);

    /**
     * @brief Multiplies a ciphertext `b` by a scalar and stores the result in *this*.
     *
     * The method copies `b` into *this* and then calls `multScalar(c,
     * rescale)`.  It is useful when the caller wishes to keep `b` unchanged.
     *
     * @param b Ciphertext operand.
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalar(const Ciphertext& b, const double c, bool rescale = false);

    /**
     * @brief Squares the ciphertext (i.e., multiplies it by itself).
     *
     * After the multiplication the metadata (noise level/factor) is updated.
     * If `rescale` is true and the technique is `FIXEDMANUAL`, a rescaling
     * step follows.
     *
     * @param rescale Perform rescaling after squaring if true.
     */
    void square(bool rescale = false);

    /**
     * @brief Squares the source ciphertext `src` and stores the result in *this*.
     *
     * Equivalent to copying `src` into *this* and then calling `square(rescale)`.
     *
     * @param src Source ciphertext to square.
     * @param rescale Perform rescaling after squaring if true.
     */
    void square(const Ciphertext& src, bool rescale = false);

    /**
     * @brief Rotates the slots of the ciphertext by `index` positions.
     *
     * The index is normalised using `normalyzeIndex`.  Depending on the
     * `moddown` flag, a modulus‑down conversion may be performed after the
     * rotation.
     *
     * @param index Number of slots to rotate (positive = left).
     * @param moddown Perform modulus down‑conversion after rotation if true.
     */
    void rotate(const int index, bool moddown = true);

    /**
     * @brief Rotates ciphertext `c` by `index` slots and stores the result in *this*.
     *
     * This overload copies `c` into *this* and then calls the single‑argument
     * `rotate(index, true)`.
     *
     * @param c Source ciphertext.
     * @param index Number of slots to rotate.
     */
    void rotate(const Ciphertext& c, const int index);

    /**
     * @brief Computes the complex conjugate of ciphertext `c` and stores it in *this*.
     *
     * The operation is performed component‑wise and updates the operation
     * count for `CONJUGATE`.
     *
     * @param c Source ciphertext.
     */
    void conjugate(const Ciphertext& c);

    /**
     * @brief Performs modulus down‑conversion on the ciphertext.
     *
     * Both polynomial components are down‑converted; when `free` is true the
     * temporary special limbs are released.
     *
     * @param free If true, frees intermediate buffers after conversion.
     */
    void modDown(bool free = false);
    /// Lever A: ModDown of both components fused with the composite rescale (level -2, NoiseLevel -1).
    void modDownRescale();
    /// Lever E2: *this = src * P with zeroed special limbs (== copy(src); extend()), one pass per limb.
    void copyExtend(const Ciphertext& src);

    /** @brief Increases the modulus level of the ciphertext (level up). */
    void modUp();

    /** @brief Rescales the ciphertext to the next level, adjusting scaling factor.
     *  Under a RescaleOneScope it drops ONE prime instead of compositeDegree() (the ciphertext then
     *  sits off the composite level grid until a matching one-prime drop brings it back). */
    void rescale();

    /**
     * @brief Grows the ciphertext to at least `level` by performing necessary modulus upgrades.
     *
     * The method ensures both components reach the requested level, generating
     * special limbs if required.
     *
     * @param level Desired minimum level.
     */
    void growToLevel(int level);

    /** @brief Drops the ciphertext to a lower level, discarding higher moduli. 
    
        @todo adjust the scale on FLEXIBLE modes
    */
    void dropToLevel(int level);

    /**
     * @brief Returns the current modulus level of the ciphertext.
     *
     * @return Current level index (both components share the same level).
     */
    [[nodiscard]] int getLevel() const;

    /**
     * @brief Performs an automorphism on the ciphertext.
     *
     * The automorphism is applied to both polynomial components; the `br`
     * argument is passed to the underlying `RNSPoly::automorph` method and
     * is implementation‑specific.
     *
     * @param index Automorphism index.
     * @param br    Additional parameter used by the underlying implementation.
     */
    void automorph(const int index, const int br);

    /**
     * @brief Performs a batch automorphism on multiple ciphertexts.
     *
     * Equivalent to calling `automorph` on each component; the `br` argument
     * is forwarded unchanged.
     *
     * @param index Automorphism index.
     * @param br    Additional parameter used by the underlying implementation.
     */
    void automorph_multi(const int index, const int br);

    /**
     * @brief Extends the ciphertext to a higher level if needed.
     *
     * If `init` is true, the method also (re)initialises internal structures
     * and, when necessary, scales the components by the prime modulus.
     *
     * @param init If true, also (re)initialises internal structures.
     */
    void extend(bool init = true);

    /**
     * @brief Performs hoisted rotations for a set of indexes.
     *
     * The method normalises each index, allocates temporary results if they do
     * not already exist, and optionally extends the ciphertexts before rotation.
     *
     * @param indexes Vector of rotation indexes.
     * @param results Vector to receive resulting ciphertext pointers.
     * @param ext     If true, extends ciphertexts before rotation.
     */
    void rotate_hoisted(const std::vector<int>& indexes, std::vector<Ciphertext*> results, bool ext);
    /// Chunked form for LinearTransform (fused single-GPU path, ext = true only): one ModUp,
    /// then the hoisted key-switch dot is launched per limb range and `on_chunk(part, begin,
    /// count)` runs right after each range (part 1 = special limbs, 0 = regular limbs), so the
    /// consumer reads the rotated limbs while they are still L2-resident. The consumer must
    /// order itself after `cc.getKeySwitchAux()`'s stream.
    void rotate_hoisted_chunked(const std::vector<int>& indexes, std::vector<Ciphertext*> results, int chunk,
                                const std::function<void(int part, int begin, int count)>& on_chunk);

    /**
     * @brief Evaluates a linear weighted sum (mutable version) over `n` ciphertexts.
     *
     * The method grows the result to the appropriate level, multiplies each
     * ciphertext by the corresponding weight (converted via `ElemForEvalMult`),
     * and accumulates the sum.  Metadata (noise level/factor) is updated to
     * reflect the combined operation.
     *
     * @param n      Number of ciphertexts.
     * @param ctxs   Vector of ciphertext pointers.
     * @param weights Vector of weights applied to each ciphertext.
     */
    void evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights);

    /**
     * @brief Adds a scaled version of `ciphertext` (multiplied by `d`) to *this*.
     *
     * The scalar multiplication is performed on a temporary copy of the
     * operand, which is then added component‑wise.  Metadata is merged via
     * `addMetadata`.
     *
     * @param ciphertext Ciphertext to be scaled and added.
     * @param d          Scalar multiplier.
     */
    void addMultScalar(const Ciphertext& ciphertext, double d);

    /**
     * @brief Adds scalar `c` to ciphertext `b` and stores result in *this*.
     *
     * Equivalent to copying `b` into *this* and invoking `addScalar(c)`.
     *
     * @param b Ciphertext operand.
     * @param c Scalar to add.
     */
    void addScalar(const Ciphertext& b, double c);

    /**
     * @brief Adds two ciphertexts and stores the result in *this*.
     *
     * Handles self‑addition and aliasing cases; the method may copy the
     * appropriate operand before performing the addition.
     *
     * @param ciphertext   First operand.
     * @param ciphertext1  Second operand.
     */
    void add(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);

    /**
     * @brief Subtracts the second ciphertext from the first and stores the result in *this*.
     *
     * Handles self‑subtraction and aliasing cases similarly to `add`.
     *
     * @param ciphertext   Minuend.
     * @param ciphertext1  Subtrahend.
     */
    void sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);

    /**
     * @brief Copies the full content (polynomials and metadata) from another ciphertext.
     *
     * @param ciphertext Source ciphertext.
     */
    void copy(const Ciphertext& ciphertext);

    /**
     * @brief Takes the polynomial buffers of `src` by swap (no device copy) and copies its
     * metadata. `src` must be dying or about to be fully overwritten: it is left holding
     * *this*'s previous buffers (valid, unspecified contents). Replaces `copy()` where the
     * source is a local accumulator about to go out of scope.
     */
    void takeFrom(Ciphertext& src);

    /**
     * @brief Adds a plaintext to a ciphertext and stores the result in *this*.
     *
     * @param ciphertext Ciphertext operand.
     * @param plaintext  Plaintext operand.
     */
    void addPt(const Ciphertext& ciphertext, const Plaintext& plaintext);

    /**
     * @brief Reinterprets the internal data of `ciphertext` under the current context.
     *
     * The method checks that the source ciphertext belongs to a compatible
     * context (same primes) and that its level does not exceed the current
     * context's maximum.  It then copies the data.
     *
     * @param ciphertext Source ciphertext.
     */
    void reinterpretContext(const Ciphertext& ciphertext);

    /**
     * @brief Performs a key‑switch operation using the provided switching key.
     *
     * @param ksk KeySwitchingKey to apply.
     */
    void keySwitch(const KeySwitchingKey& ksk);

    /**
     * @brief Adjusts internal scaling factors to make addition/subtraction possible.
     *
     * The method may rescale or multiply by 1.0 depending on the current
     * `rescaleTechnique`.  It returns `true` when the adjustment succeeds.
     *
     * @param ciphertext Operand that may have a different scaling factor.
     * @return `true` on successful adjustment.
     */
    bool adjustForAddOrSub(const Ciphertext& ciphertext);
    /** The untraced bodies of adjustForAddOrSub/adjustForMult; the public entry points are
     *  thin wrappers recording the E3a histogram under FIDESLIB_ADJUST_TRACE (default off,
     *  byte-identical behavior). Call the public names, not these. */
    bool adjustForAddOrSubBody(const Ciphertext& ciphertext);
    bool adjustForMultBody(const Ciphertext& ciphertext);

    /**
     * @brief Adjusts scaling factors to enable multiplication with `ciphertext`.
     *
     * The method first calls `adjustForAddOrSub`; if the noise level of the
     * operand is higher, the current ciphertext may be rescaled.  It returns
     * `true` when the multiplication can proceed without further adjustment.
     *
     * @param ciphertext Operand to align scaling with.
     * @return `true` on successful adjustment.
     */
    bool adjustForMult(const Ciphertext& ciphertext);

    /**
     * @brief Checks whether the ciphertext and a plaintext share the same scaling factor.
     *
     * The check accounts for a small tolerance (`1e-9` relative error).
     *
     * @param b Plaintext to compare against.
     * @return `true` if scaling factors match within tolerance.
     */
    [[nodiscard]] bool hasSameScalingFactor(const Plaintext& b) const;

    /** @brief Clears the operation record used for debugging/tracing. */
    static void clearOpRecord();

    /** @brief Prints the operation record to standard output. */
    static void printOpRecord();

    /**
     * @brief Computes the dot product between arrays of ciphertexts and plaintexts.
     *
     * The operation multiplies corresponding components and accumulates the result
     * in *this*.  When `ext` is true the method expects the inputs to have
     * already performed modulus‑up conversion.
     *
     * @param ciphertexts Pointer to array of ciphertexts.
     * @param plaintexts  Pointer to array of plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext* ciphertexts, Plaintext* plaintexts, const int n, bool ext);

    /**
     * @brief Computes the dot product between ciphertexts and an array of plaintext pointers.
     *
     * @param ciphertexts Pointer to array of ciphertexts.
     * @param plaintexts Pointer to array of pointers to plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext* ciphertexts, Plaintext** plaintexts, const int n, bool ext);

    /**
     * @brief Computes the dot product between arrays of ciphertext pointers and plaintext pointers.
     *
     * @param ciphertexts Pointer to array of pointers to ciphertexts.
     * @param plaintexts Pointer to array of pointers to plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext** ciphertexts, Plaintext** plaintexts, const int n, bool ext);
    void dotProduct(const std::vector<Ciphertext*>& a, const std::vector<Ciphertext*>& b, bool ext);

    /**
     * @brief Multiplies the ciphertext by a monomial of the given `power`.
     *
     * The monomial is cached in the context; if not present, it is generated.
     *
     * @param power Exponent of the monomial.
     */
    void multMonomial(int power);
};


/** FIDESLIB_KS_DIGIT_INTT: skip the redundant per-digit source INTT in the ct x ct keyswitch
 *  (default) or restore it (=1, the A/B arm). setKsDigitIntt overrides it at runtime so a test
 *  can run both arms in one process on the same ciphertext — see LimbPartitionMGPU.cu. */
void setKsDigitIntt(bool on);
bool ksDigitIntt();

}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_CIPHERTEXT_CUH
