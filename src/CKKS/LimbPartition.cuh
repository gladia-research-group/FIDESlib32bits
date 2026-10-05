//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_LIMBPARTITION_CUH
#define FIDESLIB_CKKS_LIMBPARTITION_CUH

#include <atomic>
#include "Limb.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "PeerUtils.cuh"
#ifdef NCCL
#include "nccl.h"
#endif
namespace FIDESlib::CKKS {

extern bool MEMCPY_PEER;
extern bool GRAPH_CAPTURE;
/* FIDESLIB_KSK_REGEN level (0 off / 1 hoisted / 2 both `a`-readers, the default / 3 = 2 + stage-A smem arm).
 * One definition: the key-loading path and the launch gates must agree, since at >=2 nothing reads the
 * `a` rows and releasing them (adoptKskASeed) is only legal then. */
int kskRegenLevel();

class LimbPartition {
   public:
    ContextData& cc;
    const uint64_t uid;
    int* level;
    const int id;
    const int device;
#ifdef NCCL
    const ncclComm_t rank;  // For NCCL / RCCL
#else
    const int rank;
#endif
    Stream s;

    std::vector<LimbRecord>& meta;
    std::vector<LimbRecord>& SPECIALmeta;
    const std::vector<int>& digitid;
    std::vector<std::vector<LimbRecord>>& DECOMPmeta;
    std::vector<std::vector<LimbRecord>>& DIGITmeta;
    std::vector<LimbRecord>& GATHERmeta;

    std::vector<LimbImpl> limb;
    std::vector<LimbImpl> SPECIALlimb;
    std::vector<std::vector<LimbImpl>> DECOMPlimb;
    std::vector<std::vector<LimbImpl>> DIGITlimb;
    std::vector<LimbImpl> GATHERlimb;

    void** bufferAUXptrs;
    VectorGPU<void*> limbptr;
    VectorGPU<void*> auxptr;
    VectorGPU<void*> SPECIALlimbptr;
    VectorGPU<void*> SPECIALauxptr;

    std::vector<VectorGPU<void*>> DECOMPlimbptr;
    //  std::vector<VectorGPU<void*>> DECOMPauxptr;
    std::vector<VectorGPU<void*>> DIGITlimbptr;
    // std::vector<VectorGPU<void*>> DIGITauxptr;
    VectorGPU<void*> GATHERptr;
    // Concatenation of all digits' DECOMPlimbptr entries (digit-major, slot-minor, i.e. entry
    // (start_d + i) == DECOMPlimbptr[d][i]) so modup can run ONE wide INTT over all source limbs
    // instead of dnum small per-digit launches. Subview of bufferAUXptrs; filled once in
    // generateAllDecompAndDigit alongside DECOMPlimbptr.
    VectorGPU<void*> DECOMPALLptr;

    uint64_t* bufferDECOMPandDIGIT = nullptr;
    uint64_t* bufferSPECIAL = nullptr;
    uint64_t* bufferLIMB = nullptr;
    uint64_t* bufferGATHER = nullptr;

    /** Byte count and allocation ROUTE for the buffers handed back to GPUfree. GPUfree derives the
     *  free-list bucket from `bytes` (bytes=0 files the block in the 1 KB bucket, stranding it), so it
     *  must get the count GPUmalloc was given; a cudaMalloc'ed buffer must not go to cudaFreeAsync. */
    size_t bufferSPECIALbytes = 0;
    bool bufferSPECIALcudaMalloc = false;
    size_t bufferLIMBbytes = 0;
    void* bufferDECOMPandDIGIT_handle = nullptr;
    void* bufferGATHER_handle = nullptr;

    /** Persistent per-purpose device scratch for the short-lived pointer tables and operand vectors the
     *  ops below upload each call: one slot per purpose (never shared), grow-only, plain cudaMalloc/cudaFree.
     *  Reuse across calls is serialised by this partition's stream `s`, like the malloc/free pair it replaced. */
    enum ScratchSlot {
        SC_BATCH_ADD,
        SC_BATCH_MULTPT,
        SC_BATCH_ADDSCALAR,
        SC_BATCH_MULTSCALAR,
        SC_BATCH_BINOMIAL,
        SC_BATCH_LTDOT,
        SC_BATCH_HOISTROT,
        SC_MGPU_DOTKSK,
        SC_MGPU_HOISTROT,
        SC_MGPU_MODDOWN,
        SC_SCALAR_MULT,
        SC_SCALAR_ADD,
        SC_SCALAR_SUB,
        SC_LINWSUM_W,
        SC_LINWSUM_PS,
        SC_N
    };
    struct DevScratch {
        void* p = nullptr;
        size_t bytes = 0;
    };
    DevScratch scratch_[SC_N];
    /** Persistent scratch for `slot`, at least `bytes` big. Returns nullptr when bytes == 0 or the
     *  allocation fails; the caller must then fall back to its per-call allocation. */
    void* scratchGet(int slot, size_t bytes);
    void scratchFreeAll();


    /*
    LimbPartition(LimbPartition && lp) :
        device(lp.device),
        rank(lp.rank),
        meta(lp.meta),
        SPECIALmeta(lp.SPECIALmeta),
        DECOMPmeta(lp.DECOMPmeta),
        limb(std::move(lp.limb)),
        SPECIALlimb(std::move(lp.SPECIALlimb)),
        DECOMPlimb(std::move(lp.DECOMPlimb)),
        limbptr(std::move(lp.limbptr)),
        SPECIALlimbptr(std::move(lp.SPECIALlimbptr)),
        DECOMPlimbptr(std::move(lp.DECOMPlimbptr))
        {}
*/

    LimbPartition(LimbPartition&& l) noexcept;

    LimbPartition(ContextData& cc, const uint64_t& uid, int* level, const int id, bool def_stream = false);

    ~LimbPartition();

    Global::Globals* getGlobals();
    void binomialDotProduct(LimbPartition& c1, LimbPartition& c2, const std::vector<const LimbPartition*>& c0s,
                            const std::vector<const LimbPartition*>& c1s, const std::vector<const LimbPartition*>& d0s,
                            const std::vector<const LimbPartition*>& d1s, bool ext);
    void binomialMult(LimbPartition& c1, LimbPartition& c2, const LimbPartition& d0, const LimbPartition& d1,
                      bool extend_ins, bool square);
    /// Lever E1: out-of-place (this, c1, c2) = (a0, a1) x (d0, d1) (square: d = a), no P extension.
    void binomialMultFrom(LimbPartition& c1, LimbPartition& c2, const LimbPartition& a0, const LimbPartition& a1,
                          const LimbPartition& d0, const LimbPartition& d1, bool square);
    /// Lever E2: this = src * P (one pass; the identity rotation of a hoisted LT stage).
    void copyScaledByP(const LimbPartition& src);
    void generateLimbToLevel(int new_level);

    enum GENERATION_MODE { AUTOMATIC, SINGLE_BUFFER, DUAL_BUFFER };

    void generate(std::vector<LimbRecord>& records, std::vector<LimbImpl>& limbs, VectorGPU<void*>& ptrs, int pos,
                  VectorGPU<void*>* auxptrs, uint64_t* buffer = nullptr, size_t offset = 0,
                  uint64_t* buffer_aux = nullptr, size_t offset_aux = 0, bool noptr = false);

    void generateLimb();

    void generateSpecialLimb(bool zero_out, bool for_communication);

    void add(const LimbPartition& p, const bool ext);
    void add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b);

    void sub(const LimbPartition& p);

    void multElement(const LimbPartition& p);

    void multPt(const LimbPartition& p);

    void modup(LimbPartition& aux_partition);

    template <ALGO algo = ALGO_SHOUP>
    /// rescale2: fused ModDown + composite rescale (lever A): also drops the two top Q limbs (caller lowers level by 2).
    void moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs, bool rescale2 = false);

    void rescale();
    /** Fused composite DOUBLE prime drop (bit-identical to two rescale() calls,
     * ~half the kernel work). Returns false if the shape doesn't fit — caller must then fall
     * back to the sequential per-prime loop. See the definition for the eligibility rules. */
    bool rescale2();

    void freeSpecialLimbs();
    /** Release bufferSPECIAL with the byte count and the allocation route it was created with.
     *  Split out because the destructor needs the identical logic. */
    void freeSpecialBuffer();

    using OptReference = LimbPartition*;
    using OptConstReference = const LimbPartition*;
    struct NTT_fusion_fields {
        OptReference op2;
        OptConstReference pt;
        OptReference res0;
        OptReference res1;
        OptConstReference kska;
        OptConstReference kskb;
    };

    template <ALGO algo, NTT_MODE mode>
    void ApplyNTT(int batch, LimbPartition::NTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                  VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init,
                  const int limbsize = -1);

    template <ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
    void NTT(int batch = 1, bool sync = false, NTT_fusion_fields fields = NTT_fusion_fields{});

    struct INTT_fusion_fields {
        OptReference res0;
        OptReference res1;
        OptConstReference kska;
        OptConstReference kskb;
        OptConstReference c0;
        OptConstReference c0tilde;
        OptConstReference c1;
        OptConstReference c1tilde;
    };

    template <ALGO algo, INTT_MODE mode>
    void ApplyINTT(int batch, LimbPartition::INTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                   VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init,
                   const int limbsize);

    template <ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
    void INTT(int batch = 1, bool sync = false, INTT_fusion_fields fields = INTT_fusion_fields{});

    static std::vector<VectorGPU<void*>> generateDecompLimbptr(void** buffer,
                                                               const std::vector<std::vector<LimbRecord>>& DECOMPmeta,
                                                               const int device, int offset);

    void generateAllDecompLimb(uint64_t* pInt, size_t offset);

    void generateAllDigitLimb(uint64_t* pInt, size_t offset, int q_band = -1);

    void copyLimb(const LimbPartition& partition);
    void copySpecialLimb(const LimbPartition& p);
    /// NTT / INTT of the special limbs in place (SmallInt lift); synchronised through `s`.
    void nttSpecialLimbs();
    void inttSpecialLimbs();

    void generateAllDecompAndDigit(bool iskey, int q_band = -1);
    // Banded key (rotation-key limb pruning): Q-limbs allocated only up to
    // q_band (chain position), digits unused at ct level <= q_band skipped.
    // -1 = full key. Guarded in dotKSK.
    int key_q_band = -1;
    // KSK bit-packing: when >0, this partition holds KEY material whose
    // limbptr/DIGITlimbptr device tables point at key_pack_bits-bit packed streams carved
    // from bufferKSKPACK; the dense DECOMP/DIGIT Limb storage is freed (DECOMPlimb/DIGITlimb
    // cleared). Only the fusedDotKSK_2_/hoistedRotateDotKSK_2_ KSK_PACKED=true arms may read
    // these tables. All-u32 (type==0) single-GPU chains only.
    int key_pack_bits = 0;
    uint64_t* bufferKSKPACK = nullptr;
    size_t bufferKSKPACKbytes = 0;
    void packKeyLimbs(int bits);
    // In-kernel regen: the 256-bit seed this KEY partition's `a` rows were expanded from
    // (recorded by expandKskADigits). When set — and FIDESLIB_KSK_REGEN >= 1 — the dot kernels'
    // REGEN arms regenerate kska(digit, p, slot) in registers from this seed (KskSeedExpand.cuh,
    // bit-identical to the expanded rows) instead of streaming the `a` half of the key from DRAM.
    uint32_t ksk_seed[8] = {};
    bool ksk_seed_set = false;
    // At FIDESLIB_KSK_REGEN>=2 EVERY reader of this chain's `a` rows regenerates them, so the rows
    // are never materialized — adoptKskASeed() records the seed and releases the storage instead.
    // The device pointer tables survive holding nullptr; anything that would READ them must throw first.
    bool ksk_a_released = false;
    void adoptKskASeed(const std::vector<uint32_t>& seed, int q_band = -1);
    // Load-time expansion: fill this KEY partition's `a` DECOMP/DIGIT limbs on-GPU from the
    // 256-bit seed instead of H2D-copying them (bit-identical; builds the same limbptr mapping
    // loadDecompDigit would).
    void expandKskADigits(const std::vector<uint32_t>& seed);

    void mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2,
                            const LimbPartition& partition3, const LimbPartition& partition4);

    void mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2);

    void generateLimbSingleMalloc();
    void generateLimbConstant();

    void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                         const std::vector<std::vector<uint64_t>>& moduli);

    void dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace = false,
                const LimbPartition* limbsrc = nullptr);

    void multElement(const LimbPartition& partition1, const LimbPartition& partition2);

    void multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0,
                         const LimbPartition& c0tilde, const LimbPartition& ksk_a, const LimbPartition& ksk_b);

    int getLimbSize(int level) const;
    void automorph(const int index, const int br, LimbPartition* src, bool ext);

    void modupInto(LimbPartition& partition, LimbPartition& partition1);
    void multScalar(std::vector<uint64_t>& vector);
    void squareElement(const LimbPartition& p);
    void binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                            const LimbPartition& c2_key_switched_1);
    void addScalar(std::vector<uint64_t>& vector);
    void subScalar(std::vector<uint64_t>& vector);
    void dropLimb();
    void addMult(const LimbPartition& partition, const LimbPartition& partition1);
    void broadcastLimb0();
    /** COMPOSITESCALING ModRaise: CRT-extend the bottom d limbs across ALL current limbs
     *  (call after grow()). qhatinv[k] = (Q0/q_k)^{-1} mod q_k; qhat is the flattened
     *  (Q0/q_k) mod q_i table with stride = current limb count. Single-GPU only. */
    void compositeModRaise(int d, const std::vector<uint64_t>& qhatinv, const std::vector<uint64_t>& qhat);
    // Centred-aggregate CRT lift for coeff plaintexts (d==2); see coeffLiftCentered2_.
    void coeffLiftCentered(uint64_t q0, uint64_t q1, uint64_t q0inv_mod_q1, uint64_t Qhalf,
                           const std::vector<uint64_t>& Q0_mod_qi);
    void evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights);
    void rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                           const LimbPartition& ksk_b);
    void squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                           const LimbPartition& ksk_b);
    void rescaleMGPU();
    void moddownMGPU(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs,
                     const std::vector<uint64_t*>& bufferSpecial_);
    void generatePartialSpecialLimb();
    void dotProductPt(LimbPartition& c1, const std::vector<const LimbPartition*>& c0s,
                      const std::vector<const LimbPartition*>& c1s, const std::vector<const LimbPartition*>& pts,
                      bool ext);

    void generateGatherLimb(bool iskey);
    void dotKSKfusedMGPU(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a,
                         const LimbPartition& ksk_b, const LimbPartition& src);
    // y_begin / y_count restrict the launch to a range of the kernel's limb axis (specials
    // first, then the partition's limbs); y_count < 0 = everything (the default path).
    void fusedHoistRotate(int n, std::vector<int> indexes, std::vector<LimbPartition*>& c0,
                          std::vector<LimbPartition*>& c1, const std::vector<LimbPartition*>& ksk_a,
                          const std::vector<LimbPartition*>& ksk_b, const LimbPartition& src_c0,
                          const LimbPartition& src_c1, bool c0_modup, int y_begin = 0, int y_count = -1);

    void modup_ksk_moddown_mgpu(LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b,
                                LimbPartition& auxLimbs1, LimbPartition& auxLimbs2, const bool moddown,
                                const std::vector<uint64_t*>& bufferGather_,
                                const std::vector<uint64_t*>& bufferSpecial_c0,
                                const std::vector<uint64_t*>& bufferSpecial_c1, const std::vector<Stream*>& external_s,
                                std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>>& signal,
                                std::vector<std::atomic_uint64_t*>& thread_stop);
    void broadcastLimb0_mgpu();
    void doubleRescaleMGPU(LimbPartition& partition);
    void scaleByP();

    void modupMGPU(LimbPartition& aux, const std::vector<uint64_t*>& bufferGather_,
                   std::vector<std::atomic_uint64_t*>& thread_stop, std::vector<Stream*>& external_s);

    void multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1,
                          const LimbPartition& in, const LimbPartition& aux);
    static void multScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                         const std::vector<std::vector<unsigned long int>>& vector,
                                         const std::vector<std::vector<unsigned long int>>& vector_shoup, int stride,
                                         double usage);

    static void addScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                        const std::vector<std::vector<unsigned long int>>& vector, int stride,
                                        double usage);

    static void multPtBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                     int stride, double usage);

    static void addBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                  int stride, double usage, bool sub, bool exta, bool extb);

    // part: -1 = regular + special limbs (default), 0 = regular limbs [limb_begin, +limb_count),
    // 1 = special limbs [limb_begin, +limb_count); limb_count < 0 = all of that part.
    static void LTdotProductPtBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                    const std::vector<LimbPartition*>& pt, int bStep, int gStep, int stride,
                                    double usage, bool ext, int part = -1, int limb_begin = 0, int limb_count = -1);

    // acc0 += Σ a0[j]·b0[j]; acc1 += Σ a0[j]·b1[j]+a1[j]·b0[j]; acc2 = Σ a1[j]·b1[j].
    // One binomialMultAccum_ launch per partition (FHE_LANE_BATCH phase 2).
    static void binomialMultAccumBatch(LimbPartition& acc0, LimbPartition& acc1, LimbPartition& acc2,
                                       const std::vector<const LimbPartition*>& a0,
                                       const std::vector<const LimbPartition*>& a1,
                                       const std::vector<const LimbPartition*>& b0,
                                       const std::vector<const LimbPartition*>& b1);

    static void fusedHoistedRotateBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                        const std::vector<LimbPartition*>& ksk_a,
                                        const std::vector<LimbPartition*>& ksk_b, const std::vector<int>& indexes,
                                        int n, int stride, double usage, bool c0_modup);

    Stream& getS() const { return const_cast<LimbPartition*>(this)->s; }
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_LIMBPARTITION_CUH
