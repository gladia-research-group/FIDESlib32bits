//
// Created by carlosad on 27/09/24.
//

#ifndef GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
#define GPUCKKS_ELEMENWISEBATCHKERNELS_CUH

#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int partition, void** l, void** l1, void** l2, void** l3,
                                    void** l4);

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0,
                                  void** bc1, void** in, void** aux);

__global__ void mult1Add2_(const __grid_constant__ int partition, void** l, void** l1, void** l2);

template <typename T>
__global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid);

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init);

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1,
                                      const __grid_constant__ int primeid_init);
template <ALGO algo>
__global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid,
                             const uint64_t* shoup_mu);

__global__ void broadcastLimb0_(void** a);
__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0);
/** COMPOSITESCALING ModRaise: writes, for every limb i in [0, limbs):
 *      a[i][idx] = sum_{k<d} lift_{k->i}( src[k][idx] * qhatinv[k] mod q_k ) * qhat[k*limbs + i] mod q_i
 *  i.e. OpenFHE's ExtendCiphertext CRT recomposition ([a]_{q0q1} = [a*q1^-1]_{q0}*q1 + ...).
 *  src must NOT alias a (the low limbs of a are overwritten) — pass snapshot copies.
 *  qhatinv[k] = (Q0/q_k)^{-1} mod q_k;  qhat[k*limbs+i] = (Q0/q_k) mod q_i. */
// Centred-aggregate CRT lift for coeff plaintexts (d==2). Unlike compositeModRaise_, which
// centres each Garner term against its own prime, this centres the RECONSTRUCTED TOTAL
// against Q0/2 — required when the encoded integer spans the product of the source primes.
__global__ void coeffLiftCentered2_(void** a, void** src, const __grid_constant__ uint64_t q0,
                                    const __grid_constant__ uint64_t q1,
                                    const __grid_constant__ uint64_t q0inv_mod_q1,
                                    const __grid_constant__ uint64_t Qhalf,
                                    const uint64_t* Q0_mod_qi);

__global__ void compositeModRaise_(void** a, void** src, const __grid_constant__ int d, const uint64_t* qhatinv,
                                   const uint64_t* qhat);
__global__ void copy_(void** a, void** b);
/* 4-elements-per-thread copy_. Launch via
 * LimbPartition's launch_copy_limbs helper, which uses it whenever N % 512 == 0 and falls
 * back to scalar copy_ otherwise (neither kernel takes a length, so the grid must cover N
 * exactly). copy_ is retained ONLY for that fallback. */
__global__ void copy_v4_(void** a, void** b);
__global__ void copy1D_(void* a, void* b);
/* TYPE-UNAWARE limb copy — the default copy path. Moves bytes with no width branch, so it drops
 * ISU64 (and its slot-vs-primeid hazard). Tuned in BYTES PER THREAD (16), implemented uniformly
 * as bytes/16 uint4 stores (no per-width vector-type special-casing). Grid {bytes_per_limb/(bytes*128), nlimbs},
 * block 128; the kernel carries no length so the grid must cover the limb exactly. Launch
 * ONLY through this host launcher — the kernel is a template, and launching it from a TU that
 * sees just a declaration yields 'invalid device function'. */
void launchCopyBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int bytes_per_thread);
__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w,
                                   const __grid_constant__ int primeid_init);

/* KSK bit-packing: the two dot kernels are template<int KSK_BITS> PRIVATE to
 * ElemenwiseBatchKernels.cu — KSK_BITS>0 reads the kska/kskb streams as KSK_BITS-bit packed
 * bitstreams (funnelshift unpack, ciphertext-side rows stay dense), 0 is the dense kernel. The
 * width is COMPILE-TIME (instantiated set {27,28}; a runtime width costs registers/occupancy).
 * Cross-TU launches go through these host launchers (a __global__ template launched from a TU
 * that only sees its declaration gets a weak local stub with no device code in that TU's
 * fatbin => 'invalid device function'). ksk_pack_bits==0 selects the dense instantiation;
 * unsupported widths throw. */
/* a_seed: 8 seed words => a REGEN arm regenerates kska in-kernel (caller gates on ksk_seed_set +
 * FIDESLIB_KSK_REGEN); nullptr = stream `a` as before.
 * n16 = N>>4 (the spec's escalation stride; ignored when a_seed is null).
 * regen_shape: 1 = the cooperative 4-slot register arm (the launcher divides grid.x by 4 itself);
 * 2 = stage-A shared-memory arm, diagnostic only. */
void launchFusedDotKSK_2(dim3 grid, dim3 block, cudaStream_t stream, void** out1, void** sout1, void** out2,
                         void** sout2, void*** digits, int num_d, int id, int num_special, int init,
                         int ksk_pack_bits, const uint32_t* a_seed = nullptr, uint32_t n16 = 0,
                         int regen_shape = 0, int chain_type = 0);
/* seeds: DEVICE pointer to n*8 seed words (one 256-bit seed per rotation key, in the same
 * order as the rotation loop) => the stage-B REGEN kernel is launched instead: each thread
 * owns 16 consecutive coefficients and regenerates each (key, digit) ChaCha block once in
 * registers (no smem, no barrier). The caller MUST then pass grid.x = N/(block.x*16) and
 * shmem = 0 (the regen kernel keeps no digit cache). nullptr = stream `a` as before. */
void launchHoistedRotateDotKSK_2(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream, void*** din1, void** c0,
                                 void*** out1, void*** sout1, void*** out2, void*** sout2, int n, const int* indexes,
                                 void*** digits, int num_d, int id, int num_special, int init, void** sc0,
                                 bool c0_modup, int ksk_pack_bits, const uint32_t* seeds = nullptr, uint32_t n16 = 0,
                                 int chain_type = 0);
/* Packs N canonical u32 residues (< 2^bits) into a dense bits-per-coefficient bitstream.
 * out must have ceil(N*bits/32) words + 1 zeroed guard word (for the consumer funnelshift). */
__global__ void packKsk_(uint32_t* out, const uint32_t* in, int N, int bits);

/* Load-time expansion: fill one KSK `a` limb from its 256-bit seed (KskSeedExpand.cuh, SPEC v1),
 * bit-identical to what the patched OpenFHE keygen stored, so this replaces the H2D copy of that
 * limb. Seed passed by value. */
struct KskSeedWords {
    uint32_t k[8];
};
__global__ void expandKskA_(uint32_t* out, KskSeedWords seed, int digit, uint32_t p, uint32_t n16, int N);
/* SPEC v2 (KSKB) twin for the NATIVE_SIZE=64 chain: primes < 2^60, n8 = N>>3. */
__global__ void expandKskA64_(uint64_t* out, KskSeedWords seed, int digit, uint64_t p, uint32_t n8, int N);
__global__ void hoistedRotateDotKSKBatched___(void*** in1, void*** din1, void*** c0, void*** sc0, void*** out1,
                                              void*** sout1, void*** out2, void*** sout2, int n, const int* indexes,
                                              void*** digits, int num_d, int id, int num_special, int init,
                                              bool c0_modup);

__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit,
                              const int n);

// Batched binomial multiply-accumulate over n ciphertext pairs (FHE_LANE_BATCH phase 2):
//   acc0 += Σ_j a0[j]·b0[j]        (loads its initial value — the lane-0 product)
//   acc1 += Σ_j a0[j]·b1[j] + a1[j]·b0[j]
//   acc2  = Σ_j a1[j]·b1[j]        (overwritten — the keyswitch aux input starts empty)
// One pass: every operand element is read exactly once, all n lanes in-thread.
__global__ void binomialMultAccum_(const __grid_constant__ int primeid_init, void** acc0, void** acc1, void** acc2,
                                   void*** a0, void*** a1, void*** b0, void*** b1, const __grid_constant__ int n);

__global__ void binomialMult_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0,
                              void** d1);
__global__ void binomialMultExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2,
                                    void** d0, void** d1);
__global__ void binomialSquare_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2);
__global__ void binomialMultFrom_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** a0, void** a1,
                                  void** d0, void** d1);
__global__ void binomialSquareFrom_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** a0, void** a1);
__global__ void copyScaleByP_(void** dst, void** src, const int primeid_init);
/// DIAGNOSTIC (wrong results): FIDESLIB_ABLATE_KB skips the key-b loads of fusedDotKSKRegen4_.
void setAblateKb(int v);
int ablateKbEnv();
__global__ void binomialSquareExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2);

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n);
__global__ void dotProductLtBatchedPt2___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n);
__global__ void dotProductLtBatchedPt3___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n);

__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init);
__global__ void scaleByP_(void** a, const int primeid_init);

__global__ void add___(void*** a, const int primeid_init, const int n);
void launchScalarMultBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, const uint64_t* b, int primeid_init,
                           const uint64_t* shoup_mu, int bytes_per_thread);
void launchEvalLinearWSumBytes(dim3 grid, dim3 block, cudaStream_t stream, int n, void** a, void*** bs, uint64_t* w,
                               int primeid_init, int bytes_per_thread);
__global__ void add_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void mult_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void add_scalar_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void mult_scalar_reuse_b___(void*** a, void*** b, void*** b_shoup, const int primeid_init, const int n,
                                       const int its);
__global__ void add_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void add_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void sub_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void copy_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);
__global__ void copy_reuse_negative_b___(void*** a, void*** b, const int primeid_init, const int n, const int its);

__global__ void binomialDotProdBatched___(const __grid_constant__ int primeid_init, void*** c0, void*** c1, void*** d0,
                                          void*** d1, void*** c0_out, void*** c1_out, void*** c2_out, int its, int n);

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
