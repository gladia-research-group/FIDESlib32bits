//
// Created by carlosad on 16/03/24.
//
#ifndef FIDESLIB_CONSTANTSGPU_CUH
#define FIDESLIB_CONSTANTSGPU_CUH

#include <cinttypes>
#include <vector>
#include "LimbUtils.cuh"

namespace FIDESlib {
constexpr int MAXP = 64;
constexpr int MAXD = 8;

enum version { BARRET, DHEM, NEIL };

constexpr version VERSION = NEIL;

namespace CKKS {
struct Scheme {
    struct Constants;
    struct Global;
};

struct Scheme::Constants {};

struct Scheme::Global {};
}  // namespace CKKS

struct Constants {
    int N, L, K, logN;
    uint64_t type;
    uint64_t primes[MAXP];
    uint64_t prime_better_barret_mu[MAXP];
    /* Lazy-reduction BConv: floor(2^64 / p). Neal_reduce_32 cannot reduce an
     * ACCUMULATOR — its `rx = c >> (qbit-2)` truncates to uint32_t, bounding its input at
     * ~2^56, i.e. exactly one 28x28 product. Deferring the reduction across a BConv dot
     * needs a reducer valid on the whole u64 range, which is what this constant is for.
     * See modreduce_lazy() in ModMult.cuh. 512 B added to the 64 KB bank. */
    uint64_t prime_mu64[MAXP];
    uint32_t prime_bits[MAXP];
    uint8_t table[MAXP * MAXP * 8];

    uint64_t N_shoup[MAXP];
    uint64_t N_inv[MAXP];
    uint64_t N_inv_shoup[MAXP];
    uint64_t root[MAXP];
    uint64_t root_shoup[MAXP];
    uint64_t inv_root[MAXP];
    uint64_t inv_root_shoup[MAXP];

    int dnum;
    uint64_t P[MAXP];
    uint64_t P_shoup[MAXP];
    uint64_t P_inv[MAXP];
    uint64_t P_inv_shoup[MAXP];

    int num_primeid_digit_from[MAXD][MAXP];
    int num_primeid_digit_to[MAXD][MAXP];
    int primeid_digit[MAXP];

    int pos_in_digit[MAXD][MAXP];

    union {
        struct {
            int primeid_partition[MAXD][MAXP];    // Limb primeids
            int primeid_digit_from[MAXD][MAXP];   // Decomp primeids
            int primeid_digit_to[MAXD][MAXP];     // Digit primeids
            int primeid_special_partition[MAXP];  // Special limb primeids
        };
        int primeid_flattened[(MAXD * 3 + 1) * MAXP];
    };
    union {
        CKKS::Scheme::Constants ckks;
        void* none;
    };
};

// The struct lives in CUDA __constant__ memory — a 64 KB bank. If it ever outgrows it
// (e.g. someone raises MAXP), fail at compile time instead of at kernel-launch time.
// NOTE for a future MAXP bump: `uint8_t table[MAXP*MAXP*8]` is DEAD (only read under
// `if constexpr (USING_CONSTANTS_TABLE)` with the constant 0, never written) and is ~77 %
// of the struct — delete it first. `type` is a single uint64_t (1 bit per prime id) and
// must become an array past MAXP=64 (with ISU64/HISU64 and both setters updated).
static_assert(sizeof(Constants) <= 65536, "Constants must fit the 64 KB __constant__ bank");

constexpr int PARTITION(int id, int j) {
    return (offsetof(Constants, primeid_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + id * MAXP +
           j;
}
constexpr int SPECIAL(int d, int j) {
    return (offsetof(Constants, primeid_special_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + j;
}
constexpr int DECOMP(int d, int j) {
    return (offsetof(Constants, primeid_digit_from) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP +
           j;
}
constexpr int DIGIT(int d, int j) {
    return (offsetof(Constants, primeid_digit_to) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP +
           j;
}

struct Global {
    // NTT
    uint64_t type;
    void* psi[MAXP];
    void* inv_psi[MAXP];
    void* psi_no[MAXP];
    void* inv_psi_no[MAXP];
    void* psi_middle_scale[MAXP];
    void* inv_psi_middle_scale[MAXP];
    void* psi_shoup[MAXP];
    void* inv_psi_shoup[MAXP];
    // GPU

    void* psi_ptr[MAXD][MAXP];
    void* inv_psi_ptr[MAXD][MAXP];
    void* psi_no_ptr[MAXD][MAXP];
    void* inv_psi_no_ptr[MAXD][MAXP];
    void* psi_middle_scale_ptr[MAXD][MAXP];
    void* inv_psi_middle_scale_ptr[MAXD][MAXP];
    void* psi_shoup_ptr[MAXD][MAXP];
    void* inv_psi_shoup_ptr[MAXD][MAXP];

    uint64_t root[MAXP];
    uint64_t inv_root[MAXP];

    // Conv
    //uint64_t q_[MAXP][MAXP];
    //uint64_t Q_[MAXP][MAXP];
    uint64_t q_inv[MAXP][MAXP];
    uint64_t QlQlInvModqlDivqlModq[MAXP][MAXP];
    uint64_t aux;

    uint64_t ModDown_pre_scale[MAXP];
    uint64_t ModDown_pre_scale_shoup[MAXP];
    uint64_t ModDown_matrix[MAXP][MAXP];
    uint64_t ModDown_matrix_shoup[MAXP][MAXP];

    uint64_t DecompAndModUp_pre_scale[MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_pre_scale_shoup[MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_matrix[MAXP][MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_matrix_shoup[MAXP][MAXD][MAXP][MAXP];

    union {
        CKKS::Scheme::Global ckks;
        void* none;
    };

    struct Globals {
        void* psi[MAXP];
        void* inv_psi[MAXP];
        void* psi_middle_scale[MAXP];
        void* inv_psi_middle_scale[MAXP];
        void* psi_no[MAXP];
        void* inv_psi_no[MAXP];
        void* psi_shoup[MAXP];
        void* inv_psi_shoup[MAXP];

        //__device__ uint64_t q_[MAXP][MAXP];
        //__device__ uint64_t Q_[MAXP][MAXP];
        uint64_t q_inv[MAXP * MAXP];
        uint64_t QlQlInvModqlDivqlModq[MAXP * MAXP];

        uint64_t ModDown_pre_scale[MAXP];
        uint64_t ModDown_pre_scale_shoup[MAXP];
        uint64_t ModDown_matrix[MAXP * MAXP];
        uint64_t ModDown_matrix_shoup[MAXP * MAXP];

        uint64_t DecompAndModUp_pre_scale[MAXD * MAXP * MAXP];
        uint64_t DecompAndModUp_pre_scale_shoup[MAXD * MAXP * MAXP];
        uint64_t DecompAndModUp_matrix[MAXP * MAXD * MAXP * MAXP];
        uint64_t DecompAndModUp_matrix_shoup[MAXP * MAXD * MAXP * MAXP];

        // Width-typed (u32) shadow copies of the conversion matrices, filled ONLY on
        // all-U32 chains (hC_.type == 0) and read ONLY by the C_.type == 0 fast paths in
        // Conv.cu. On such chains every matrix entry is a residue < 2^28 and every *_shoup
        // companion carries the 2^32-scaled convention, so the narrow copy is exact; the
        // u64 originals stay authoritative for u64/mixed chains and debug dumps. Halves
        // the constant stream pulled through L2 by the two hottest base-conversion
        // kernels (DecompAndModUpConv / ModDown2). Cost: +~16 MB device per GPU
        // (allocated for every chain; unfilled off the U32 path).
        uint32_t ModDown_matrix32[MAXP * MAXP];
        uint32_t ModDown_matrix_shoup32[MAXP * MAXP];
        uint32_t DecompAndModUp_matrix32[MAXP * MAXD * MAXP * MAXP];
        uint32_t DecompAndModUp_matrix_shoup32[MAXP * MAXD * MAXP * MAXP];

        // Fused ModDown + composite rescale (lever A, 2026-10-04), indexed by the TOP prime id `a` of the
        // dropped pair (q_{a-1}, q_a): base B_a = P ∪ {q_{a-1}, q_a} (K+2 <= 16 limbs) -> Q_{< a-1}.
        //   FMD_pre[a*16 + i]            = (B_a / b_i)^{-1} mod b_i
        //   FMD_matrix32[(a*16 + i)*MAXP + j] = (B_a / b_i) mod q_j
        //   FMD_Binv[a*MAXP + j]         = B_a^{-1} mod q_j
        uint64_t FMD_pre[MAXP * 16];
        uint64_t FMD_pre_shoup[MAXP * 16];
        uint32_t FMD_matrix32[MAXP * 16 * MAXP];
        uint32_t FMD_matrix_shoup32[MAXP * 16 * MAXP];
        uint64_t FMD_Binv[MAXP * MAXP];
        uint64_t FMD_Binv_shoup[MAXP * MAXP];
        // exact overflow correction (Halevi-Polyakov-Shoup, CT-RSA 2019): B mod q_j, and 1/b_i for the floating estimate
        uint32_t FMD_Bmod32[MAXP * MAXP];
        double FMD_rb[MAXP * 16];
    };

    Globals* globals[MAXD];

    ~Global();
};

extern __constant__ Constants constants;
//extern Constants host_constants;
//extern Constants host_constants_per_gpu[8];
//extern Global host_global;

#define C_ (constants)
#define G_ Globals
#define TABLE32(i, j) (constants.table[MAXP * 8 * (i) + 4 * (j)])
#define TABLE64(i, j) (constants.table[MAXP * 8 * (i) + 8 * (j)])

#define MODDOWN_MATRIX(i, j) (MAXP * (i) + (j))
#define MODUPIDX_SCALE(i, j, k) (MAXP * MAXP * (i) + MAXP * (j) + (k))
#define MODUPIDX_MATRIX(i, j, k, l) (MAXD * MAXP * MAXP * (i) + MAXP * MAXP * (j) + MAXP * (k) + (l))
#define hC_ (host_constants)
#define hG_ (host_global)

#define ISU64(x) (constants.type & (((uint64_t)1) << (x)))
#define HISU64(x) (host_constants.type & (((uint64_t)1) << (x)))

uint64_t shoup_precomp(uint64_t val, int primeid, Constants& host_constants_);

template <typename Scheme>
__global__ void printConstants();

template <typename Scheme>
std::pair<std::vector<Constants>, std::unique_ptr<Global>> SetupConstants(
    const std::vector<PrimeRecord>& q, const std::vector<std::vector<LimbRecord>>& meta,
    const std::vector<PrimeRecord>& p, const std::vector<LimbRecord>& smeta,
    const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
    const std::vector<std::vector<std::vector<LimbRecord>>>& DIGITmeta, const std::vector<std::vector<int>>& digitGPUid,
    const std::vector<int>& GPUid, const int N, const Scheme& parameters);

}  // namespace FIDESlib
#endif  //FIDESLIB_CONSTANTSGPU_CUH
