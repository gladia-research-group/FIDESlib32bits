//
// Created by carlosad on 21/10/24.
//

#ifndef GPUCKKS_NTTHELPER_CUH
#define GPUCKKS_NTTHELPER_CUH

namespace FIDESlib {
    template<typename T>
    __device__ __inline__ void swap(T &a, T &b) {
        T c = a;
        a = b;
        b = c;
    }

#define A(i) \
 (((T *) buffer) + 2 * blockDim.x * (i) )

// Shared-memory PARITY SWIZZLE for the u32 NTT/INTT rows: logical element e = 4g+r of row i lives at
// pos = 4*quad + lane with quad = Gray(g) [^ row term at level 2], which keeps every 4-element group
// contiguous/16-B aligned (int4 accesses survive) and removes the butterfly bank conflicts. u64 rows are identity.
#ifndef FIDESLIB_NTT_SWIZZLE
#define FIDESLIB_NTT_SWIZZLE 1
#endif
#if FIDESLIB_NTT_SWIZZLE < 0 || FIDESLIB_NTT_SWIZZLE > 2
#error "FIDESLIB_NTT_SWIZZLE must be 0 (off), 1 (Gray, default) or 2 (Gray + row term)"
#endif

// SPLIT FORM — this is what makes the swizzle affordable. pos(i,e) factors exactly into a
// row-invariant part and a per-row part, because the quad term lives in bits >=2 and the lane
// terms in bits 0..1, so the carries never interact and the sum becomes an XOR:
//     pos(i,e) = swz_base(e) ^ swz_row(i)
// In every hot loop `e` is loop-invariant across the unrolled i-loop, so swz_base is computed
// ONCE per stage and swz_row(i) folds into an immediate. A FUSED form cost +23.7 % instructions
// and ate the entire conflict win; do not re-fuse it.
//
// swz_base = the Gray code, e ^ (e>>1). Two ops. It is the optimum of THREE independent searches:
// the cute::Swizzle<B,M,S> family (it IS Swizzle<5,0,1>), a 2104-map chess-style magic-multiplier
// search, and an exhaustive Pareto sweep of unit lower-triangular GF(2) maps (9.09 % at cost 2,
// nothing at cost 3-9 beats it). The constraint: each butterfly stage reads one hyperplane
// {g_s = v} of the quad cube, so a map that fixes ALL stages needs kernel exactly {0000,1111}.
template <typename T>
__device__ __forceinline__ int swz_base(const int e) {
    if constexpr (FIDESLIB_NTT_SWIZZLE >= 1 && sizeof(T) == 4) {
        return e ^ (e >> 1);
    } else {
        return e;
    }
}

// The ROW term is the only part with a PER-ACCESS cost (one LOP3 on every shared access). It buys
// the last 9.09 % -> 0 % — the transposed accesses, which differ only in the ROW and so cannot be
// separated by anything row-invariant — and measured a net LOSS. Off unless FIDESLIB_NTT_SWIZZLE=2.
template <typename T>
__device__ __forceinline__ int swz_row(const int i) {
    if constexpr (FIDESLIB_NTT_SWIZZLE == 2 && sizeof(T) == 4) {
        return (4 * (i & 7)) ^ ((i >> 2) & 1);
    } else {
        return 0;
    }
}

template <typename T>
__device__ __forceinline__ int swz_pos(const int i, const int e) {
    return swz_base<T>(e) ^ swz_row<T>(i);
}

// int4 slot (quad index) holding the logical quad that contains element e of row i.
template <typename T>
__device__ __forceinline__ int swz_quad(const int i, const int e) {
    return swz_pos<T>(i, e) >> 2;
}

// The lane permutation code for that quad. NOTE this is NOT a plain XOR under Gray — it is
// gray2(r) ^ c with gray2 = [0,1,3,2]; swz_perm4 below consumes `c` and applies both halves.
// e is masked to its quad base first: c is the permutation of the QUAD, not the physical lane of
// e itself, and those coincide only when e is already quad-aligned. Returns 0 on the u64 path.
template <typename T>
__device__ __forceinline__ int swz_lx(const int i, const int e) {
    return swz_pos<T>(i, e & ~3) & 3;
}

// Reorder the four register lanes of an int4 so logical lane r lands at physical lane
// gray2(r) ^ c. Inverting: out.lane[l] = v.lane[gray2(l ^ c)].
__device__ __forceinline__ int4 swz_perm4(int4 v, [[maybe_unused]] const int c) {
#if FIDESLIB_NTT_SWIZZLE == 0
    return v;                                   // must be the exact identity, not a lane shuffle
#elif FIDESLIB_NTT_SWIZZLE == 1
    // Without the row term c is only 0 or 2, so this is a FIXED 2<->3 swap (free — a static
    // reorder) composed with a conditional half-swap: 4 SEL.
    return c ? make_int4(v.w, v.z, v.x, v.y) : make_int4(v.x, v.y, v.w, v.z);
#else
    // With the row term c spans 0..3: a conditional base perm, then a conditional half-swap. 8 SEL.
    const int4 b = (c & 1) ? make_int4(v.y, v.x, v.z, v.w) : make_int4(v.x, v.y, v.w, v.z);
    return (c & 2) ? make_int4(b.z, b.w, b.x, b.y) : b;
#endif
}

// Swizzled scalar access to logical element `e` of shared row `i`.
#define AS(i, e) (A(i)[FIDESlib::swz_base<T>(e) ^ FIDESlib::swz_row<T>(i)])

// WARP-SHUFFLE BUTTERFLIES: the last NTT_SHFL_STAGES forward stages (m = 16..1) and the first inverse
// ones are intra-warp (a warp covers one aligned 64-element block), so they run in registers with one
// __shfl_xor per stage; the exchange is an involution, so one helper serves both directions. 0 = shared memory.
#ifndef FIDESLIB_NTT_WARP_SHFL
#define FIDESLIB_NTT_WARP_SHFL 1
#endif
#if FIDESLIB_NTT_WARP_SHFL < 0 || FIDESLIB_NTT_WARP_SHFL > 1
#error "FIDESLIB_NTT_WARP_SHFL must be 0 (off) or 1 (on)"
#endif

// Number of trailing stages run in registers, and the block size below which the path is
// disabled (one exchange spans lane bits 0..3, so the warp must be full).
#define NTT_SHFL_STAGES 5

// DIAGNOSTIC (produces incorrect results by design): replace the middle-scale twiddle of the 4-step
// transform with a block-uniform constant, to bound what any cheaper twiddle scheme could save.
#ifndef FIDESLIB_NTT_TWIDDLE_ABLATE
#define FIDESLIB_NTT_TWIDDLE_ABLATE 0
#endif

// DIAGNOSTIC (produces incorrect results by design): replace the transposed stage-1 load / final store
// indices with a linear index (same bytes, no address arithmetic), to bound a copy-engine rewrite.
#ifndef FIDESLIB_NTT_TRANSPOSE_ABLATE
#define FIDESLIB_NTT_TRANSPOSE_ABLATE 0
#endif

// ABLATION ONLY (cuTraNTT incomplete-NTT bound): skip the SKIP_LOW smallest-span stages — the last forward / first inverse
// warp-shuffle stages. Results are WRONG for any value > 0; it prices what an incomplete NTT could save in the transform.
#ifndef FIDESLIB_NTT_SKIP_LOW
#define FIDESLIB_NTT_SKIP_LOW 0
#endif

// EXTENDED ON-THE-FLY TWIDDLES at the middle-scale site: exp = block_pos*br_j is affine in the row i,
// so W(exp_i) = W(blockIdx.x*M*br_j) * W(br_j)^i — one base value per k plus a running modular multiply
// replaces M scattered table lookups. Bit-identical (W is a homomorphism). 0 = per-element lookups.
#ifndef FIDESLIB_NTT_EOT
#define FIDESLIB_NTT_EOT 1
#endif
#if FIDESLIB_NTT_EOT < 0 || FIDESLIB_NTT_EOT > 1
#error "FIDESLIB_NTT_EOT must be 0 (off) or 1 (on)"
#endif

template <typename T>
__device__ __forceinline__ T shfl_xor_(const T v, const int lane_mask) {
    if constexpr (sizeof(T) == 8) {
        return (T)__shfl_xor_sync(0xFFFFFFFFu, (unsigned long long)v, lane_mask);
    } else {
        return (T)__shfl_xor_sync(0xFFFFFFFFu, (unsigned int)v, lane_mask);
    }
}

// Move the register-held butterfly pair from the stage whose paired bit is s to the one whose
// paired bit is s-1 (or back — the map is its own inverse). `s_lo` = min(s, s-1) = the LANE bit
// that separates the two partners = log2 of the stage being entered (forward) / left (inverse).
template <typename T>
__device__ __forceinline__ void warp_pair_exchange(T& a0, T& a1, const int tid, const int s_lo) {
    const int t = (tid >> s_lo) & 1;
    const T got = shfl_xor_<T>(t ? a0 : a1, 1 << s_lo);
    if (t)
        a0 = got;
    else
        a1 = got;
}

#define OFFSET_T(i) \
 ((blockDim.x * 2 * M) * blockIdx.x + 2 * blockDim.x * (i) + 2 * threadIdx.x)

#define OFFSET_2T(i) \
 ((blockDim.x * M) * blockIdx.x +  blockDim.x * (i) + threadIdx.x)

    template<typename T, ALGO algo = ALGO_SHOUP>
    __device__ __forceinline__ void CT_butterfly(T &c, T &d, T psi, const int primeid, T shoup_psi = 2) {
        T a = c;
        T b = d;
        if constexpr (algo == 1) {

        } else if constexpr (algo == 2) {
            const uint64_t hi = __umul64hi(b, shoup_psi);
            b = b * psi - hi * C_.primes[primeid];
            d = a - b;
            c = a + b;
        } else if constexpr (algo == 3) {
            b = modmult<algo>(b, psi, primeid, shoup_psi);
            c = modadd(a, b, primeid);
            d = modsub(a, b, primeid);
        } else if constexpr (algo <= 5) {
            // assert(b < primes[primeid]);
            // assert(a < primes[primeid]);
            //  T baux = modmult<0>(b, psi, primeid);
            b = modmult<algo>(b, psi, primeid);
            // assert(psi < primes[primeid]);
            // assert(b < primes[primeid]);
            // assert(b == baux);
            c = modadd(a, b, primeid);
            d = modsub(a, b, primeid);
        }
    }

    template<typename T, ALGO algo = ALGO_SHOUP>
    __device__ __forceinline__ void GS_butterfly(T &c, T &d, T psi, const int primeid, T shoup_psi = 2) {
        T a = c;
        T b = d;
        if constexpr (algo == 1) {
        } else if constexpr (algo == 2) {
            d = a - b;
            c = a + b;
            const uint64_t hi = __umul64hi(d, shoup_psi);
            d = d * psi - hi * C_.primes[primeid];
        } else if constexpr (algo == 3) {
            c = modadd(a, b, primeid);
            b = modsub(a, b, primeid);
            d = modmult<algo>(b, psi, primeid, shoup_psi);
        } else if constexpr (algo <= 5) {
            //      assert(b < primes[primeid]);
            //      assert(a < primes[primeid]);
            c = modadd(a, b, primeid);
            b = modsub(a, b, primeid);
            //   T baux = modmult<0>(b, psi, primeid);
            d = modmult<algo>(b, psi, primeid);
            //     assert(psi < primes[primeid]);
            //     assert(d < primes[primeid]);
            //     assert(d == baux);
        }
    }

}
#endif //GPUCKKS_NTTHELPER_CUH
