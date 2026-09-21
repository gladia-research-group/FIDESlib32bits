//
// Created by carlosad on 24/03/24.
//

#ifndef FIDESLIB_MODMULT_CUH
#define FIDESLIB_MODMULT_CUH

#include <cassert>
#include <cinttypes>
#include "CKKS/forwardDefs.cuh"
#include "ConstantsGPU.cuh"
namespace FIDESlib {
/**
    The main idea is to implement a compiletime/runtime switch for different modular reduction implementations,
    as tweak factors can be stored on constant memory, we do not need pass them in the parameter list and the
    rest of the implementation is agnostic to this variability.
*/

// ------------------------------------ BASIC MODULAR MULT KERNELS ----------------------------------------

/** Inplace element-wise modular mult of an array: a_i = a_i * b_i % p */
template <typename T, ALGO algo = DEFAULT_ALGO>
__global__ void mult_(T* a, const T* b, const int primeid);

/** Element-wise modular mult of an array: a_i = b_i * c_i % p */
template <typename T, ALGO algo = DEFAULT_ALGO>
__global__ void mult_(T* a, const T* b, const T* c, const int primeid);

/** Inplace scalar modular mult of an array: a_i = a_i * b % p */
template <typename T, ALGO algo = DEFAULT_ALGO>
__global__ void scalar_mult_(T* a, const T b, const int primeid, const T shoup_mu = 0);

// ------------------------------------ INLINEABLE GPU MODULAR MULT KERNELS -------------------------------

/** 32-bit modular mult, to be inlined inside more complex kernels. */
template <ALGO algo = DEFAULT_ALGO>
__forceinline__ __device__ uint32_t modmult(const uint32_t a, const uint32_t b, const int primeid,
                                            const uint32_t shoup_b = 0);

/** 64-bit modular mult, to be inlined inside more complex kernels. */
template <ALGO algo = DEFAULT_ALGO>
__forceinline__ __device__ uint64_t modmult(const uint64_t a, const uint64_t b, const int primeid,
                                            const uint64_t shoup_b = 0);

/** 64-bit integer improved Barret modular multiplication implementation. (p < 2^62) */
__forceinline__ __device__ uint64_t Neal_mult_64(const uint64_t op1, const uint64_t op2, const uint64_t mu,
                                                 const uint64_t prime, const uint32_t qbit) {
    /*
        //        assert(op1 < prime);
//        assert(op2 < prime);

        const uint64_t rx = __umul64hi(op1 << (64 - qbit), op2 << (VERSION == BARRET ? 1 : 2));
        uint64_t quot = __umul64hi(rx, mu << (64 - (VERSION == BARRET ? 1 : (VERSION == DHEM ? 5 : 3))- qbit));
        uint64_t rem = op1 * op2 - quot * prime;

      //  const__uint128_t rx = (uint128_t) op1 * op2;
      //  uint64_t quot = __umul64hi(rx >> (qbit - 1), mu << (64 - (VERSION == BARRET ? 1 : (VERSION == DHEM ? 5 : 3))- qbit));
      //  uint64_t rem = ((uint64_t)rx) - quot * prime;

        if constexpr(VERSION == BARRET) rem = rem - 2 * prime * (rem >= 2 * prime);
        rem = rem - prime * (rem >= prime);
 //       assert(rem < prime);
        return rem;

*/
    __uint128_t c = (__uint128_t)op1 * op2;
    uint64_t rx = c >> (qbit - 2);
    uint64_t rb = __umul64hi(rx << (62 - qbit), mu) >> 1;
    rb *= prime;
    uint64_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/** 32-bit integer improved Barret modular multiplication implementation. (p < 2^30) */
__forceinline__ __device__ uint32_t Neal_mult_32(const uint32_t op1, const uint32_t op2, const uint32_t mu,
                                                 const uint32_t prime, const uint32_t& qbit) {
    uint64_t c = (uint64_t)op1 * op2;
    uint32_t rx = c >> (qbit - 2);
    uint32_t rb = __umulhi(rx << (30 - qbit), mu) >> 1;
    rb *= prime;
    uint32_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/**
     * 64-bit integer Shoup modular multiplication.
     * From paper: Modular SIMD arithmetic in Mathemagix: Algorithm 8
     * Requires: psi = op2 * 2^64 / prime
     *           prime < 2^63
     * Output: op1 * op2 % prime
     */
__forceinline__ __device__ uint64_t Shoup_mult_64(const uint64_t op1, const uint64_t op2, const uint64_t psi,
                                                  const uint64_t prime) {
    uint64_t c = __umul64hi(op1, psi);
    uint64_t c_lo = op1 * op2 - c * prime;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/**
     * 32-bit integer Shoup modular multiplication.
     * From paper: Modular SIMD arithmetic in Mathemagix: Algorithm 8
     * Requires: psi = op2 * 2^32 / prime
     *           prime < 2^31
     * Output: op1 * op2 % prime
     */
__forceinline__ __device__ uint32_t Shoup_mult_32(const uint32_t op1, const uint32_t op2, const uint32_t psi,
                                                   const uint32_t prime) {
    uint32_t c = __umulhi(op1, psi);
    uint32_t c_lo = op1 * op2 - c * prime;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/** Helper function that computes the higher 42 bits of a 106 bit wide multiplication of two 53-bit integers.
     *  Note: it may actually compute up to 53 bits but the lower 11 are discarded.
     */
__forceinline__ __device__ uint64_t fp64himult(const uint64_t a, const uint64_t b) {
    /// Estructura fp64: (sign) 63 | (exp) 62 - 52 | (mantissa) 51 - 0
    const int lza = __clzll(a);
    const int lzb = __clzll(b);
    uint64_t aux_a = a << (lza - 11);  // signo + exponente - 1
    uint64_t aux_b = b << (lzb - 11);
    // eliminar bit implicito
    aux_a &= ~(1ul << 52);
    aux_b &= ~(1ul << 52);
    // Poner exponentes a 0.
    aux_a |= 0x3FF0000000000000;
    aux_b |= 0x3FF0000000000000;
    // Conversión literal de los datos
    const double da = *((double*)&aux_a), db = *((double*)&aux_b);
    // Multiplicamos double sin redondeo (redondeo hacia abajo)
    const double dc = __dmul_rd(da, db);
    const uint64_t c = *((uint64_t*)&dc);
    // Eliminamos exponente y signo de la interpretación entera.
    uint64_t res = (c & 0x000FFFFFFFFFFFFF) | (1ul << 52);

    // Nos quedamos con los 42 bits más significativos (los 64 bits bajos se eliminan) con el shift.
    // Si el exponente está a uno, hacemos un shift menos.
    //res >>= -10 + lza + lzb - ((c & (1ul << 62)) != 0); // 52 - (53*2 - 64) + (lza - 11) + (lzb - 11);

    // Nos quedamos con los 53 bits más significativos (los 53 bits bajos se eliminan) con el shift.
    // Si el exponente está a uno, hacemos un shift menos.
    res >>= -21 + lza + lzb - ((c & (1ul << 62)) != 0);  // 52 - (53*2 - 53) + (lza - 11) + (lzb - 11);

    //uint64_t good_res = __umul64hi(a, b << 11);
    //if(__umul64hi(a, b << 11) != res)
    //  printf("obj: %p, obtained: %p, a: %A, b: %A, c: %A, lza: %d, lzb: %d\n", good_res, res, da, db, dc, lza, lzb);
    //assert(__umul64hi(a, b << 11) == res);
    return res;
}

/** Helper function that computes the higher 42 bits of a 106 bit wide multiplication of two 53-bit integers.
     *  This function leverages CUDA's type conversion intrinsics/operations which operate with higher throughput
     *  on server architectures.
     *  Note: it may actually compute up to 53 bits but the lower 11 are discarded.
     */

__forceinline__ __device__ uint64_t fp64himult_ver2(const uint64_t a, const uint64_t b) {

    const double da = __ll2double_rz(a), db = __ll2double_rz(b);
    // Multiplicamos double sin redondeo (redondeo hacia abajo)

    const double dc = __dmul_rz(da, db);
    // Divsión rápida entre 2 ^ 53
    uint64_t aux = __double_as_longlong(dc);
    aux -= (53ul << 52);

    const uint64_t res = __double2ll_rz(__longlong_as_double(aux));

    return res;
}

/** 53-bit integer improved Barret modular multiplication implementation leveraging fp84 computation. (p < 2^51) */
__forceinline__ __device__ uint64_t Neal_mult_53(const uint64_t op1, const uint64_t op2, const uint64_t mu,
                                                 const uint64_t prime, const uint32_t qbit) {
    //        assert(op1 < prime);
    //        assert(op2 < prime);
    //        assert(mu < (1ll << 53));
    //const uint64_t aux = fp64himult(op1, op2);
    //assert(aux == __umul64hi(op1, op2 << 11));
    /*
        const uint64_t rx = fp64himult(op1 << (53 - qbit), op2 << (VERSION == BARRET ? 1 : 2));
        uint64_t quot = fp64himult(rx, mu << (53 - (VERSION == BARRET ? 1 : (VERSION == DHEM ? 5 : 3))- qbit));
        uint64_t rem = (((op1 * op2)) - ((quot * prime))) & 0x001FFFFFFFFFFFFF;

        if constexpr(VERSION == BARRET) rem = (rem - 2 * prime * (rem >= 2 * prime));
        rem = (rem - prime * (rem >= prime));

  //      assert(rem < prime);
        return rem;
  */

    const uint64_t rx = fp64himult_ver2(op1 << (53 - qbit), op2 << 2);
    uint64_t rb = fp64himult_ver2(rx << (51 - qbit), mu) >> 1;
    rb *= prime;
    uint64_t c_lo = op1 * op2;
    c_lo -= rb;
    c_lo &= 0x001FFFFFFFFFFFFFU;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

template <ALGO algo>
__forceinline__ __device__ uint64_t modmult(const uint64_t a, const uint64_t b, const int primeid,
                                            const uint64_t shoup_b) {
    const uint64_t p = C_.primes[primeid];
    uint64_t res{0};
    if constexpr (algo >= 1 && algo <= 2) {
    } else if constexpr (algo == 3) {
        res = Shoup_mult_64(a, b, shoup_b, p);
    } else if constexpr (algo == 4) {
        res = Neal_mult_64(a, b, C_.prime_better_barret_mu[primeid], p, C_.prime_bits[primeid]);
    } else if constexpr (algo == 5) {
        res = Neal_mult_53(a, b, C_.prime_better_barret_mu[primeid], p, C_.prime_bits[primeid]);
    } else {
        res = (__uint128_t)a * b % p;
    }
    return res;
}

// Forward declaration: the wide-prime guard in modmult() below needs this, and its definition
// sits with the other reducers further down.
__forceinline__ __device__ uint32_t modreduce_lazy(const uint64_t a, const int primeid);

template <ALGO algo>
__device__ uint32_t modmult(const uint32_t a, const uint32_t b, const int primeid, const uint32_t shoup_b) {
    const uint32_t p = C_.primes[primeid];
    uint32_t res{0};
    if constexpr (algo >= 1 && algo <= 2) {

    } else if constexpr (algo == 3) {
        res = Shoup_mult_32(a, b, shoup_b, p);
    } else if constexpr (algo == 4) {
        // Same wide-prime guard as modreduce: Neal_mult_32 reduces its own a*b product with the
        // identical ~2^56-bounded shape, so a 30x30 product (2^60) is out of envelope. Reduce the
        // exact 64-bit product instead — modreduce_lazy is exact over the whole u64 range.
        // Prefer SHOUP on wide primes when the caller supplied a precomputed psi (the 4-arg
        // form: NTT twiddles, key/plaintext constants) — it is two multiplies and needs only
        // prime < 2^31, so it is both cheaper and in-envelope at 30 bits. It is NOT a blanket
        // substitute: `shoup_b` DEFAULTS TO 0, and the 3-arg call sites (NTT.cu eot_tw x
        // eot_step, NTThelper, Rescale's b==1 pure-reduction use) multiply two runtime values
        // with no psi to precompute — Shoup against a zero psi is silently wrong. Those take
        // the exact 64-bit-product reduction instead.
        res = (C_.prime_bits[primeid] > 28)
                  ? (shoup_b ? Shoup_mult_32(a, b, shoup_b, p)
                             : modreduce_lazy((uint64_t)a * (uint64_t)b, primeid))
                  : Neal_mult_32(a, b, C_.prime_better_barret_mu[primeid], p, C_.prime_bits[primeid]);
    } else {
        res = (uint64_t)a * (uint64_t)b % (uint64_t)p;
    }
    return res;
}

//--------------------------------------- MODULAR REDUCTION ------------------------------------------------

/** 64-bit integer improved Barret modular reduction implementation. (p < 2^62) */  // TODO test
__forceinline__ __device__ uint64_t Neal_reduce_64(__uint128_t c, const uint64_t mu, const uint64_t prime,
                                                   const uint32_t qbit) {
    /*
    __uint128_t c = (__uint128_t)op1 * op2;
    uint64_t rx = c >> (qbit - 2);
    uint64_t rb = __umul64hi(rx << (62 - qbit), mu) >> 1;
    rb *= prime;
    uint64_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
*/

    /*
    __uint128_t p2 = (__uint128_t)prime * prime;
    for (int i = 10; i > 0; i--) {
        if (c > (p2 << i)) {
            c = c - (p2 << i);
        }
    }
    */

    uint64_t rx = c >> (qbit - 2);
    uint64_t rb = __umul64hi(rx << (62 - qbit), mu) >> 1;
    rb *= prime;
    uint64_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/** 32-bit integer improved Barret modular reduction implementation. (p < 2^30) */  // TODO test
__forceinline__ __device__ uint32_t Neal_reduce_32(const uint64_t c, const uint32_t mu, const uint32_t prime,
                                                   const uint32_t& qbit) {
    uint32_t rx = c >> (qbit - 2);
    uint32_t rb = __umulhi(rx << (30 - qbit), mu) >> 1;
    rb *= prime;
    uint32_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

/* Lazy-reduction BConv: exact `a mod p` for ANY a < 2^64, p < 2^32.
 *
 * Why not modreduce<>/Neal_reduce_32: that reducer does `uint32_t rx = c >> (qbit-2)` and then
 * `rx << (30-qbit)`, so its input is bounded at roughly 2^56 — precisely one 28x28 product. It
 * is the right tool for reduce-after-every-multiply and the wrong one for an accumulator, which
 * is what makes the lazy BConv shape need its own reducer rather than reusing that path.
 *
 * mu = floor(2^64/p) gives q_hat = floor(a*mu / 2^64) in {floor(a/p)-1, floor(a/p)}: writing
 * mu = (2^64 - e)/p with 0 <= e < p, a*mu/2^64 = a/p - a*e/(p*2^64) and a*e/(p*2^64) < 1 for
 * a < 2^64. So the remainder lands in [0, 2p) and ONE conditional subtract finishes it. The
 * result is the exact canonical residue, so callers stay bit-exact against the eager form. */
__forceinline__ __device__ uint32_t modreduce_lazy(const uint64_t a, const int primeid) {
    const uint64_t p = C_.primes[primeid];
    const uint64_t r = a - __umul64hi(a, C_.prime_mu64[primeid]) * p;
    return (uint32_t)(r >= p ? r - p : r);
}

template <ALGO algo>
__device__ uint32_t modreduce(const uint64_t a, const int primeid) {
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu \n", primeid, p);
    uint32_t res{0};
    if constexpr (algo >= 0 && algo <= 2) {
        res = a % C_.primes[primeid];
    } else if constexpr (algo == 3 || algo == 4) {
        // WIDE-PRIME GUARD. Neal_reduce_32's `rx = c >> (qbit-2)` / `rx << (30-qbit)`
        // shape bounds its INPUT at ~2^56 — exactly one 28x28 product. A chain with primes wider
        // than 28 bits (FIRST_MOD_BITS=60 on d=2 gives two 30-bit primes) feeds it up to 2^60 and
        // it returns silent garbage. modreduce_lazy is exact for ANY a < 2^64 and is already the
        // reducer the lazy-BConv path uses, so route only the wide primes there. The branch is
        // warp-uniform (primeid is uniform per block) and touches 2 of ~54 primes.
        res = (C_.prime_bits[primeid] > 28)
                  ? modreduce_lazy(a, primeid)
                  : Neal_reduce_32(a, C_.prime_better_barret_mu[primeid], C_.primes[primeid],
                                   C_.prime_bits[primeid]);
    } else {
        assert("fp64 reduce not implemented" == nullptr);
    }
    return res;
}

template <ALGO algo>
__device__ uint64_t modreduce(const __uint128_t a, const int primeid) {
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu \n", primeid, p);
    if constexpr (algo >= 0 && algo <= 2) {
        return a % (__uint128_t)C_.primes[primeid];
    } else if constexpr (algo == 3 || algo == 4) {
        return Neal_reduce_64(a, C_.prime_better_barret_mu[primeid], C_.primes[primeid], C_.prime_bits[primeid]);
    } else {
        assert("fp64 reduce not implemented" == nullptr);
    }
    return ULLONG_MAX;
}
}  // namespace FIDESlib

#endif  //FIDESLIB_MODMULT_CUH
