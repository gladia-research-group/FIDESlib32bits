#pragma once
// Exact small-integer arithmetic on RNS polynomials (u32 chains, single GPU). A polynomial is "small" when every
// coefficient x satisfies |x| < Q_k/2 for the first k chain primes: true right after ModRaise (|c| <= q0/2) and
// for products of such with plaintexts. For those, the integer coefficient is reconstructed exactly (Garner on
// the first k limbs), optionally divided by a real D (fixed-point reciprocal, round to nearest), and re-reduced
// modulo ANY set of primes — a rescale that keeps the level (level-conserving rescaling of eprint 2025/1403), or
// an exact extension to the special primes (what the GHS key switch of the AKS stage needs instead of an
// approximate base conversion). Used by CKKS/AksKeys.cu.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "CKKS/Plaintext.cuh"

namespace FIDESlib::CKKS {
class RNSPoly;
class ContextData;
class Plaintext;
class Ciphertext;

constexpr int SI_MAXK = 16;    // source limbs
constexpr int SI_MAXW = 16;    // 32-bit words of the reconstructed integer (<= 512 bits)
constexpr int SI_MAXR = 8;     // words of the fixed-point reciprocal (D >= 2^-63)
constexpr int SI_SHIFT = 192;  // q = round(x * R / 2^192), R = round(2^192 / D)
constexpr int SI_MAXOUT = 72;

struct SmallIntTab {
    int k, nout, nw, nr, divide;      // divide: 0 = lift only, 1 = multiply by R and shift
    uint32_t q[SI_MAXK];
    uint32_t qinv[SI_MAXK][SI_MAXK];  // qinv[i][j] = q_j^{-1} mod q_i, j < i
    uint32_t Qk[SI_MAXW], Qkh[SI_MAXW];
    uint32_t R[SI_MAXR];
    uint32_t pout[SI_MAXOUT];
    uint32_t pow32[SI_MAXOUT][SI_MAXW + SI_MAXR];  // (2^32)^w mod pout
};

/// Build the table. D <= 0: no division.
SmallIntTab buildSmallIntTab(const std::vector<uint64_t>& srcPrimes, const std::vector<uint64_t>& outPrimes,
                             long double D);
/// 2^192 / R: the divisor the table really applies (for exact scale bookkeeping).
long double smallIntEffectiveDivisor(const SmallIntTab& t);

/// p (NTT form over Q, level L): allocate the special limbs and fill them with the exact residues of the small
/// integer coefficients reconstructed from the first k limbs; p ends in NTT form over Q+P (isModUp).
void smallIntLiftToSpecial(ContextData& cc, RNSPoly& p, int k);
/// p: every Q limb (and the specials when `withSpecials`) := round(x / D), x the small integer reconstructed
/// from the first k limbs. The level is unchanged. D <= 0: no division (pure re-lift). `srcCoeff`: the first k
/// limbs are already in coefficient form (skips their INTT); the result is always left in NTT form.
void smallIntDivideKeepLevel(ContextData& cc, RNSPoly& p, int k, long double D, bool withSpecials,
                             bool srcCoeff = false);
/// dst (allocated at its level, specials allocated when `withSpecials`) := round(x / D) on every limb, x the small
/// integer reconstructed from the first k limbs of src (NTT form). dst ends in NTT form.
void smallIntDivideFromTo(ContextData& cc, const RNSPoly& src, RNSPoly& dst, int k, long double D, bool withSpecials);
/// A fresh plaintext `shift` composite levels above pt, encoding factor * (its values) (exact small-integer lift of
/// its first 4 limbs combined with the real rescale sf(old) / (factor sf(new)): one rounding of <= 0.5 at scale). Bootstrap plaintexts are
/// constant-allocated and cannot be grown in place, hence the copy.
Plaintext relevelPlaintext(Context& cc_, ContextData& cc, const Plaintext& pt, int shift, double factor = 1.0);
/// ct := round(c * ct) exactly (both polynomials; k = 3 source limbs: the raised ciphertext has |coef| < q0).
/// The message is scaled by c, the scale/level bookkeeping is untouched (what multScalar + rescale achieve, one
/// composite level cheaper); rounding adds <= 0.5 per coefficient (noise ~ (h+1)/2).
void smallIntScalarMultiply(ContextData& cc, Ciphertext& ct, double c);
/// Self-test of the exact division against a host reference on synthetic signed integers |x| < q0 (the raised
/// ciphertext's range) placed on the first 3 limbs; returns the number of mismatching residues (0 = exact).
/// `any` only supplies the context (tests cannot read ContextData fields across TUs).
long smallIntSelfTest(Ciphertext& any, double c);
/// Diagnostic: is p (NTT form) a single small integer per coefficient? Reconstructs each coefficient from the
/// first k limbs on the host (centred) and counts the limbs that disagree; prints the magnitude range. Returns the
/// number of inconsistent (limb, coefficient) pairs.
long smallIntConsistencyCheck(ContextData& cc, const RNSPoly& p, int k, const char* tag);
/// Diagnostic: apply smallIntScalarMultiply to a COPY of ct and compare every residue with the host reference
/// round(c * x) computed from FIDESlib's own INTT'd copy of the original. Returns mismatching residues.
long smallIntScalarCheck(ContextData& cc, const Ciphertext& ct, double c, const char* tag);
/// Diagnostic: re-level `pt` by 0 levels (D = 1) and compare every Q and special limb bitwise with the original
/// (NTT form). Returns mismatching residues; nonzero means the re-level misreads this plaintext's storage.
long relevelSelfTest(Context& cc_, ContextData& cc, const Plaintext& pt, const char* tag);
/// Diagnostics without decryption noise flooding: keep the secret key (EVAL form over Q, from the keygen side) and
/// dump c0 + c1*s in coefficient form (all limbs) with metadata, for an exact host-side CRT + decode.
void loadDiagSecret(ContextData& cc, const std::vector<std::vector<uint64_t>>& skLimbs, const std::vector<uint64_t>& moduli);
bool exactDecryptDump(const Ciphertext& ct, const char* path);
/// Same file format for a plaintext (no key): its c0 in coefficient form.
bool exactPlainDump(const Plaintext& pt, const char* path);
/// Raw polynomials of a ciphertext (no key): <base>-c0.ct and <base>-c1.ct in coefficient form over Q.
bool exactCtPolyDump(const Ciphertext& ct, const char* base);
}  // namespace FIDESlib::CKKS
