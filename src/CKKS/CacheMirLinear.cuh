// CacheMIR-style BSGS linear packing for FIDESlib CKKS.
//
// Key idea (from the CacheMIR/cachemir-go design):
//   Pack P = numSlots/n replicas of an n-dim token into one ciphertext:
//       slot[p*n + i] = x[p][i]     (p = replica, i = feature index)
//   Then apply a single BSGS linear-transform that simultaneously computes
//   y[p][j] = sum_k W[j][k] * x[p][k]  for all P replicas.
//
//   Baby steps  = bStep = floor(sqrt(n))
//   Giant steps = gStep = ceil(n / bStep)
//   Weight plaintexts encode the "output-minus-input diagonal" of W:
//       pt[g*bStep + b][p*n + i] = W[(i + g*bStep) % n, (i - b + n) % n]
//
// Rotation keys needed: {b : 1 <= b < bStep} union {g*bStep : 1 <= g < gStep}
//
// Supported: square matrices (outDim == inDim == n).
// Rectangular matrices can be handled via zero-padding to the larger dimension.

#pragma once

#include <vector>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {

// ── Precomputed plaintexts for one weight matrix ──────────────────────────────
struct CacheMirLinearPrecomp {
    int n;        ///< matrix dimension  (square: inDim = outDim = n)
    int bStep;    ///< number of baby steps
    int gStep;    ///< number of giant steps
    int numSlots; ///< total ciphertext slots (= P * n)

    /// Weight diagonals pt[d] for d = 0 .. n-1, packed for GPU.
    /// Stored in row-major order: d = g * bStep + b.
    std::vector<Plaintext> weights;

    CacheMirLinearPrecomp() = default;
    ~CacheMirLinearPrecomp() = default;
    CacheMirLinearPrecomp(const CacheMirLinearPrecomp&)            = delete;
    CacheMirLinearPrecomp& operator=(const CacheMirLinearPrecomp&) = delete;
    CacheMirLinearPrecomp(CacheMirLinearPrecomp&&) noexcept            = default;
    CacheMirLinearPrecomp& operator=(CacheMirLinearPrecomp&&) noexcept = default;
};

// ── Rotation indices required for EvalRotateKeyGen ───────────────────────────
/// Returns all rotation amounts needed by cacheMirLinear for a given (n, bStep).
/// Pass this list to OpenFHE's EvalRotateKeyGen before using cacheMirLinear.
std::vector<int> getCacheMirRotationIndices(int n, int bStep);

// ── Weight encoding ───────────────────────────────────────────────────────────
/// Encode a square weight matrix W [n x n] for CacheMIR BSGS packing.
/// @param gctx     FIDESlib GPU context.
/// @param context  OpenFHE CPU context (used for plaintext encoding).
/// @param W        Weight matrix, row-major: W[outRow][inCol].
/// @param n        Matrix dimension.
/// @param bStep    Baby-step size; use floor(sqrt(n)) for optimal depth/KS cost.
/// @param level    Ciphertext level at which the linear layer will be applied.
CacheMirLinearPrecomp encodeCacheMirWeights(
    FIDESlib::CKKS::Context&                            gctx,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&         context,
    const std::vector<std::vector<double>>&              W,
    int n,
    int bStep,
    int level);

// ── Homomorphic linear transform ─────────────────────────────────────────────
/// Apply the CacheMIR linear transform: result = W * x (for all packed replicas).
/// @pre  x has P = numSlots/n replicas of an n-dim token, i.e.
///       decrypted_slot[p*n + i] == token[i] for all p.
/// @pre  Rotation keys for getCacheMirRotationIndices(n, bStep) have been generated.
/// @return New ciphertext with result[p*n + j] = sum_k W[j][k] * x[p][k].
Ciphertext cacheMirLinear(Ciphertext& x, const CacheMirLinearPrecomp& precomp);

// ── Plaintext reference ───────────────────────────────────────────────────────
/// Plaintext linear transform (for testing / comparison).
std::vector<double> cacheMirLinearPlaintext(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& W,
    int n, int numSlots);

// ═══════════════════════════════════════════════════════════════════════════════
// Interleaved Replicated Packing  (CacheMIR paper §4.1, Algorithm 1)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Layout:  slot[K*i + j] = x[(i + j) mod d]   where K = numSlots / d
//
// Preprocess (log K rotations+adds, depth 0):
//   Expand one copy of x (in slots 0..d-1) into K interleaved replicas.
//
// Multiply-accumulate (⌈d/K⌉ rotations, depth 1):
//   Apply ⌈d/K⌉ diagonal plaintexts with rotation amount = K per step.
//   With BSGS (bStep_inner = floor(sqrt(d/K))):
//     baby steps: 0, K, 2K, ..., (bStep_inner-1)*K
//     giant steps: 0, bStep_inner*K, 2*bStep_inner*K, ...
//
// Reduce (log K rotations+adds, depth 0):
//   Tree-sum K partial outputs into one output replica; broadcast if needed.
//
// Rotation keys required: getCacheMirInterleavedRotationIndices(d, numSlots)
//   Count: O(2*log(numSlots/d) + 2*sqrt(d*d/numSlots))
//
// Note: when d*d >= numSlots (i.e. K <= d), interleaved packing offers fewer
// rotation keys than replicated packing (which needs O(2*sqrt(d)) keys).
// For GPT-2 d=768, numSlots=32768: replicated needs ~2*28=56 keys,
// interleaved needs ~2*log(43)+2*sqrt(18)≈ ~18 keys.
// ═══════════════════════════════════════════════════════════════════════════════

struct CacheMirInterleavedPrecomp {
    int d;        ///< vector / matrix dimension
    int K;        ///< numSlots / d  (number of interleaved replicas)
    int bStep;    ///< baby-step count: floor(sqrt(ceil(d/K)))
    int gStep;    ///< giant-step count: ceil(ceil(d/K) / bStep)
    int numSlots; ///< total ciphertext slots

    /// Weight diagonals, indexed by [g*bStep + b], encoded for interleaved input.
    std::vector<Plaintext> weights;

    CacheMirInterleavedPrecomp() = default;
    ~CacheMirInterleavedPrecomp() = default;
    CacheMirInterleavedPrecomp(const CacheMirInterleavedPrecomp&)            = delete;
    CacheMirInterleavedPrecomp& operator=(const CacheMirInterleavedPrecomp&) = delete;
    CacheMirInterleavedPrecomp(CacheMirInterleavedPrecomp&&) noexcept            = default;
    CacheMirInterleavedPrecomp& operator=(CacheMirInterleavedPrecomp&&) noexcept = default;
};

/// Rotation indices for interleaved packing linear layer.
/// Register with cc.eval_rotate_key_gen() before calling encodeCacheMirInterleavedWeights.
std::vector<int> getCacheMirInterleavedRotationIndices(int d, int numSlots);

/// Encode W [d x d] for the interleaved replicated packing algorithm.
CacheMirInterleavedPrecomp encodeCacheMirInterleavedWeights(
    FIDESlib::CKKS::Context&                            gctx,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&         context,
    const std::vector<std::vector<double>>&              W,
    int d,
    int numSlots,
    int level);

/// Apply interleaved linear transform.
/// @pre  x has K interleaved replicas: slot[K*i + j] = x_input[(i+j) mod d].
///       Encrypt with encrypt_interleaved() or preprocess with cacheMirPreprocess().
/// @return Ciphertext with same interleaved layout: slot[K*i+j] = (W @ x_input)[(i+j)%d].
Ciphertext cacheMirInterleavedLinear(Ciphertext& x, const CacheMirInterleavedPrecomp& precomp);

/// Preprocess a replicated-layout ciphertext into interleaved layout.
/// Input:  slot[p*d + i] = x[i]   (standard replicated packing, P=K replicas)
/// Output: slot[K*i + j] = x[(i+j) mod d]
/// Cost: log2(K) rotations + adds; depth 0.
Ciphertext cacheMirPreprocess(Ciphertext& x_replicated, int d, int numSlots);

/// Reduce interleaved-layout output into a single replicated output.
/// Sums K interleaved partial results: output slot[p*d+j] = (W @ x)[j].
/// Cost: log2(K) rotations + adds; depth 0.
Ciphertext cacheMirReduce(Ciphertext& y_interleaved, int d, int numSlots);

/// Encrypt a vector in interleaved format directly (no HE ops needed).
/// Fills: slot[K*i + j] = x[(i + j) mod d] for all i, j.
std::vector<double> makeInterleavedPlaintext(
    const std::vector<double>& x, int d, int numSlots);

/// Plaintext reference for interleaved linear transform.
std::vector<double> cacheMirInterleavedLinearPlaintext(
    const std::vector<double>& x_interleaved,
    const std::vector<std::vector<double>>& W,
    int d, int numSlots);

}  // namespace FIDESlib::CKKS
