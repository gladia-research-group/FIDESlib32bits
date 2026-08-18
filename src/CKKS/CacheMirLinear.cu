// CacheMIR-style BSGS linear packing implementation.
// See CacheMirLinear.cuh for algorithm description.

#include "CKKS/CacheMirLinear.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include <cassert>
#include <cmath>
#include <set>
#include <stdexcept>

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

namespace FIDESlib::CKKS {

// ── helpers ───────────────────────────────────────────────────────────────────

static inline int mod_pos(int v, int n) { return ((v % n) + n) % n; }

// ── rotation indices ──────────────────────────────────────────────────────────

std::vector<int> getCacheMirRotationIndices(int n, int bStep) {
    std::set<int> s;
    int gStep = (n + bStep - 1) / bStep;
    for (int b = 1; b < bStep; ++b)
        s.insert(b);
    for (int g = 1; g < gStep; ++g)
        s.insert(g * bStep);
    return {s.begin(), s.end()};
}

// ── weight encoding ───────────────────────────────────────────────────────────

CacheMirLinearPrecomp encodeCacheMirWeights(
    FIDESlib::CKKS::Context&                   gctx,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const std::vector<std::vector<double>>&     W,
    int n,
    int bStep,
    int level)
{
    assert(n > 0 && bStep > 0 && bStep <= n);
    assert((int)W.size() == n);
    for (auto& row : W) assert((int)row.size() == n);

    int gStep    = (n + bStep - 1) / bStep;
    int numSlots = gctx->N / 2;        // N/2 complex CKKS slots
    int preProc  = numSlots / n;       // number of packed replicas

    if (preProc <= 0)
        throw std::invalid_argument("CacheMirLinear: n > numSlots/2");

    CacheMirLinearPrecomp precomp;
    precomp.n        = n;
    precomp.bStep    = bStep;
    precomp.gStep    = gStep;
    precomp.numSlots = numSlots;
    precomp.weights.reserve(n);  // n = gStep * bStep diagonals (with padding)

    // Encode all n diagonals d = g*bStep + b, d = 0 .. n-1.
    // pt[d][p*n + i] = W[(i + g*bStep) % n, (i - b + n) % n]
    for (int d = 0; d < n; ++d) {
        int g = d / bStep;
        int b = d % bStep;

        std::vector<double> vals(numSlots, 0.0);
        for (int p = 0; p < preProc; ++p) {
            for (int i = 0; i < n; ++i) {
                int row = mod_pos(i + g * bStep, n);
                int col = mod_pos(i - b,         n);
                vals[p * n + i] = W[row][col];
            }
        }

        // Encode at the ciphertext level where the linear layer is applied.
        // OpenFHE convention: level argument = L - ciphertext_level_count.
        auto cpuPt = context->MakeCKKSPackedPlaintext(
            vals, 1, gctx->L - level, nullptr, numSlots);
        auto raw = GetRawPlainText(context, cpuPt);
        precomp.weights.emplace_back(gctx, raw);
    }

    return precomp;
}

// ── homomorphic linear transform ──────────────────────────────────────────────

Ciphertext cacheMirLinear(Ciphertext& x, const CacheMirLinearPrecomp& precomp) {
    Context&    cc_ = x.cc_;
    int bStep       = precomp.bStep;
    int gStep       = precomp.gStep;
    int n           = precomp.n;

    // Ensure input is at noise level 1 before we start.
    if (x.NoiseLevel == 2) x.rescale();

    // ── Baby-step hoisted rotations: ctRot[b] = Rot(x, b) ─────────────────
    // We compute ctRot[b] = Rot(x, b) for b = 0 .. bStep-1 using
    // rotate_hoisted so that the expensive key-switching base decomposition
    // is performed only once.
    std::vector<Ciphertext> ctRot;
    ctRot.reserve(bStep);
    {
        // Allocate bStep destination ciphertexts at the same level as x.
        std::vector<Ciphertext*> ptrs;
        ptrs.reserve(bStep);
        for (int b = 0; b < bStep; ++b) {
            ctRot.emplace_back(cc_);
            ctRot.back().growToLevel(x.getLevel());
            ctRot.back().dropToLevel(x.getLevel());
            ctRot.back().extend(false);
            ptrs.push_back(&ctRot.back());
        }
        // Baby-step rotation amounts: [0, 1, 2, ..., bStep-1]
        std::vector<int> baby_steps(bStep);
        for (int b = 0; b < bStep; ++b) baby_steps[b] = b;

        x.rotate_hoisted(baby_steps, ptrs, /*ext=*/false);
    }

    // ── Giant steps: for each g, accumulate baby-step products ─────────────
    //   tmp_g = sum_{b=0}^{bStep-1} ctRot[b] * pt[g*bStep + b]
    //   Rot(tmp_g, g*bStep) contributes to the output.

    Ciphertext result(cc_);
    bool result_initialized = false;

    for (int g = 0; g < gStep; ++g) {
        Ciphertext tmp(cc_);
        bool tmp_initialized = false;

        for (int b = 0; b < bStep; ++b) {
            int d = g * bStep + b;
            if (d >= n) break;  // last giant step may be partial

            if (!tmp_initialized) {
                // First baby step: copy ctRot[b] into tmp then multiply.
                tmp.multPt(ctRot[b], precomp.weights[d], /*rescale=*/false);
                tmp_initialized = true;
            } else {
                // Subsequent baby steps: fused add-multiply (no rescale yet).
                tmp.addMultPt(ctRot[b], precomp.weights[d], /*rescale=*/false);
            }
        }

        if (!tmp_initialized) continue;  // should not happen

        // Rescale once after all baby-step accumulations.
        tmp.rescale();

        // Giant-step rotation to align this partial sum to output slot positions.
        if (g > 0) {
            tmp.rotate(g * bStep);
        }

        // Accumulate into result.
        if (!result_initialized) {
            result.copy(tmp);
            result_initialized = true;
        } else {
            result.add(tmp);
        }
    }

    if (!result_initialized)
        throw std::runtime_error("CacheMirLinear: empty precomp (n=0?)");

    return result;
}

// ── plaintext reference ───────────────────────────────────────────────────────

std::vector<double> cacheMirLinearPlaintext(
    const std::vector<double>&              x,
    const std::vector<std::vector<double>>& W,
    int n, int numSlots)
{
    int preProc = numSlots / n;
    std::vector<double> out(numSlots, 0.0);

    for (int p = 0; p < preProc; ++p) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                acc += W[j][k] * x[p * n + k];
            out[p * n + j] = acc;
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Interleaved Replicated Packing implementation
// ═══════════════════════════════════════════════════════════════════════════════

// Interleaved layout: slot[K*i + j] = x[(i + j) % d]  where K = numSlots / d.
// Rotation by K advances the x-index by 1:
//   Rot(ct, K)[K*i + j] = ct[K*(i+1)+j (mod numSlots)] = x[(i+1+j) % d]
// So the "effective" cyclic structure has period d with step K in the ciphertext.

std::vector<int> getCacheMirInterleavedRotationIndices(int d, int numSlots) {
    assert(numSlots % d == 0);
    int K      = numSlots / d;            // replicas
    int nDiag  = (d + K - 1) / K;        // number of unique diagonals = ceil(d/K)
    int bStep  = std::max(1, (int)std::floor(std::sqrt((double)nDiag)));
    int gStep  = (nDiag + bStep - 1) / bStep;

    std::set<int> s;
    // Baby steps: rotate by K, 2K, ..., (bStep-1)*K
    for (int b = 1; b < bStep; ++b)
        s.insert(b * K);
    // Giant steps: rotate by bStep*K, 2*bStep*K, ...
    for (int g = 1; g < gStep; ++g)
        s.insert(g * bStep * K);
    // Preprocessing / reduce: rotate by K, 2K, 4K, ..., K*(K/2) (powers of 2 up to K/2)
    for (int step = K; step < numSlots; step *= 2)
        s.insert(step);
    // Also negative-direction reduce rotations: numSlots - step
    for (int step = K; step < numSlots; step *= 2)
        s.insert(numSlots - step);
    return {s.begin(), s.end()};
}

// Encoding: diagonal d_idx of W for interleaved layout.
// For rotation amount r = d_idx * K (d_idx = g*bStep+b):
//   Rot(ct, d_idx*K)[K*i+j] = x[(i + d_idx + j) % d]
// We want y[m] = sum_{d_idx} diagW[d_idx][m] * x[(m - d_idx) % d]
// In interleaved output slot K*i+j we want y[(i+j)%d]:
//   y[(i+j)%d] = sum_{d_idx} diagW[d_idx][(i+j)%d] * Rot(ct, d_idx*K)[K*i+j]
//              = sum_{d_idx} diagW[d_idx][(i+j)%d] * x[(i + d_idx + j) % d]
// Set m=(i+j)%d, k=(m+d_idx)%d:  diagW[d_idx][m] = W[m, (m+d_idx)%d] (forward diagonal)
//
// Wait: y[m] = sum_k W[m,k]*x[k]. With k=(m+d_idx)%d: W[m, (m+d_idx)%d] * x[(m+d_idx)%d].
// But Rot(ct, d_idx*K)[K*i+j] = x[(i+j+d_idx)%d] = x[(m + d_idx)%d]. ✓

CacheMirInterleavedPrecomp encodeCacheMirInterleavedWeights(
    FIDESlib::CKKS::Context&                   gctx,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const std::vector<std::vector<double>>&     W,
    int d,
    int numSlots,
    int level)
{
    assert(numSlots % d == 0);
    assert((int)W.size() == d);
    for (auto& row : W) assert((int)row.size() == d);

    int K     = numSlots / d;
    int nDiag = (d + K - 1) / K;          // ceil(d/K) unique diagonals needed
    int bStep = std::max(1, (int)std::floor(std::sqrt((double)nDiag)));
    int gStep = (nDiag + bStep - 1) / bStep;

    CacheMirInterleavedPrecomp precomp;
    precomp.d        = d;
    precomp.K        = K;
    precomp.bStep    = bStep;
    precomp.gStep    = gStep;
    precomp.numSlots = numSlots;
    precomp.weights.reserve(nDiag);

    // Encode nDiag diagonals.  Diagonal index t = g*bStep + b (t = 0..nDiag-1).
    // pt[t][K*i + j] = W[(i+j)%d, (i+j+t)%d]   (forward diagonal of W)
    for (int t = 0; t < nDiag; ++t) {
        std::vector<double> vals(numSlots, 0.0);
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < d; ++j) {
                int m   = mod_pos(i + j, d);          // output index
                int k   = mod_pos(m + t, d);           // input index after rotation by t
                vals[K * i + j] = W[m][k];
            }
        }
        // Accumulate partial sums for t ≥ K: the same diagonal repeats with period d.
        // All K interleaved replicas carry the same contribution, so the vals above
        // already cover all slots.  No extra replica loop needed.

        auto cpuPt = context->MakeCKKSPackedPlaintext(
            vals, 1, gctx->L - level, nullptr, numSlots);
        auto raw = GetRawPlainText(context, cpuPt);
        precomp.weights.emplace_back(gctx, raw);
    }

    return precomp;
}

// makeInterleavedPlaintext: build the slot vector for encrypting x in interleaved layout.
std::vector<double> makeInterleavedPlaintext(
    const std::vector<double>& x, int d, int numSlots)
{
    assert(numSlots % d == 0);
    int K = numSlots / d;
    std::vector<double> out(numSlots);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < d; ++j)
            out[K * i + j] = x[mod_pos(i + j, d)];
    return out;
}

// cacheMirPreprocess: replicated layout → interleaved layout.
// Input:  slot[p*d + i] = x[i]   (P = K identical replicas)
// Output: slot[K*i + j] = x[(i+j) % d]
// Method: the interleaved layout is a sum of K rotations of the replicated input:
//   sum_{s=0}^{K-1} mask_s * Rot(x_rep, -s*K) ... but masks cost levels.
// Instead we use the observation that Rot(x_interleaved, 1) has the property
// that x_interleaved + Rot(x_interleaved, K) = x_interleaved * 2 (same layout).
// Actually the cheapest rotation-only conversion is:
//   Start from single copy in slots 0..d-1 (not replicated), then "expand"
//   using log(K) doubling steps: ct = ct + Rot(ct, d<<step) won't give interleaved.
// The correct preprocessing uses the replicated form:
//   Interleaved can be obtained from replicated by log(K) rotate-add steps.
//   Step 0: x_rep = [x0..xd-1 | x0..xd-1 | ... | x0..xd-1]  (K copies)
//   After log(K) steps we get the interleaved layout.
//
// Key: Rot(x_rep, 1)[slot j] = x_rep[(j+1) mod N/2] = x[(j+1) mod d].
//   So x_rep + Rot(x_rep, 1) = [x0+x1, x1+x2, ...] — not what we want.
//
// The simplest "free" approach: just encrypt in interleaved format directly
// (use makeInterleavedPlaintext at key generation time).  For cases where the
// input arrives as replicated and must be converted server-side, masks are needed.
// Here we implement the mask-free "doubling" that works when K is a power of 2
// and the input starts as a SINGLE copy in slots 0..d-1 (zero elsewhere):
//
//   For s = 0 to log2(K)-1:
//     ct = ct + Rot(ct, d * 2^s)
//
// This creates K = 2^log2(K) standard replicated copies (NOT interleaved).
// True interleaved preprocessing requires masks (one level per log step).
// In practice, interleaved inputs are provided directly by the encryptor.
Ciphertext cacheMirPreprocess(Ciphertext& x_single, int d, int numSlots) {
    // x_single has one copy of x in slots 0..d-1; zeros elsewhere.
    // Expand to standard replicated layout using log2(K) rotations+adds.
    int K = numSlots / d;
    Ciphertext ct(x_single.cc_);
    ct.copy(x_single);
    int step = d;
    while (step < numSlots) {
        Ciphertext rot(x_single.cc_);
        rot.copy(ct);
        rot.rotate(step);
        ct.add(rot);
        step *= 2;
    }
    // ct is now standard replicated: slot[p*d+i] = x[i].
    // NOTE: true interleaved layout (slot[K*i+j]=x[(i+j)%d]) differs from this.
    // Use this result directly with cacheMirInterleavedLinear only if the diagonal
    // encoding already accounts for the replicated layout (see notes in .cuh).
    return ct;
}

// cacheMirReduce: sum K interleaved partial results into one replicated output.
// The multiply-accumulate produces slot[K*i+j] = partial_y[(i+j)%d].
// Reduce to replicated: slot[p*d+i] = y[i] using log(K) rotate+adds.
Ciphertext cacheMirReduce(Ciphertext& y_interleaved, int d, int numSlots) {
    int K = numSlots / d;
    Ciphertext result(y_interleaved.cc_);
    result.copy(y_interleaved);
    // Sum K copies spaced by K apart: Rot(ct, K) + Rot(ct, 2K) + ... + Rot(ct, (K-1)K).
    // Use tree: at each step, double the accumulated span.
    for (int step = K; step < numSlots; step *= 2) {
        Ciphertext rot(y_interleaved.cc_);
        rot.copy(result);
        rot.rotate(step);
        result.add(rot);
    }
    return result;
}

// cacheMirInterleavedLinear: apply W to an interleaved-layout ciphertext.
// Input: slot[K*i+j] = x[(i+j)%d]
// Output: slot[K*i+j] = (W @ x)[(i+j)%d]
Ciphertext cacheMirInterleavedLinear(Ciphertext& x, const CacheMirInterleavedPrecomp& precomp) {
    Context& cc_ = x.cc_;
    int bStep    = precomp.bStep;
    int gStep    = precomp.gStep;
    int K        = precomp.K;
    int nDiag    = (int)precomp.weights.size();

    if (x.NoiseLevel == 2) x.rescale();

    // Baby-step hoisted rotations: rotate by b*K for b = 0..bStep-1.
    std::vector<Ciphertext> ctRot;
    ctRot.reserve(bStep);
    {
        std::vector<Ciphertext*> ptrs;
        ptrs.reserve(bStep);
        for (int b = 0; b < bStep; ++b) {
            ctRot.emplace_back(cc_);
            ctRot.back().growToLevel(x.getLevel());
            ctRot.back().dropToLevel(x.getLevel());
            ctRot.back().extend(false);
            ptrs.push_back(&ctRot.back());
        }
        std::vector<int> baby_steps(bStep);
        for (int b = 0; b < bStep; ++b) baby_steps[b] = b * K;
        x.rotate_hoisted(baby_steps, ptrs, /*ext=*/false);
    }

    // Giant steps: for each g, accumulate sum_b ctRot[b] * pt[g*bStep+b],
    // then rotate by g*bStep*K.
    Ciphertext result(cc_);
    bool result_init = false;

    for (int g = 0; g < gStep; ++g) {
        Ciphertext tmp(cc_);
        bool tmp_init = false;

        for (int b = 0; b < bStep; ++b) {
            int t = g * bStep + b;
            if (t >= nDiag) break;

            if (!tmp_init) {
                tmp.multPt(ctRot[b], precomp.weights[t], /*rescale=*/false);
                tmp_init = true;
            } else {
                tmp.addMultPt(ctRot[b], precomp.weights[t], /*rescale=*/false);
            }
        }
        if (!tmp_init) continue;

        tmp.rescale();

        if (g > 0) tmp.rotate(g * bStep * K);

        if (!result_init) {
            result.copy(tmp);
            result_init = true;
        } else {
            result.add(tmp);
        }
    }

    if (!result_init)
        throw std::runtime_error("cacheMirInterleavedLinear: empty precomp");
    return result;
}

// Plaintext reference for interleaved linear.
std::vector<double> cacheMirInterleavedLinearPlaintext(
    const std::vector<double>& x_interleaved,
    const std::vector<std::vector<double>>& W,
    int d, int numSlots)
{
    // Recover x[k] from interleaved: slot[K*0 + k] = x[(0+k)%d] = x[k] for k<d.
    assert((int)x_interleaved.size() >= d);
    int K = numSlots / d;
    std::vector<double> x(d);
    for (int k = 0; k < d; ++k) x[k] = x_interleaved[k];  // slot[k] = x[(0+k)%d] = x[k]

    // Compute y = W @ x.
    std::vector<double> y(d, 0.0);
    for (int m = 0; m < d; ++m)
        for (int k = 0; k < d; ++k)
            y[m] += W[m][k] * x[k];

    // Pack y into interleaved output layout.
    std::vector<double> out(numSlots, 0.0);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < d; ++j)
            out[K * i + j] = y[mod_pos(i + j, d)];
    return out;
}

}  // namespace FIDESlib::CKKS
