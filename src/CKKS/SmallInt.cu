#include "CKKS/SmallInt.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <memory>
#include <random>
#include <stdexcept>
#include "CKKS/Context.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"

namespace FIDESlib::CKKS {

namespace {
// ---- multiword helpers (little-endian u32 words, fixed capacities) ----
__device__ __forceinline__ void mw_mul_add(uint32_t* a, int& n, uint32_t m, uint32_t add) {
    uint64_t carry = add;
    for (int i = 0; i < n; ++i) {
        const uint64_t t = (uint64_t)a[i] * m + carry;
        a[i] = (uint32_t)t;
        carry = t >> 32;
    }
    if (carry)
        a[n++] = (uint32_t)carry;
}
__device__ __forceinline__ bool mw_gt(const uint32_t* a, const uint32_t* b, int n) {
    for (int i = n - 1; i >= 0; --i)
        if (a[i] != b[i])
            return a[i] > b[i];
    return false;
}
__device__ __forceinline__ void mw_sub_from(uint32_t* r, const uint32_t* a, const uint32_t* b, int n) {  // r = a - b
    int64_t borrow = 0;
    for (int i = 0; i < n; ++i) {
        int64_t t = (int64_t)a[i] - (int64_t)b[i] - borrow;
        borrow = t < 0;
        if (t < 0)
            t += (int64_t)1 << 32;
        r[i] = (uint32_t)t;
    }
}

// One thread per coefficient: reconstruct, centre, (divide), reduce modulo every output prime.
__global__ void smallIntDivRound_(void** src, void** out, const SmallIntTab* __restrict__ tab) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = tab->k, nw = tab->nw;
    // 1. Garner mixed-radix digits v_i
    uint32_t v[SI_MAXK];
    for (int i = 0; i < k; ++i) {
        const uint32_t qi = tab->q[i];
        uint64_t t = ((const uint32_t*)src[i])[idx] % qi;
        for (int j = 0; j < i; ++j) {
            const uint64_t vj = v[j] % qi;
            t = (t + qi - vj) % qi;
            t = (t * tab->qinv[i][j]) % qi;
        }
        v[i] = (uint32_t)t;
    }
    // 2. x = v_0 + q_0 (v_1 + q_1 (v_2 + ...)) as a multiword integer in [0, Q_k)
    uint32_t x[SI_MAXW + SI_MAXR];
    for (int i = 0; i < SI_MAXW + SI_MAXR; ++i)
        x[i] = 0;
    int nx = 1;
    x[0] = v[k - 1];
    for (int i = k - 2; i >= 0; --i)
        mw_mul_add(x, nx, tab->q[i], v[i]);
    // 3. centre: x > Q_k/2  ->  x := Q_k - x, negative
    bool neg = false;
    if (mw_gt(x, tab->Qkh, nw)) {
        neg = true;
        uint32_t y[SI_MAXW];
        mw_sub_from(y, tab->Qk, x, nw);
        for (int i = 0; i < nw; ++i)
            x[i] = y[i];
    }
    int nq = nw;
    // 4. q = round(x * R / 2^192)
    if (tab->divide) {
        uint32_t prod[SI_MAXW + SI_MAXR];
        for (int i = 0; i < SI_MAXW + SI_MAXR; ++i)
            prod[i] = 0;
        for (int i = 0; i < nw; ++i) {
            uint64_t carry = 0;
            for (int j = 0; j < tab->nr; ++j) {
                const uint64_t t = (uint64_t)x[i] * tab->R[j] + prod[i + j] + carry;
                prod[i + j] = (uint32_t)t;
                carry = t >> 32;
            }
            int p_ = i + tab->nr;
            while (carry) {
                const uint64_t t = (uint64_t)prod[p_] + carry;
                prod[p_] = (uint32_t)t;
                carry = t >> 32;
                ++p_;
            }
        }
        // + 2^191 (round half up), then >> 192 == drop 6 words
        {
            uint64_t carry = 1ull << 31;
            for (int i = 5; i < SI_MAXW + SI_MAXR && carry; ++i) {
                const uint64_t t = (uint64_t)prod[i] + carry;
                prod[i] = (uint32_t)t;
                carry = t >> 32;
            }
        }
        constexpr int DW = SI_SHIFT / 32;
        nq = nw + tab->nr - DW;
        if (nq < 1)
            nq = 1;
        for (int i = 0; i < nq; ++i)
            x[i] = prod[i + DW];
    }
    // 5. reduce modulo every output prime (Horner with (2^32)^w mod p), restore the sign
    for (int o = 0; o < tab->nout; ++o) {
        const uint32_t p = tab->pout[o];
        uint64_t acc = 0;
        for (int w = 0; w < nq; ++w)
            acc = (acc + (uint64_t)(x[w] % p) * tab->pow32[o][w]) % p;
        uint32_t r = (uint32_t)acc;
        if (neg && r)
            r = p - r;
        ((uint32_t*)out[o])[idx] = r;
    }
}

uint64_t host_mulmod(uint64_t a, uint64_t b, uint64_t p) { return (uint64_t)((__uint128_t)a * b % p); }
uint64_t host_powmod(uint64_t a, uint64_t e, uint64_t p) {
    uint64_t r = 1;
    for (a %= p; e; e >>= 1, a = host_mulmod(a, a, p))
        if (e & 1)
            r = host_mulmod(r, a, p);
    return r;
}

void* limbData(LimbImpl& l) {
    void* v = nullptr;
    SWITCH_RET(l, v.data, v);
    return v;
}
// Every per-limb stream of P joins P.s / P.s fans out to them (the per-limb launch pattern used throughout).
void joinLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i)
        P.s.wait(STREAM(P.limb[i]));
    for (auto& l : P.SPECIALlimb)
        P.s.wait(STREAM(l));
}
void forkLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i)
        STREAM(P.limb[i]).wait(P.s);
    for (auto& l : P.SPECIALlimb)
        STREAM(l).wait(P.s);
}

// Copy the first k limbs of p into a fresh poly (coefficient form on return unless srcCoeff).
struct SrcCopy {
    RNSPoly tmp;
    SrcCopy(ContextData& cc, const RNSPoly& p_, int k, bool srcCoeff) : tmp(cc, k - 1) {
        RNSPoly& p = const_cast<RNSPoly&>(p_);  // read-only use of its limbs and streams
        auto& P = p.GPU.at(0);
        auto& T = tmp.GPU.at(0);
        cudaSetDevice(P.device);
        joinLimbs(P);
        T.s.wait(P.s);
        // Source pointers from the authoritative device table: after an INTT/NTT pair the current data may sit
        // in the swapped `aux` buffer (limbptr follows the swap, the Limb's v.data does not).
        std::vector<void*> src(k), dst(k);
        cudaStreamSynchronize(P.s.ptr());
        cudaMemcpy(src.data(), P.limbptr.data, (size_t)k * sizeof(void*), cudaMemcpyDeviceToHost);
        cudaMemcpy(dst.data(), T.limbptr.data, (size_t)k * sizeof(void*), cudaMemcpyDeviceToHost);
        for (int i = 0; i < k; ++i)
            cudaMemcpyAsync(dst[i], src[i], (size_t)cc.N * sizeof(uint32_t), cudaMemcpyDeviceToDevice, T.s.ptr());
        forkLimbs(T);
        if (!srcCoeff)
            tmp.INTT(cc.batch, false);
        joinLimbs(T);
    }
};

SmallIntTab* uploadTab(const SmallIntTab& t, cudaStream_t s) {
    SmallIntTab* d = nullptr;
    cudaMallocAsync((void**)&d, sizeof(SmallIntTab), s);
    cudaMemcpyAsync(d, &t, sizeof(SmallIntTab), cudaMemcpyHostToDevice, s);
    return d;
}
}  // namespace

SmallIntTab buildSmallIntTab(const std::vector<uint64_t>& srcPrimes, const std::vector<uint64_t>& outPrimes,
                             long double D) {
    SmallIntTab t{};
    t.k = (int)srcPrimes.size();
    t.nout = (int)outPrimes.size();
    assert(t.k >= 1 && t.k <= SI_MAXK && t.nout <= SI_MAXOUT);
    for (int i = 0; i < t.k; ++i) {
        assert(srcPrimes[i] < (1ull << 32));
        t.q[i] = (uint32_t)srcPrimes[i];
        for (int j = 0; j < i; ++j)
            t.qinv[i][j] = (uint32_t)host_powmod(srcPrimes[j] % srcPrimes[i], srcPrimes[i] - 2, srcPrimes[i]);
    }
    // Q_k as words
    std::vector<uint32_t> Q(1, 1u);
    for (int i = 0; i < t.k; ++i) {
        uint64_t carry = 0;
        for (auto& w : Q) {
            const uint64_t v = (uint64_t)w * (uint32_t)srcPrimes[i] + carry;
            w = (uint32_t)v;
            carry = v >> 32;
        }
        if (carry)
            Q.push_back((uint32_t)carry);
    }
    t.nw = (int)Q.size() + 1;  // one spare word for the Horner carries
    assert(t.nw <= SI_MAXW);
    for (int i = 0; i < (int)Q.size(); ++i)
        t.Qk[i] = Q[i];
    // Q_k / 2
    uint32_t rem = 0;
    for (int i = (int)Q.size() - 1; i >= 0; --i) {
        const uint64_t cur = ((uint64_t)rem << 32) | Q[i];
        t.Qkh[i] = (uint32_t)(cur >> 1);
        rem = (uint32_t)(cur & 1);
    }
    // R = round(2^192 / D) with a 64-bit mantissa
    t.divide = D > 0 ? 1 : 0;
    t.nr = 0;
    if (t.divide) {
        int e = 0;
        const long double f = frexpl(ldexpl(1.0L, SI_SHIFT) / D, &e);  // 2^192/D = f * 2^e, f in [0.5, 1)
        const uint64_t m = (uint64_t)ldexpl(f, 64);                      // 64-bit mantissa
        const int sh = e - 64;  // R = m << sh
        assert(sh >= 0 && sh + 64 <= 32 * SI_MAXR);
        const unsigned __int128 R = (unsigned __int128)m << (sh % 32);
        const int w0 = sh / 32;
        for (int i = 0; i < 4 && w0 + i < SI_MAXR; ++i)
            t.R[w0 + i] = (uint32_t)(R >> (32 * i));
        t.nr = SI_MAXR;
        while (t.nr > 1 && t.R[t.nr - 1] == 0)
            --t.nr;
    }
    for (int o = 0; o < t.nout; ++o) {
        const uint64_t p = outPrimes[o];
        t.pout[o] = (uint32_t)p;
        uint64_t pw = 1 % p;
        const uint64_t b32 = (1ull << 32) % p;
        for (int w = 0; w < SI_MAXW + SI_MAXR; ++w) {
            t.pow32[o][w] = (uint32_t)pw;
            pw = host_mulmod(pw, b32, p);
        }
    }
    return t;
}

long double smallIntEffectiveDivisor(const SmallIntTab& t) {
    if (!t.divide)
        return 1.0L;
    long double R = 0;
    for (int i = t.nr - 1; i >= 0; --i)
        R = R * 4294967296.0L + t.R[i];
    return ldexpl(1.0L, SI_SHIFT) / R;
}

void smallIntLiftToSpecial(ContextData& cc, RNSPoly& p, int k) {
    assert(cc.GPUid.size() == 1 && cc.precom.constants[0].type == 0 && "small-int: u32 chain, single GPU");
    auto& P = p.GPU.at(0);
    cudaSetDevice(P.device);
    SrcCopy src(cc, p, k, false);
    auto& T = src.tmp.GPU.at(0);
    if (!p.isModUp() && P.SPECIALlimb.empty())
        p.generateSpecialLimbs(false, false);
    std::vector<uint64_t> sp, op;
    for (int i = 0; i < k; ++i)
        sp.push_back(cc.prime.at(i).p);
    for (auto& s : cc.specialPrime)
        op.push_back(s.p);
    const SmallIntTab tab = buildSmallIntTab(sp, op, 0);
    joinLimbs(P);
    P.s.wait(T.s);
    SmallIntTab* d = uploadTab(tab, P.s.ptr());
    smallIntDivRound_<<<dim3{(uint32_t)cc.N / 128}, 128, 0, P.s.ptr()>>>(T.limbptr.data, P.SPECIALlimbptr.data, d);
    cudaFreeAsync(d, P.s.ptr());
    P.nttSpecialLimbs();
    forkLimbs(P);
    T.s.wait(P.s);
    forkLimbs(T);  // tmp's limb streams see the kernel before tmp is destroyed
    p.SetModUp(true);
}

void smallIntDivideKeepLevel(ContextData& cc, RNSPoly& p, int k, long double D, bool withSpecials, bool srcCoeff) {
    assert(cc.GPUid.size() == 1 && cc.precom.constants[0].type == 0 && "small-int: u32 chain, single GPU");
    auto& P = p.GPU.at(0);
    cudaSetDevice(P.device);
    SrcCopy src(cc, p, k, srcCoeff);
    auto& T = src.tmp.GPU.at(0);
    if (withSpecials && P.SPECIALlimb.empty())
        p.generateSpecialLimbs(false, false);
    std::vector<uint64_t> sp, oq, os;
    for (int i = 0; i < k; ++i)
        sp.push_back(cc.prime.at(i).p);
    const int nq = p.getLevel() + 1;
    for (int i = 0; i < nq; ++i)
        oq.push_back(cc.prime.at(i).p);
    for (auto& s : cc.specialPrime)
        os.push_back(s.p);
    const SmallIntTab tq = buildSmallIntTab(sp, oq, D);
    joinLimbs(P);
    P.s.wait(T.s);
    SmallIntTab* dq = uploadTab(tq, P.s.ptr());
    smallIntDivRound_<<<dim3{(uint32_t)cc.N / 128}, 128, 0, P.s.ptr()>>>(T.limbptr.data, P.limbptr.data, dq);
    cudaFreeAsync(dq, P.s.ptr());
    if (withSpecials) {
        const SmallIntTab ts = buildSmallIntTab(sp, os, D);
        SmallIntTab* ds = uploadTab(ts, P.s.ptr());
        smallIntDivRound_<<<dim3{(uint32_t)cc.N / 128}, 128, 0, P.s.ptr()>>>(T.limbptr.data, P.SPECIALlimbptr.data,
                                                                            ds);
        cudaFreeAsync(ds, P.s.ptr());
        P.nttSpecialLimbs();
    }
    forkLimbs(P);
    p.NTT(cc.batch, false);
    joinLimbs(P);
    T.s.wait(P.s);
    forkLimbs(T);
    if (withSpecials)
        p.SetModUp(true);
}

void smallIntDivideFromTo(ContextData& cc, const RNSPoly& src, RNSPoly& dst, int k, long double D, bool withSpecials) {
    assert(cc.GPUid.size() == 1 && cc.precom.constants[0].type == 0 && "small-int: u32 chain, single GPU");
    auto& P = dst.GPU.at(0);
    cudaSetDevice(P.device);
    SrcCopy s(cc, src, k, false);
    auto& T = s.tmp.GPU.at(0);
    if (withSpecials && P.SPECIALlimb.empty())
        dst.generateSpecialLimbs(false, false);
    std::vector<uint64_t> sp, oq, os;
    for (int i = 0; i < k; ++i)
        sp.push_back(cc.prime.at(i).p);
    const int nq = dst.getLevel() + 1;
    for (int i = 0; i < nq; ++i)
        oq.push_back(cc.prime.at(i).p);
    for (auto& x : cc.specialPrime)
        os.push_back(x.p);
    const SmallIntTab tq = buildSmallIntTab(sp, oq, D);
    joinLimbs(P);
    P.s.wait(T.s);
    SmallIntTab* dq = uploadTab(tq, P.s.ptr());
    smallIntDivRound_<<<dim3{(uint32_t)cc.N / 128}, 128, 0, P.s.ptr()>>>(T.limbptr.data, P.limbptr.data, dq);
    cudaFreeAsync(dq, P.s.ptr());
    if (withSpecials) {
        const SmallIntTab ts = buildSmallIntTab(sp, os, D);
        SmallIntTab* ds = uploadTab(ts, P.s.ptr());
        smallIntDivRound_<<<dim3{(uint32_t)cc.N / 128}, 128, 0, P.s.ptr()>>>(T.limbptr.data, P.SPECIALlimbptr.data,
                                                                            ds);
        cudaFreeAsync(ds, P.s.ptr());
        P.nttSpecialLimbs();
    }
    forkLimbs(P);
    dst.NTT(cc.batch, false);
    joinLimbs(P);
    T.s.wait(P.s);
    forkLimbs(T);
    if (withSpecials)
        dst.SetModUp(true);
}

Plaintext relevelPlaintext(Context& cc_, ContextData& cc, const Plaintext& pt, int shift, double factor) {
    const int d = cc.compositeDegree();
    const int oldL = pt.c0.getLevel();
    const int newL = oldL + d * shift;
    if (newL > cc.L)
        throw std::runtime_error("relevelPlaintext: beyond the top level");
    const long double D = (long double)cc.sfAtLimb(oldL) / ((long double)factor * (long double)cc.sfAtLimb(newL));
    Plaintext np(cc_);
    np.c0.grow(newL);
    smallIntDivideFromTo(cc, pt.c0, np.c0, 4, D, pt.c0.isModUp());
    np.copyMetadata(pt);
    np.NoiseFactor = cc.sfAtLimb(newL);
    cudaDeviceSynchronize();
    return np;
}

Plaintext relevelPlaintextNF(Context& cc_, ContextData& cc, const Plaintext& pt, int newL, double newNF) {
    if (newL > cc.L)
        throw std::runtime_error("relevelPlaintextNF: beyond the top level");
    const long double D = (long double)pt.NoiseFactor / (long double)newNF;
    Plaintext np(cc_);
    np.c0.grow(newL);
    smallIntDivideFromTo(cc, pt.c0, np.c0, 4, D, pt.c0.isModUp());
    np.copyMetadata(pt);
    np.NoiseFactor = newNF;
    cudaDeviceSynchronize();
    return np;
}

void smallIntScalarMultiply(ContextData& cc, Ciphertext& ct, double c) {
    assert(c > 0);
    // k = 3: compositeModRaise centres each Garner term on its own prime, so the raised coefficients lie in
    // (-q0, q0) rather than (-q0/2, q0/2); Q_2 = q0 would be ambiguous, Q_3 ~ 2^83 is not.
    const long double D = 1.0L / (long double)c;
    smallIntDivideKeepLevel(cc, ct.c0, 3, D, false);
    smallIntDivideKeepLevel(cc, ct.c1, 3, D, false);
}

long smallIntSelfTest(Ciphertext& any, double c) {
    ContextData& cc = any.cc;
    const int N = cc.N, L = cc.L;
    const long double D = 1.0L / (long double)c;
    std::vector<uint64_t> q(L + 1);
    for (int i = 0; i <= L; ++i) q[i] = cc.prime[i].p;
    const __int128 q0 = (__int128)q[0] * q[1];
    std::mt19937_64 g(7);
    std::vector<__int128> x(N);
    for (int n = 0; n < N; ++n) {
        __int128 r = (__int128)(g() % (uint64_t)1000000000000000000ull) * 1000 + (g() % 1000);
        r %= q0;
        if (g() & 1) r = -r;
        if (n % 7 == 0) r %= 1000000;
        x[n] = r;
    }
    std::vector<std::vector<uint64_t>> in(L + 1, std::vector<uint64_t>(N)), ref(L + 1, std::vector<uint64_t>(N));
    for (int i = 0; i <= L; ++i)
        for (int n = 0; n < N; ++n) {
            __int128 xi = x[n] % (__int128)q[i];
            if (xi < 0) xi += q[i];
            in[i][n] = (uint64_t)xi;
            const __int128 yi = (__int128)llroundl((long double)x[n] / D);
            __int128 yr = yi % (__int128)q[i];
            if (yr < 0) yr += q[i];
            ref[i][n] = (uint64_t)yr;
        }
    RNSPoly P(cc, -1), R(cc, -1);
    P.load(in, q);
    R.load(ref, q);
    P.NTT(cc.batch, true);
    R.NTT(cc.batch, true);
    cudaDeviceSynchronize();
    smallIntDivideKeepLevel(cc, P, 3, D, false);
    cudaDeviceSynchronize();
    std::vector<std::vector<uint64_t>> got, want;
    P.store(got);
    R.store(want);
    cudaDeviceSynchronize();
    long bad = 0;
    int first = -1, firstLimb = -1;
    for (int i = 0; i <= L; ++i)
        for (int n = 0; n < N; ++n)
            if (got[i][n] != want[i][n]) {
                if (first < 0) { first = n; firstLimb = i; }
                ++bad;
            }
    std::fprintf(stderr, "[smallint] L=%d N=%d c=%g mismatches=%ld first=(limb %d, n %d, x=%lld)\n", L, N, c, bad,
                 firstLimb, first, first >= 0 ? (long long)x[first] : 0LL);
    return bad;
}

long smallIntConsistencyCheck(ContextData& cc, const RNSPoly& p_, int k, const char* tag) {
    RNSPoly& p = const_cast<RNSPoly&>(p_);
    const int N = cc.N, L = p.getLevel();
    RNSPoly t(cc, L);
    t.copy(p);
    t.INTT(cc.batch, true);
    cudaDeviceSynchronize();
    std::vector<std::vector<uint64_t>> h;
    t.store(h);
    cudaDeviceSynchronize();
    std::vector<uint64_t> q(L + 1);
    for (int i = 0; i <= L; ++i) q[i] = cc.prime[i].p;
    // Garner from the first k limbs, centred
    __int128 Qk = 1;
    for (int i = 0; i < k; ++i) Qk *= q[i];
    long bad = 0, badCoef = 0;
    long double maxAbs = 0;
    std::vector<long> badPerLimb(L + 1, 0);
    for (int n = 0; n < N; ++n) {
        __int128 x = 0, m = 1;
        for (int i = 0; i < k; ++i) {  // incremental CRT
            __int128 r = (__int128)h[i][n];
            __int128 diff = ((r - x) % (__int128)q[i] + q[i]) % q[i];
            __int128 inv = 1;
            {  // m^{-1} mod q[i] via Fermat
                uint64_t base = (uint64_t)(m % (__int128)q[i]), e = q[i] - 2, acc = 1;
                while (e) { if (e & 1) acc = (uint64_t)((__uint128_t)acc * base % q[i]); base = (uint64_t)((__uint128_t)base * base % q[i]); e >>= 1; }
                inv = acc;
            }
            x += m * ((diff * inv) % (__int128)q[i]);
            m *= q[i];
        }
        if (x > Qk / 2) x -= Qk;
        if ((long double)(x < 0 ? -x : x) > maxAbs) maxAbs = (long double)(x < 0 ? -x : x);
        bool any = false;
        for (int i = k; i <= L; ++i) {
            __int128 r = x % (__int128)q[i]; if (r < 0) r += q[i];
            if ((uint64_t)r != h[i][n]) { ++bad; ++badPerLimb[i]; any = true; }
        }
        if (any) ++badCoef;
    }
    std::fprintf(stderr, "[smallint-check] %s: level %d, k=%d, inconsistent pairs=%ld coefficients=%ld/%d, max|x|=2^%.2Lf (q0 = 2^%.2f)\n",
                 tag, L, k, bad, badCoef, N, maxAbs > 0 ? log2l(maxAbs) : 0.0L, std::log2((double)q[0] * q[1]));
    if (bad) {
        std::fprintf(stderr, "[smallint-check]   per-limb inconsistent counts:");
        for (int i = k; i <= L; ++i) std::fprintf(stderr, " %ld", badPerLimb[i]);
        std::fprintf(stderr, "\n");
    }
    return bad;
}

long smallIntScalarCheck(ContextData& cc, const Ciphertext& ct_, double c, const char* tag) {
    Ciphertext& ct = const_cast<Ciphertext&>(ct_);
    const int N = cc.N;
    const long double D = 1.0L / (long double)c;
    long total = 0;
    for (int which = 0; which < 2; ++which) {
        RNSPoly& src = which ? ct.c1 : ct.c0;
        const int L = src.getLevel();
        std::vector<uint64_t> q(L + 1);
        for (int i = 0; i <= L; ++i) q[i] = cc.prime[i].p;
        // host reference from FIDESlib's own copy + INTT
        RNSPoly t(cc, L);
        t.copy(src);
        t.INTT(cc.batch, true);
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> h;
        t.store(h);
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> ref(L + 1, std::vector<uint64_t>(N));
        const int k = 3;
        __int128 Qk = 1;
        for (int i = 0; i < k; ++i) Qk *= q[i];
        for (int n = 0; n < N; ++n) {
            __int128 x = 0, m = 1;
            for (int i = 0; i < k; ++i) {
                __int128 r = (__int128)h[i][n];
                __int128 diff = ((r - x) % (__int128)q[i] + q[i]) % q[i];
                uint64_t base = (uint64_t)(m % (__int128)q[i]), e = q[i] - 2, acc = 1;
                while (e) { if (e & 1) acc = (uint64_t)((__uint128_t)acc * base % q[i]); base = (uint64_t)((__uint128_t)base * base % q[i]); e >>= 1; }
                x += m * ((diff * (__int128)acc) % (__int128)q[i]);
                m *= q[i];
            }
            if (x > Qk / 2) x -= Qk;
            const __int128 y = (__int128)llroundl((long double)x / D);
            for (int i = 0; i <= L; ++i) {
                __int128 yr = y % (__int128)q[i];
                if (yr < 0) yr += q[i];
                ref[i][n] = (uint64_t)yr;
            }
        }
        // the primitive on a copy of the ORIGINAL (NTT form), then INTT + store
        RNSPoly u(cc, L);
        u.copy(src);
        cudaDeviceSynchronize();
        smallIntDivideKeepLevel(cc, u, 3, D, false);
        cudaDeviceSynchronize();
        u.INTT(cc.batch, true);
        cudaDeviceSynchronize();
        std::vector<std::vector<uint64_t>> g;
        u.store(g);
        cudaDeviceSynchronize();
        long bad = 0; int firstLimb = -1, firstN = -1;
        for (int i = 0; i <= L; ++i)
            for (int n = 0; n < N; ++n)
                if (g[i][n] != ref[i][n]) { if (firstLimb < 0) { firstLimb = i; firstN = n; } ++bad; }
        std::fprintf(stderr, "[smallint-scalar] %s %s: level %d mismatches=%ld first=(limb %d, n %d) got=%llu ref=%llu\n", tag,
                     which ? "c1" : "c0", L, bad, firstLimb, firstN,
                     firstLimb >= 0 ? (unsigned long long)g[firstLimb][firstN] : 0ull,
                     firstLimb >= 0 ? (unsigned long long)ref[firstLimb][firstN] : 0ull);
        total += bad;
    }
    return total;
}

long relevelSelfTest(Context& cc_, ContextData& cc, const Plaintext& pt, const char* tag) {
    Plaintext np = relevelPlaintext(cc_, cc, pt, 0);
    cudaDeviceSynchronize();
    std::vector<std::vector<uint64_t>> a, b;
    const_cast<RNSPoly&>(pt.c0).store(a);
    np.c0.store(b);
    cudaDeviceSynchronize();
    long bad = 0; int fl = -1, fn = -1;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        for (size_t n = 0; n < a[i].size(); ++n)
            if (a[i][n] != b[i][n]) { if (fl < 0) { fl = (int)i; fn = (int)n; } ++bad; }
    // specials: compare via a raw download of the special limb pointer tables
    long badS = 0;
    const int K = (int)cc.specialPrime.size();
    if (pt.c0.isModUp() && np.c0.isModUp()) {
        auto& PA = const_cast<RNSPoly&>(pt.c0).GPU.at(0);
        auto& PB = np.c0.GPU.at(0);
        std::vector<void*> pa(K), pb(K);
        cudaMemcpy(pa.data(), PA.SPECIALlimbptr.data, K * sizeof(void*), cudaMemcpyDeviceToHost);
        cudaMemcpy(pb.data(), PB.SPECIALlimbptr.data, K * sizeof(void*), cudaMemcpyDeviceToHost);
        std::vector<uint32_t> ha(cc.N), hb(cc.N);
        for (int j = 0; j < K; ++j) {
            cudaMemcpy(ha.data(), pa[j], cc.N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), pb[j], cc.N * 4, cudaMemcpyDeviceToHost);
            for (int n = 0; n < cc.N; ++n) if (ha[n] != hb[n]) ++badS;
        }
    }
    std::fprintf(stderr, "[relevel-self] %s: level %d limbs %zu/%zu, Q mismatches=%ld first=(limb %d, n %d), special mismatches=%ld (modUp %d/%d), NF %g vs %g\n",
                 tag, pt.c0.getLevel(), a.size(), b.size(), bad, fl, fn, badS, (int)pt.c0.isModUp(), (int)np.c0.isModUp(),
                 pt.NoiseFactor, np.NoiseFactor);
    return bad + badS;
}

static std::unique_ptr<RNSPoly> g_diag_sk;
static std::vector<uint64_t> g_diag_moduli;

void loadDiagSecret(ContextData& cc, const std::vector<std::vector<uint64_t>>& skLimbs, const std::vector<uint64_t>& moduli) {
    g_diag_sk = std::make_unique<RNSPoly>(cc, -1);
    g_diag_sk->load(skLimbs, moduli);
    g_diag_moduli = moduli;
    cudaDeviceSynchronize();
}

static bool dumpCoeffPoly(ContextData& cc, RNSPoly& m, int L, uint32_t noiseLevel, uint32_t slots, double nf, const char* path) {
    m.INTT(cc.batch, true);
    cudaDeviceSynchronize();
    std::vector<std::vector<uint64_t>> h;
    m.store(h);
    cudaDeviceSynchronize();
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t hdr[4] = {(uint32_t)cc.N, (uint32_t)(L + 1), noiseLevel, slots};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    std::fwrite(&nf, sizeof(nf), 1, f);
    for (int i = 0; i <= L; ++i) { const uint64_t q = cc.prime[i].p; std::fwrite(&q, 8, 1, f); }
    for (int i = 0; i <= L; ++i) std::fwrite(h[i].data(), 8, h[i].size(), f);
    std::fclose(f);
    return true;
}

bool exactPlainDump(const Plaintext& pt_, const char* path) {
    Plaintext& pt = const_cast<Plaintext&>(pt_);
    ContextData& cc = pt.cc;
    const int L = pt.c0.getLevel();
    RNSPoly m(cc, L);
    m.copy(pt.c0);
    return dumpCoeffPoly(cc, m, L, (uint32_t)pt.NoiseLevel, (uint32_t)pt.slots, pt.NoiseFactor, path);
}

bool exactCtPolyDump(const Ciphertext& ct_, const char* base) {
    Ciphertext& ct = const_cast<Ciphertext&>(ct_);
    ContextData& cc = ct.cc;
    const int L = ct.getLevel();
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        RNSPoly m(cc, L);
        m.copy(which ? ct.c1 : ct.c0);
        ok &= dumpCoeffPoly(cc, m, L, (uint32_t)ct.NoiseLevel, (uint32_t)ct.slots, ct.NoiseFactor,
                            (std::string(base) + (which ? "-c1.ct" : "-c0.ct")).c_str());
    }
    return ok;
}

bool exactDecryptDump(const Ciphertext& ct_, const char* path) {
    if (!g_diag_sk) return false;
    Ciphertext& ct = const_cast<Ciphertext&>(ct_);
    ContextData& cc = ct.cc;
    const int L = ct.getLevel();
    RNSPoly s(cc, g_diag_sk->getLevel());
    s.copy(*g_diag_sk);          // copy() takes the source's level...
    s.dropToLevel(L);            // ...so truncate the key to the ciphertext's limbs afterwards
    RNSPoly m(cc, L);
    m.copy(ct.c1);
    m.multElement(s);           // c1 * s, NTT pointwise over Q
    m.add(ct.c0);
    m.INTT(cc.batch, true);
    cudaDeviceSynchronize();
    std::vector<std::vector<uint64_t>> h;
    m.store(h);
    cudaDeviceSynchronize();
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t hdr[4] = {(uint32_t)cc.N, (uint32_t)(L + 1), (uint32_t)ct.NoiseLevel, (uint32_t)ct.slots};
    std::fwrite(hdr, sizeof(hdr), 1, f);
    const double nf = ct.NoiseFactor;
    std::fwrite(&nf, sizeof(nf), 1, f);
    for (int i = 0; i <= L; ++i) { const uint64_t q = cc.prime[i].p; std::fwrite(&q, 8, 1, f); }
    for (int i = 0; i <= L; ++i) std::fwrite(h[i].data(), 8, h[i].size(), f);
    std::fclose(f);
    return true;
}

}  // namespace FIDESlib::CKKS
