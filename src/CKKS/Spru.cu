// SPRU bootstrapping for one complex slot (see Spru.cuh).
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <map>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CKKS/Spru.cuh"  // last: pke/openfhe.h defines a `duration` macro
#ifdef duration
#undef duration
#endif

namespace FIDESlib::CKKS {

struct SpruKey {
    int h = 64, n = 2, N = 0, B = 0;
    std::vector<int> pos;  // the 1 of block b sits at b*B + pos[b]; pos[0] = 0 (s'_0 = 1)
    std::unique_ptr<KeySwitchingKey> atob;
    std::vector<std::unique_ptr<Ciphertext>> cs;  // 2n bootstrapping key ciphertexts (top level)
    int accBStep = 4;
    // GPU encoder state: FFT input positions of slot j (zeta^{5^j}) and of its conjugate, the FFT ping-pong buffers
    // (2n of them, N complex each), the residues of the encoding scale, and the mask plaintexts per level
    int* uPos = nullptr;
    int* uNeg = nullptr;
    double2* buf = nullptr;
    double2* buf2 = nullptr;
    uint32_t* primes = nullptr;
    std::map<int, Plaintext> maskAt;
    bool gpuEncode = true;
    ~SpruKey() {
        cudaFree(uPos); cudaFree(uNeg); cudaFree(buf); cudaFree(buf2); cudaFree(primes);
    }
};

namespace {
void joinLimbsS(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) P.s.wait(STREAM(P.limb[i]));
}
void forkLimbsS(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) STREAM(P.limb[i]).wait(P.s);
}

// e_u[slot] for every key index i and component a; written straight into the FFT input of encoding u at the
// positions of zeta^{5^slot} (value) and of its conjugate (conjugate value)
__global__ void spruExtract_(void** c0, void** c1, uint32_t p0, uint32_t p1, uint32_t inv, int N, int h, int n, int B,
                             double delta, const int* uPos, const int* uNeg, double2* w) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * n) return;
    const int i = tid / n, a = tid % n;
    const int ka = a * (N / n);
    const uint64_t q0 = (uint64_t)p0 * p1;
    auto crt = [&](void** r, int k) -> uint64_t {
        const uint64_t x0 = ((const uint32_t*)r[0])[k], x1 = ((const uint32_t*)r[1])[k];
        const uint64_t t = ((x1 + p1 - x0 % p1) % p1) * inv % p1;
        return x0 + (uint64_t)p0 * t;
    };
    uint64_t v;
    if (i <= ka) v = crt(c1, ka - i);
    else { v = crt(c1, N + ka - i); v = v ? q0 - v : 0; }
    if (i == 0) { v += crt(c0, ka); if (v >= q0) v -= q0; }
    const double ang = 2.0 * (double)v / (double)q0;  // in units of pi
    double sn, cs;
    sincospi(ang, &sn, &cs);
    const int sub = B / (2 * n), b = i / B, rem = i % B, u = rem / sub, k = rem % sub;
    const int slot = k * h * n + b * n + a;
    double2* W = w + (size_t)u * N;
    W[uPos[slot]] = make_double2(delta * cs, delta * sn);
    W[uNeg[slot]] = make_double2(delta * cs, -delta * sn);
}

// one radix-2 Stockham pass (forward transform, exp(-2 pi i ...)), batched over the 2n encodings
__global__ void spruFftPass_(const double2* in, double2* out, int N, int Ns, int batch) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = N / 2;
    if (tid >= half * batch) return;
    const int e = tid / half, j = tid % half;
    const double2* I = in + (size_t)e * N;
    double2* O = out + (size_t)e * N;
    double2 v0 = I[j], v1 = I[j + half];
    const int k = j % Ns;
    double sn, cs;
    sincospi(-(double)k / (double)Ns, &sn, &cs);
    const double2 t = make_double2(v1.x * cs - v1.y * sn, v1.x * sn + v1.y * cs);
    const int idx = (j / Ns) * Ns * 2 + k;
    O[idx] = make_double2(v0.x + t.x, v0.y + t.y);
    O[idx + Ns] = make_double2(v0.x - t.x, v0.y - t.y);
}

// m_k = round(Re(Y_k zeta^{-k} / N) * scale), reduced into every limb (coefficient form)
__global__ void spruScatter_(const double2* Y, int N, double scale, void** limbs, const uint32_t* primes, int nl) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    double sn, cs;
    sincospi(-(double)k / (double)N, &sn, &cs);
    const double re = (Y[k].x * cs - Y[k].y * sn) / (double)N;
    const long long m = llrint(re * scale);
    for (int l = 0; l < nl; ++l) {
        const long long p = primes[l];
        long long r = m % p;
        if (r < 0) r += p;
        ((uint32_t*)limbs[l])[k] = (uint32_t)r;
    }
}
}  // namespace

static thread_local bool g_spruTimed = true;
static double ms_since(std::chrono::steady_clock::time_point& t0) {
    if (!g_spruTimed) return 0.0;
    cudaDeviceSynchronize();
    auto t = std::chrono::steady_clock::now();
    const double r = std::chrono::duration<double, std::milli>(t - t0).count();
    t0 = t;
    return r;
}

std::shared_ptr<SpruKey> spruSetup(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                   const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, Context& GPUcc_, int h) {
    ContextData& GPUcc = *GPUcc_;
    auto K = std::make_shared<SpruKey>();
    K->h = h;
    K->N = (int)cc->GetRingDimension();
    K->B = K->N / h;
    const int N = K->N, B = K->B, n = K->n;
    // ---- block key s' (binary, one 1 per block, s'_0 = 1)
    std::mt19937_64 rng(std::random_device{}());
    K->pos.resize(h);
    std::vector<int64_t> sv(N, 0);
    for (int b = 0; b < h; ++b) {
        K->pos[b] = b == 0 ? 0 : (int)(rng() % (uint64_t)B);
        sv[b * B + K->pos[b]] = 1;
    }
    auto params = cc->GetElementParams();
    lbcrypto::DCRTPoly sp(params, Format::COEFFICIENT, true);
    sp = sv;  // signed integer coefficients on every tower
    sp.SetFormat(Format::EVALUATION);
    auto skNew = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cc);
    skNew->SetPrivateElement(std::move(sp));
    skNew->SetKeyTag(keys.secretKey->GetKeyTag());
    auto atob = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        cc->GetScheme()->KeySwitchGen(keys.secretKey, skNew));
    K->atob = std::make_unique<KeySwitchingKey>(GPUcc_);
    {
        RawKeySwitchKey raw = GetKeySwitchKey(atob);
        K->atob->Initialize(raw);
    }
    // ---- bootstrapping key: cs_u = Enc(S_u), S_u slots k*h*n + b*n + a = s'_{b*B + u*B/(2n) + k}
    const int sub = B / (2 * n);
    for (int u = 0; u < 2 * n; ++u) {
        std::vector<double> S(N / 2, 0.0);
        for (int k = 0; k < sub; ++k)
            for (int b = 0; b < h; ++b)
                for (int a = 0; a < n; ++a)
                    S[k * h * n + b * n + a] = (double)sv[b * B + u * sub + k];
        auto pt = cc->MakeCKKSPackedPlaintext(S, 1, 0, nullptr, N / 2);
        auto ct = cc->Encrypt(keys.publicKey, pt);
        auto raw = GetRawCipherText(cc, ct);
        K->cs.push_back(std::make_unique<Ciphertext>(GPUcc_, raw));
    }
    // ---- rotation keys: trace (Accumulate over stride h*n), product (n * 2^i), recombination (1), conjugation exists
    std::vector<int> idx = GetAccumulateRotationIndices(K->accBStep, h * n, N / 2 / (h * n));
    for (int s = n; s < h * n; s <<= 1) idx.push_back(s);
    idx.push_back(1);
    std::vector<int> need;
    for (int i : idx)
        if (!GPUcc.HasRotationKey(i, K->cs[0]->keyID)) need.push_back(i);
    if (!need.empty()) GenAndAddRotationKeys(cc, keys, GPUcc_, need);
    cudaDeviceSynchronize();
    {   // FFT positions: slot j <-> zeta^{t}, t = 5^j mod 2N, w index u = (t-1)/2; conjugate at t' = 2N - t
        std::vector<int> up(N / 2), un(N / 2);
        uint64_t t = 1;
        for (int j = 0; j < N / 2; ++j) {
            up[j] = (int)((t - 1) / 2);
            un[j] = (int)((2ull * N - t - 1) / 2);
            t = (t * 5) % (2ull * N);
        }
        cudaMalloc(&K->uPos, sizeof(int) * N / 2);
        cudaMalloc(&K->uNeg, sizeof(int) * N / 2);
        cudaMemcpy(K->uPos, up.data(), sizeof(int) * N / 2, cudaMemcpyHostToDevice);
        cudaMemcpy(K->uNeg, un.data(), sizeof(int) * N / 2, cudaMemcpyHostToDevice);
        cudaMalloc(&K->buf, sizeof(double2) * (size_t)N * 2 * n);
        cudaMalloc(&K->buf2, sizeof(double2) * (size_t)N * 2 * n);
        if (const char* e = std::getenv("FIDESLIB_SPRU_HOST_ENCODE"); e && std::atoi(e) > 0) K->gpuEncode = false;
    }
    std::cerr << "[spru] setup: N=" << N << " h=" << h << " B=" << B << " n=" << n << " key cts=" << K->cs.size()
              << " at limb index " << K->cs[0]->getLevel() << ", rotation keys added " << need.size() << "\n";
    return K;
}

void spruBootstrap(Ciphertext& ct, SpruKey& K, uint32_t correction, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                   SpruTimes* times) {
    Context& cc_ = ct.cc_;
    ContextData& C = ct.cc;
    const int d = C.compositeDegree(), N = K.N, h = K.h, n = K.n, B = K.B, sub = B / (2 * n);
    SpruTimes tm;
    g_spruTimed = times != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto tStart = t0;
    // ---- 1. FLEXIBLEAUTO adjust to the composite bottom: integers m = v * sf(L) * 2^-correction
    if (ct.NoiseLevel == 2) ct.rescale();
    const double targetSF = C.sfAtLimb(C.L);
    const double adj = (targetSF / ct.NoiseFactor) * (C.modReduceProduct(ct.getLevel()) / C.sfAtLimb(ct.getLevel())) *
                       std::ldexp(1.0, -(int)correction);
    ct.multScalar(adj);
    ct.rescale();
    ct.dropToLevel(d - 1);
    tm.adjust_ms = ms_since(t0);
    // ---- 2. dense -> block key at the bottom
    ct.keySwitch(*K.atob);
    tm.switch_ms = ms_since(t0);
    const double targetSFv = C.sfAtLimb(C.L);
    std::vector<Plaintext> Ept;
    if (K.gpuEncode) {
        // ---- 3+4 on the GPU: extraction -> roots of unity in FFT order -> FFT -> round + reduce into every limb -> NTT
        // ct is overwritten by the result at the end: transform it in place (no copy)
        ct.c0.INTT(C.batch, true);
        ct.c1.INTT(C.batch, true);
        cudaDeviceSynchronize();
        const uint32_t p0 = (uint32_t)C.prime[0].p, p1 = (uint32_t)C.prime[1].p;
        const uint32_t inv = [&] {
            unsigned __int128 a = p0 % p1, res = 1, e = p1 - 2;
            while (e) { if (e & 1) res = res * a % p1; a = a * a % p1; e >>= 1; }
            return (uint32_t)res;
        }();
        const long double qd = (long double)p0 * (long double)p1;
        const double Dm = targetSFv * std::ldexp(1.0, -(int)correction);
        const double delta = std::pow((double)(qd / (4.0L * M_PIl * (long double)Dm)), 1.0 / h);
        cudaMemset(K.buf, 0, sizeof(double2) * (size_t)N * 2 * n);
        spruExtract_<<<(N * n + 255) / 256, 256>>>(ct.c0.GPU.at(0).limbptr.data, ct.c1.GPU.at(0).limbptr.data, p0, p1, inv,
                                                   N, h, n, B, delta, K.uPos, K.uNeg, K.buf);
        double2* in = K.buf;
        double2* out = K.buf2;
        for (int Ns = 1; Ns < N; Ns <<= 1) {
            spruFftPass_<<<((N / 2) * 2 * n + 255) / 256, 256>>>(in, out, N, Ns, 2 * n);
            std::swap(in, out);
        }
        tm.host_ms = ms_since(t0);
        const int topL = K.cs[0]->getLevel();
        if (!K.primes) {
            std::vector<uint32_t> pr(topL + 1);
            for (int l = 0; l <= topL; ++l) pr[l] = (uint32_t)C.prime[l].p;
            cudaMalloc(&K.primes, sizeof(uint32_t) * pr.size());
            cudaMemcpy(K.primes, pr.data(), sizeof(uint32_t) * pr.size(), cudaMemcpyHostToDevice);
        }
        Ept.reserve(2 * n);
        for (int u = 0; u < 2 * n; ++u) {
            Ept.emplace_back(cc_);
            Ept.back().c0.grow(topL);
        }
        cudaDeviceSynchronize();  // the limb buffers are written on the default stream below
        for (int u = 0; u < 2 * n; ++u)
            spruScatter_<<<(N + 255) / 256, 256>>>(in + (size_t)u * N, N, targetSFv, Ept[u].c0.GPU.at(0).limbptr.data,
                                                   K.primes, topL + 1);
        cudaDeviceSynchronize();
        for (auto& pt : Ept) {
            pt.c0.NTT(C.batch, false);
            pt.NoiseFactor = targetSFv;
            pt.NoiseLevel = 1;
            pt.slots = N / 2;
        }
        tm.encode_ms = ms_since(t0);
    } else {
    // ---- 3. host: coefficients mod q0, the two LWE columns, the candidate roots of unity
    Ciphertext w(cc_);
    w.copy(ct);
    w.c0.INTT(C.batch, true);
    w.c1.INTT(C.batch, true);
    std::vector<std::vector<uint64_t>> r0, r1;
    w.c0.store(r0);
    w.c1.store(r1);
    cudaDeviceSynchronize();
    const unsigned __int128 p0 = C.prime[0].p, p1 = C.prime[1].p, q0 = p0 * p1;
    const uint64_t inv = [&] {  // p0^{-1} mod p1
        unsigned __int128 a = p0 % p1, res = 1, e = p1 - 2;
        while (e) { if (e & 1) res = res * a % p1; a = a * a % p1; e >>= 1; }
        return (uint64_t)res;
    }();
    auto crt = [&](const std::vector<std::vector<uint64_t>>& r, int k) -> __int128 {
        const unsigned __int128 x0 = r[0][k], x1 = r[1][k];
        const unsigned __int128 t = ((x1 + p1 - x0 % p1) % p1) * inv % p1;
        return (__int128)(x0 + p0 * t);  // in [0, q0)
    };
    std::vector<__int128> c0(N), c1(N);
    for (int k = 0; k < N; ++k) { c0[k] = crt(r0, k); c1[k] = crt(r1, k); }
    const long double qd = (long double)q0;
    const double Dm = targetSF * std::ldexp(1.0, -(int)correction);  // message scale of the integers
    const double delta = std::pow((double)(qd / (4.0L * M_PIl * (long double)Dm)), 1.0 / h);
    std::vector<std::vector<std::complex<double>>> E(2 * n, std::vector<std::complex<double>>(N / 2));
    for (int a = 0; a < n; ++a) {
        const int ka = a * (N / n);  // coefficient X^{a N/n}
        for (int b = 0; b < h; ++b)
            for (int u = 0; u < 2 * n; ++u)
                for (int k = 0; k < sub; ++k) {
                    const int i = b * B + u * sub + k;  // key index
                    __int128 v = (i <= ka) ? c1[ka - i] : (__int128)q0 - c1[N + ka - i];
                    if (i == 0) v += c0[ka];
                    v %= (__int128)q0;
                    const long double ang = 2.0L * M_PIl * (long double)v / qd;
                    E[u][k * h * n + b * n + a] = std::complex<double>((double)cosl(ang), (double)sinl(ang)) * delta;
                }
    }
    tm.host_ms = ms_since(t0);
    // ---- 4. encode E_u at the key level (host OpenFHE encode in this prototype)
    const int keyLevelDropped = (int)C.L - K.cs[0]->getLevel();  // OpenFHE level = primes dropped
    for (int u = 0; u < 2 * n; ++u) {
        auto pt = cc->MakeCKKSPackedPlaintext(E[u], 1, (uint32_t)keyLevelDropped, nullptr, N / 2);
        Ept.emplace_back(cc_, GetRawPlainText(cc, pt));
    }
    tm.encode_ms = ms_since(t0);
    }
    // ---- 5. acc = sum_u cs_u * E_u, one rescale
    Ciphertext acc(cc_), t(cc_);
    acc.multPt(*K.cs[0], Ept[0], false);
    for (int u = 1; u < 2 * n; ++u) {
        t.multPt(*K.cs[u], Ept[u], false);
        acc.add(t);
    }
    acc.rescale();
    tm.extmult_ms = ms_since(t0);
    // ---- 6. trace: sums over the slots sharing the index mod h*n
    acc.slots = N / 2;
    Accumulate(acc, K.accBStep, h * n, N / 2 / (h * n));
    acc.slots = N / 2;
    tm.trace_ms = ms_since(t0);
    // ---- 7. product over the h blocks (slots b*n + a within the h*n period)
    for (int s = n; s < h * n; s <<= 1) {
        Ciphertext r(cc_);
        r.rotate(acc, s);
        acc.mult(r, false);
        acc.rescale();
    }
    tm.product_ms = ms_since(t0);
    // ---- 8. 2 Im: (x - conj x) * (-i);  9. z = m0 + i m1 in every slot
    Ciphertext cj(cc_);
    cj.conjugate(acc);
    acc.sub(cj);
    acc.multMonomial(3 * N / 2);  // * (-i)
    {
        auto it = K.maskAt.find(acc.getLevel());
        if (it == K.maskAt.end()) {
            std::vector<std::complex<double>> mask(N / 2);
            for (int j = 0; j < N / 2; ++j) mask[j] = (j % 2 == 0) ? std::complex<double>(1, 0) : std::complex<double>(0, 1);
            auto pm = cc->MakeCKKSPackedPlaintext(mask, 1, (uint32_t)((int)C.L - acc.getLevel()), nullptr, N / 2);
            it = K.maskAt.emplace(acc.getLevel(), Plaintext(cc_, GetRawPlainText(cc, pm))).first;
        }
        Plaintext& pmask = it->second;
        // out = acc*p + rot(acc, 1)*rot(p, 1): the recombination without rescaling, so the output is a lazy deg-2
        // ciphertext like every other bootstrap route's landing (the planner's degree pins assume it)
        auto it2 = K.maskAt.find(-1 - acc.getLevel());
        if (it2 == K.maskAt.end()) {
            std::vector<std::complex<double>> mask(N / 2);
            for (int j = 0; j < N / 2; ++j) mask[j] = (j % 2 == 1) ? std::complex<double>(1, 0) : std::complex<double>(0, 1);
            auto pm = cc->MakeCKKSPackedPlaintext(mask, 1, (uint32_t)((int)C.L - acc.getLevel()), nullptr, N / 2);
            it2 = K.maskAt.emplace(-1 - acc.getLevel(), Plaintext(cc_, GetRawPlainText(cc, pm))).first;
        }
        Ciphertext rr(cc_);
        rr.rotate(acc, 1);
        rr.multPt(it2->second, false);
        acc.multPt(pmask, false);
        acc.add(rr);
    }
    tm.finish_ms = ms_since(t0);
    tm.total_ms = std::chrono::duration<double, std::milli>(t0 - tStart).count();
    acc.slots = ct.slots;
    ct.copy(acc);
    if (times) *times = tm;
}

}  // namespace FIDESlib::CKKS
