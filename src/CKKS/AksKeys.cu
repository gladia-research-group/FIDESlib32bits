#include "CKKS/AksKeys.cuh"
#include <cassert>
#include <cmath>
#include <string>
#include <cstdio>
#include <random>
#include <stdexcept>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "CKKS/SmallInt.cuh"
#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "Math.cuh"
#include "Rotation.cuh"

namespace FIDESlib::CKKS {

namespace {
// primeid of limb `limb` (Q limbs first, then the specials); qInit/sInit = PARTITION(0, 0) / SPECIAL(0, 0) from the host
__device__ __forceinline__ int aksPrimeid(int limb, int nq, int qInit, int sInit) {
    return limb < nq ? C_.primeid_flattened[qInit + limb] : C_.primeid_flattened[sInit + limb - nq];
}
__device__ __forceinline__ uint32_t barrett64(uint64_t v, uint32_t p, uint64_t mu) {
    const uint64_t q = __umul64hi(v, mu);
    uint64_t r = v - q * (uint64_t)p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}

// uniform residues (splitmix64 of (seed, salt, limb, idx)); uniform in NTT form == uniform polynomial
__global__ void aksFillUniform_(void** limbs, const int primeid_init, const uint64_t seed, const uint32_t salt) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const uint64_t p = C_.primes[primeid];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t z = seed ^ ((uint64_t)salt << 48) ^ ((uint64_t)blockIdx.y << 40) ^ (uint64_t)idx;
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    ((uint32_t*)limbs[blockIdx.y])[idx] = (uint32_t)(z % p);
}

// b = X + E - a * s  (mod p), per limb over Q then P
__global__ void aksKeyB_(void** bQ, void** bS, void** xQ, void** xS, void** eQ, void** eS, void** aQ, void** aS,
                         void** sQ, void** sS, const int nq, const int qInit, const int sInit) {
    const int limb = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool q = limb < nq;
    const int l = q ? limb : limb - nq;
    const uint64_t p = C_.primes[aksPrimeid(limb, nq, qInit, sInit)];
    const uint32_t X = ((const uint32_t*)(q ? xQ[l] : xS[l]))[idx];
    const uint32_t E = ((const uint32_t*)(q ? eQ[l] : eS[l]))[idx];
    const uint32_t A = ((const uint32_t*)(q ? aQ[l] : aS[l]))[idx];
    const uint32_t S = ((const uint32_t*)(q ? sQ[l] : sS[l]))[idx];
    const uint64_t as = ((uint64_t)A * S) % p;
    uint64_t b = ((uint64_t)X + E) % p;
    b = (b + p - as) % p;
    ((uint32_t*)(q ? bQ[l] : bS[l]))[idx] = (uint32_t)b;
}

// acc1 = sum_i rot_i(ext) * a_i, acc0 = sum_i rot_i(ext) * b_i, over Q+P. rot_i(ext)[idx] = ext[sigma_{5^rot}(idx)]
// (the inverse of the scatter automorph_multi_ applies). Lazy 64-bit accumulation: r <= 255.
__global__ void aksDot_(void** extQ, void** extS, void** keys, void** acc0Q, void** acc0S, void** acc1Q, void** acc1S,
                        const int* kinv, const int r, const int nq, const int nl, const int logN, const int qInit,
                        const int sInit) {
    const int limb = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool q = limb < nq;
    const int l = q ? limb : limb - nq;
    const uint32_t p = (uint32_t)C_.primes[aksPrimeid(limb, nq, qInit, sInit)];
    const uint64_t mu = ~0ull / p;
    const uint32_t* e = (const uint32_t*)(q ? extQ[l] : extS[l]);
    uint64_t s0 = 0, s1 = 0;
    for (int i = 0; i < r; ++i) {
        const uint32_t v = e[automorph_slot(logN, kinv[i], (uint32_t)idx)];
        const uint32_t ka = ((const uint32_t*)keys[(2 * i) * nl + limb])[idx];
        const uint32_t kb = ((const uint32_t*)keys[(2 * i + 1) * nl + limb])[idx];
        s1 += (uint64_t)v * ka;
        s0 += (uint64_t)v * kb;
    }
    ((uint32_t*)(q ? acc0Q[l] : acc0S[l]))[idx] = barrett64(s0, p, mu);
    ((uint32_t*)(q ? acc1Q[l] : acc1S[l]))[idx] = barrett64(s1, p, mu);
}

// acc = sum_i rot_i(c0) * m_i over Q
__global__ void aksC0_(void** c0, void** pts, void** acc, const int* kinv, const int r, const int nq, const int logN,
                       const int qInit) {
    const int limb = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t p = (uint32_t)C_.primes[C_.primeid_flattened[qInit + limb]];
    const uint64_t mu = ~0ull / p;
    const uint32_t* c = (const uint32_t*)c0[limb];
    uint64_t s = 0;
    for (int i = 0; i < r; ++i) {
        const uint32_t v = c[automorph_slot(logN, kinv[i], (uint32_t)idx)];
        s += (uint64_t)v * ((const uint32_t*)pts[i * nq + limb])[idx];
    }
    ((uint32_t*)acc[limb])[idx] = barrett64(s, p, mu);
}

// Self-test of the gather convention: scatter (automorph_multi_) of identity == gather with 5^rot.
__global__ void aksPermCheck_(const uint32_t* scattered, const int kinv, const int logN, int* bad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // scattered[sigma_k(j)] = j  ->  scattered[idx] == sigma_k^{-1}(idx) == automorph_slot(kinv, idx)
    if (scattered[idx] != automorph_slot(logN, kinv, (uint32_t)idx))
        atomicAdd(bad, 1);
}

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
std::vector<void*> hostPtrs(VectorGPU<void*>& tab, int n) {
    std::vector<void*> h(n);
    if (n)
        cudaMemcpy(h.data(), tab.data, (size_t)n * sizeof(void*), cudaMemcpyDeviceToHost);
    return h;
}
int normIdx(long v, int N) {
    v %= N / 2;
    if (v < 0)
        v += N / 2;
    return (int)v;
}
}  // namespace

static void buildKeyTables(ContextData& cc, AksStage& aks) {
    const int L = cc.L, K = (int)cc.specialPrime.size(), r = aks.r, nl = L + 1 + K;
    std::vector<void*> keys((size_t)2 * r * nl);
    for (int i = 0; i < r; ++i)
        for (int w = 0; w < 2; ++w) {
            RNSPoly& kp = w == 0 ? *aks.a[i] : *aks.b[i];
            auto hq = hostPtrs(kp.GPU.at(0).limbptr, L + 1), hs = hostPtrs(kp.GPU.at(0).SPECIALlimbptr, K);
            for (int l = 0; l < L + 1; ++l) keys[(2 * i + w) * nl + l] = hq[l];
            for (int l = 0; l < K; ++l) keys[(2 * i + w) * nl + L + 1 + l] = hs[l];
        }
    cudaMalloc((void**)&aks.keyTab, keys.size() * sizeof(void*));
    cudaMemcpy(aks.keyTab, keys.data(), keys.size() * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&aks.kinvDev, r * sizeof(int));
    cudaMemcpy(aks.kinvDev, aks.kinv.data(), r * sizeof(int), cudaMemcpyHostToDevice);
}

std::shared_ptr<AksStage> MakeGhsKeyStage(Context& cc_, const std::vector<std::vector<uint64_t>>& a,
                                          const std::vector<std::vector<uint64_t>>& b,
                                          const std::vector<uint64_t>& moduli) {
    ContextData& cc = *cc_;
    auto st = std::make_shared<AksStage>();
    st->r = 1;
    st->rot = {0};
    st->kinv = {1};
    st->D = 1;
    auto A = std::make_unique<RNSPoly>(cc, -1);
    auto B = std::make_unique<RNSPoly>(cc, -1);
    A->load(a, moduli);  // load() allocates Q limbs and, for the extra moduli, the special limbs
    B->load(b, moduli);
    A->SetModUp(true);
    B->SetModUp(true);
    cudaDeviceSynchronize();
    st->a.push_back(std::move(A));
    st->b.push_back(std::move(B));
    buildKeyTables(cc, *st);
    cudaDeviceSynchronize();
    return st;
}

void ghsSwitchSmall(Ciphertext& ct, AksStage& key) {
    ContextData& cc = ct.cc;
    const int L = cc.L, N = cc.N, K = (int)cc.specialPrime.size();
    assert(ct.getLevel() == L && key.r == 1);
    cudaSetDevice(cc.GPUid[0]);
    RNSPoly ext(cc, L);
    ext.copy(ct.c1);
    smallIntLiftToSpecial(cc, ext, 3);
    RNSPoly acc0(cc, L), acc1(cc, L);
    acc0.generateSpecialLimbs(false, false);
    acc1.generateSpecialLimbs(false, false);
    {
        auto &PE = ext.GPU.at(0), &P0 = acc0.GPU.at(0), &P1 = acc1.GPU.at(0);
        for (auto* P : {&PE, &P0, &P1}) joinLimbs(*P);
        P0.s.wait(PE.s);
        P0.s.wait(P1.s);
        aksDot_<<<dim3{(uint32_t)N / 128, (uint32_t)(L + 1 + K)}, 128, 0, P0.s.ptr()>>>(
            PE.limbptr.data, PE.SPECIALlimbptr.data, key.keyTab, P0.limbptr.data, P0.SPECIALlimbptr.data,
            P1.limbptr.data, P1.SPECIALlimbptr.data, key.kinvDev, 1, L + 1, L + 1 + K, cc.logN, PARTITION(0, 0),
            SPECIAL(0, 0));
        P1.s.wait(P0.s);
        PE.s.wait(P0.s);
        forkLimbs(P0);
        forkLimbs(P1);
        forkLimbs(PE);
    }
    acc0.SetModUp(true);
    acc1.SetModUp(true);
    acc0.moddown(true, false, 0);
    acc1.moddown(true, false, 1);
    ct.c0.add(acc0);
    ct.c1.copy(acc1);
}

AksStage::~AksStage() {
    if (keyTab) cudaFree(keyTab);
    if (ptTab) cudaFree(ptTab);
    if (kinvDev) cudaFree(kinvDev);
}

void GenerateAksStage0(Context& cc_, BootstrapPrecomputation& pre, const std::vector<std::vector<uint64_t>>& skLimbs,
                       const std::vector<uint64_t>& skModuli, const std::vector<std::vector<uint64_t>>& sparseLimbs,
                       uint64_t seed) {
    ContextData& cc = *cc_;
    if (pre.CtS.empty())
        return;
    auto& step = pre.CtS.at(0);
    const int L = cc.L, N = cc.N, K = (int)cc.specialPrime.size();
    const int r = step.slots, bStep = step.bStep;
    if (step.A.at(0).c0.getLevel() != L)
        throw std::runtime_error("AKS: the stage-0 plaintexts are not at the top level — set FIDESLIB_BTS_SHIFT=1");
    if (r > 255)
        throw std::runtime_error("AKS: more than 255 diagonals (lazy accumulation bound)");
    const int stride = bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
    const int offset = step.rotOut[0];
    auto aks = std::make_shared<AksStage>();
    aks->r = r;
    {
        // divisor sf_pt / (c 2^t): the EvalMod constant rides the keys, the output keeps 2^t so the switch's
        // absolute noise (~2^9/coef) stays 2^t below the message (lesson of lever 1a)
        if (pre.cts0_const <= 0)
            throw std::runtime_error("GenerateAksStage0: needs FIDESLIB_BTS_SHIFT>=1 (cts0_const unset)");
        aks->t = pre.cts0_t;
        const long double Dt = (long double)step.A.at(0).NoiseFactor /
                               ((long double)pre.cts0_const * std::ldexp(1.0L, aks->t));
        const SmallIntTab probe = buildSmallIntTab({cc.prime[0].p}, {cc.prime[0].p}, Dt);
        aks->D = smallIntEffectiveDivisor(probe);
        std::fprintf(stderr, "[aks] stage 0: r=%d, D = 2^%.3Lf (sf_pt 2^%.2f / (c 2^%d)), t=%d\n", r, log2l(aks->D),
                     std::log2(step.A.at(0).NoiseFactor), aks->t, aks->t);
    }
    cudaSetDevice(cc.GPUid[0]);

    // dense secret s (encrypts the keys) and sparse secret s~ (inside the keys), both over Q+P (ternary: exact
    // lift from one limb)
    RNSPoly s(cc, L);
    s.load(skLimbs, skModuli);
    smallIntLiftToSpecial(cc, s, 1);
    RNSPoly st(cc, L);
    st.load(sparseLimbs, skModuli);
    smallIntLiftToSpecial(cc, st, 1);
    if (const char* td = std::getenv("BTS_TRACE_DIR")) {  // diagnostics: both secrets in coefficient form
        Plaintext ps(cc_), pst(cc_);
        ps.c0.grow(L); pst.c0.grow(L);
        ps.c0.copy(s); pst.c0.copy(st);
        ps.NoiseFactor = pst.NoiseFactor = 1.0; ps.slots = pst.slots = N / 2;
        exactPlainDump(ps, (std::string(td) + "/secret-s.ct").c_str());
        exactPlainDump(pst, (std::string(td) + "/secret-st.ct").c_str());
    }

    // gather-convention self-test on one limb
    if (const char* e = std::getenv("FIDESLIB_AKS_SELFTEST"); e && std::atoi(e)) {
        uint32_t *ident, *scat;
        int* bad;
        cudaMalloc(&ident, N * 4);
        cudaMalloc(&scat, N * 4);
        cudaMalloc(&bad, 4);
        std::vector<uint32_t> h(N);
        for (int i = 0; i < N; ++i) h[i] = i;
        cudaMemcpy(ident, h.data(), N * 4, cudaMemcpyHostToDevice);
        cudaMemset(bad, 0, 4);
        void* ha[1] = {ident};
        void* hb[1] = {scat};
        void **da, **db;
        cudaMalloc(&da, 8);
        cudaMalloc(&db, 8);
        cudaMemcpy(da, ha, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, 8, cudaMemcpyHostToDevice);
        const int rot = 5;
        const int k = (int)modpow(5, 2 * N - rot, 2 * N), kinv = (int)modpow(5, rot, 2 * N);
        automorph_multi_<<<dim3{(uint32_t)N / 128, 1}, 128>>>(da, db, k, 1, PARTITION(0, 0));
        aksPermCheck_<<<N / 128, 128>>>(scat, kinv, cc.logN, bad);
        int hbad = -1;
        cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "[aks] gather self-test (rot 5): %d mismatching slots\n", hbad);
        cudaFree(ident); cudaFree(scat); cudaFree(bad); cudaFree(da); cudaFree(db);
    }

    const uint64_t q0 = cc.prime[0].p;
    // limbs that determine the key payload Y = P m s~_rot exactly: |Y| < P sf_pt ||s~||_1 (h~ = 32) with 8 bits
    // of margin; 13 at dnum 6 (P ~ 2^243), 16 at dnum 4 (P ~ 2^351). Scales with the special-prime count.
    int kY = 0;
    {
        long double lP = 0;
        for (const auto& sp : cc.specialPrime) lP += std::log2((long double)sp.p);
        const long double need = lP + std::log2((long double)step.A.at(0).NoiseFactor) + 5 + 8;
        long double have = 0;
        for (kY = 0; kY < SI_MAXK && have < need; ++kY) have += std::log2((long double)cc.prime[kY].p);
        if (have < need)
            throw std::runtime_error("GenerateAksStage0: key payload needs more than SI_MAXK limbs (P too large)");
        std::fprintf(stderr, "[aks] key payload: log2 P = %.1Lf, k = %d limbs (2^%.0Lf)\n", lP, kY, have);
    }
    for (int i = 0; i < r; ++i) {
        const int j = i / bStep, b = i % bStep;
        const int rot = normIdx((long)(j * bStep + b) * stride + offset, N);
        const int G = normIdx((long)j * bStep * stride + offset, N);
        // m_i: undo the BSGS pre-rotation of the stored diagonal
        auto m = std::make_unique<Plaintext>(cc_);
        m->copy(step.A.at(i));
        if (G)
            m->automorph(G);
        // s^{(rot_i)}
        RNSPoly srot(cc, L);
        if (rot)
            srot.automorph(rot, 1, &st);
        else
            srot.copy(st);
        // X = round(P m_i s_rot / D) over Q+P
        RNSPoly Y(cc, L);
        Y.copy(m->c0);
        Y.multElement(srot);
        Y.scaleByP();
        smallIntDivideKeepLevel(cc, Y, kY, aks->D, true);
        // a: uniform over Q+P
        RNSPoly A(cc, L);
        A.generateSpecialLimbs(false, false);
        {
            auto& PA = A.GPU.at(0);
            joinLimbs(PA);
            aksFillUniform_<<<dim3{(uint32_t)N / 128, (uint32_t)(L + 1)}, 128, 0, PA.s.ptr()>>>(
                PA.limbptr.data, PARTITION(0, 0), seed, (uint32_t)(2 * i));
            aksFillUniform_<<<dim3{(uint32_t)N / 128, (uint32_t)K}, 128, 0, PA.s.ptr()>>>(
                PA.SPECIALlimbptr.data, SPECIAL(0, 0), seed, (uint32_t)(2 * i + 1));
            forkLimbs(PA);
            A.SetModUp(true);
        }
        // e: discrete Gaussian (sigma 3.19) lifted exactly to Q+P
        RNSPoly E(cc, 0);
        {
            std::mt19937_64 g(seed * 7919ull + 104729ull * (uint64_t)i);
            std::normal_distribution<double> nd(0.0, 3.19);
            std::vector<std::vector<uint64_t>> e1(1, std::vector<uint64_t>(N));
            for (int n = 0; n < N; ++n) {
                const long v = std::lround(nd(g));
                e1[0][n] = v >= 0 ? (uint64_t)v : (uint64_t)((int64_t)q0 + v);
            }
            E.load(e1, {q0});
            E.grow(L);
            E.generateSpecialLimbs(false, false);
            smallIntDivideKeepLevel(cc, E, 1, 0, true, /*srcCoeff=*/true);
        }
        // b = X + e - a s
        RNSPoly B(cc, L);
        B.generateSpecialLimbs(false, false);
        {
            auto &PB = B.GPU.at(0), &PY = Y.GPU.at(0), &PE = E.GPU.at(0), &PA = A.GPU.at(0), &PS = s.GPU.at(0);
            for (auto* P : {&PB, &PY, &PE, &PA, &PS}) joinLimbs(*P);
            for (auto* P : {&PY, &PE, &PA, &PS}) PB.s.wait(P->s);
            aksKeyB_<<<dim3{(uint32_t)N / 128, (uint32_t)(L + 1 + K)}, 128, 0, PB.s.ptr()>>>(
                PB.limbptr.data, PB.SPECIALlimbptr.data, PY.limbptr.data, PY.SPECIALlimbptr.data, PE.limbptr.data,
                PE.SPECIALlimbptr.data, PA.limbptr.data, PA.SPECIALlimbptr.data, PS.limbptr.data, PS.SPECIALlimbptr.data,
                L + 1, PARTITION(0, 0), SPECIAL(0, 0));
            for (auto* P : {&PY, &PE, &PA, &PS}) P->s.wait(PB.s);
            forkLimbs(PB);
            B.SetModUp(true);
        }
        cudaDeviceSynchronize();
        aks->a.push_back(std::make_unique<RNSPoly>(std::move(A)));
        aks->b.push_back(std::make_unique<RNSPoly>(std::move(B)));
        aks->m.push_back(std::move(m));
        aks->rot.push_back(rot);
        aks->kinv.push_back(rot ? (int)modpow(5, rot, 2 * N) : 1);
    }
    cudaDeviceSynchronize();
    // device pointer tables
    buildKeyTables(cc, *aks);
    {
        std::vector<void*> pts((size_t)r * (L + 1));
        for (int i = 0; i < r; ++i) {
            auto hp = hostPtrs(aks->m[i]->c0.GPU.at(0).limbptr, L + 1);
            for (int l = 0; l < L + 1; ++l) pts[(size_t)i * (L + 1) + l] = hp[l];
        }
        cudaMalloc((void**)&aks->ptTab, pts.size() * sizeof(void*));
        cudaMemcpy(aks->ptTab, pts.data(), pts.size() * sizeof(void*), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    std::fprintf(stderr, "[aks] stage 0: %d diagonals (bStep %d, stride %d, offset %d), GHS keys %.0f MB, D = 2^%.4Lf\n",
                 r, bStep, stride, offset, 2.0 * r * (L + 1 + K) * N * 4 / 1048576.0, std::log2(aks->D) / 1.0L);
    pre.aks0 = std::move(aks);
}

void LinearTransformAKS(Ciphertext& ct, BootstrapPrecomputation::LTstep& step, AksStage& aks) {
    ContextData& cc = ct.cc;
    const int L = cc.L, N = cc.N, K = (int)cc.specialPrime.size();
    assert(ct.getLevel() == L && ct.NoiseLevel == 1 && "AKS stage 0 expects the raised ciphertext");
    cudaSetDevice(cc.GPUid[0]);
    // c1 -> Q+P exactly (|c1| < q0 after compositeModRaise's per-term centring: three limbs determine it)
    RNSPoly ext(cc, L);
    ext.copy(ct.c1);
    smallIntLiftToSpecial(cc, ext, 3);
    RNSPoly acc0(cc, L), acc1(cc, L);
    acc0.generateSpecialLimbs(false, false);
    acc1.generateSpecialLimbs(false, false);
    {
        auto &PE = ext.GPU.at(0), &P0 = acc0.GPU.at(0), &P1 = acc1.GPU.at(0);
        for (auto* P : {&PE, &P0, &P1}) joinLimbs(*P);
        P0.s.wait(PE.s);
        P0.s.wait(P1.s);
        aksDot_<<<dim3{(uint32_t)N / 128, (uint32_t)(L + 1 + K)}, 128, 0, P0.s.ptr()>>>(
            PE.limbptr.data, PE.SPECIALlimbptr.data, aks.keyTab, P0.limbptr.data, P0.SPECIALlimbptr.data,
            P1.limbptr.data, P1.SPECIALlimbptr.data, aks.kinvDev, aks.r, L + 1, L + 1 + K, cc.logN, PARTITION(0, 0),
            SPECIAL(0, 0));
        P1.s.wait(P0.s);
        PE.s.wait(P0.s);
        forkLimbs(P0);
        forkLimbs(P1);
        forkLimbs(PE);
    }
    acc0.SetModUp(true);
    acc1.SetModUp(true);
    acc0.moddown(true, false, 0);
    acc1.moddown(true, false, 1);
    // c0 path: sum_i m_i rot_i(c0), then the exact rescale that keeps the level
    RNSPoly accc(cc, L);
    {
        auto &PC = accc.GPU.at(0), &PS = ct.c0.GPU.at(0);
        joinLimbs(PS);
        joinLimbs(PC);
        PC.s.wait(PS.s);
        aksC0_<<<dim3{(uint32_t)N / 128, (uint32_t)(L + 1)}, 128, 0, PC.s.ptr()>>>(
            PS.limbptr.data, aks.ptTab, PC.limbptr.data, aks.kinvDev, aks.r, L + 1, cc.logN, PARTITION(0, 0));
        PS.s.wait(PC.s);
        forkLimbs(PC);
        forkLimbs(PS);
    }
    smallIntDivideKeepLevel(cc, accc, 7, aks.D, false);
    accc.add(acc0);
    ct.c0.copy(accc);
    ct.c1.copy(acc1);
    ct.NoiseFactor *= std::ldexp(1.0, aks.t);  // value c*m*V at scale NF 2^t (D = sf_pt / (c 2^t))
    ct.NoiseLevel = 1;
}

}  // namespace FIDESlib::CKKS
