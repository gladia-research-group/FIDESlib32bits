#include "CKKS/LinearKS.cuh"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {
namespace {
void joinLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) P.s.wait(STREAM(P.limb[i]));
    for (auto& l : P.SPECIALlimb) P.s.wait(STREAM(l));
}
void forkLimbs(LimbPartition& P) {
    const int n = P.getLimbSize(*P.level);
    for (int i = 0; i < n; ++i) STREAM(P.limb[i]).wait(P.s);
    for (auto& l : P.SPECIALlimb) STREAM(l).wait(P.s);
}
__device__ __forceinline__ uint32_t barrett64(uint64_t v, uint32_t p, uint64_t mu) {
    uint64_t q = __umul64hi(v, mu);
    uint64_t r = v - q * p;
    if (r >= p) r -= p;
    if (r >= p) r -= p;
    return (uint32_t)r;
}
__device__ __forceinline__ uint32_t hash32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c * 0xC2B2AE3Du);
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
// limb l of a poly (via its pointer table) := random residue mod primes[l]
__global__ void fillRandom_(void** limbs, const uint32_t* primes, uint32_t seed) {
    const int l = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    ((uint32_t*)limbs[l])[n] = hash32(seed, l, n) % primes[l];
}
// flat random keys: v[idx][n] < tprime[idx % 3]
__global__ void fillKeys_(uint32_t* v, size_t nlimbs, const uint32_t* tprime, uint32_t seed, size_t N) {
    const size_t l = blockIdx.y + (size_t)gridDim.y * blockIdx.z, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= nlimbs) return;
    v[l * N + n] = hash32(seed, (uint32_t)l, (uint32_t)n) % tprime[l % 3];
}
// linear lift: digit i (coefficient form, limb i of c') -> b[i] limbs k<3: r mod t_k
__global__ void liftT_(void** cprime, void** bOut, const uint32_t* tprime) {
    const int i = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = ((const uint32_t*)cprime[i])[n];
    for (int k = 0; k < 3; ++k) ((uint32_t*)bOut[i * 3 + k])[n] = r % tprime[k];
}
// baseline lift: digit i -> all nq limbs: r mod q_j
__global__ void liftQ_(void** cprime, int i, void** rOut, const uint32_t* qprimes) {
    const int j = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = ((const uint32_t*)cprime[i])[n];
    ((uint32_t*)rOut[j])[n] = r % qprimes[j];
}
// acc[j][k] = sum_i b[i][k] * v[(i*nout + j)*3 + k] (mod t_k), nd digits; for c0 and c1 key halves
__global__ void dotT_(void** b, const uint32_t* v, int nd, int nout, void** acc0, void** acc1, const uint32_t* tprime,
                      size_t N) {
    const int jk = blockIdx.y, j = jk / 3, k = jk % 3, n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t p = tprime[k];
    const uint64_t mu = ~0ull / p;
    uint64_t s0 = 0, s1 = 0;
    for (int i = 0; i < nd; ++i) {
        const uint64_t bv = ((const uint32_t*)b[i * 3 + k])[n];
        const size_t base = ((size_t)(i * nout + j) * 3 + k) * 2;
        s0 += bv * v[base * N + n];
        s1 += bv * v[(base + 1) * N + n];
    }
    ((uint32_t*)acc0[jk])[n] = barrett64(s0, p, mu);
    ((uint32_t*)acc1[jk])[n] = barrett64(s1, p, mu);
}
// CRT over T (3 limbs, coefficient form) -> integer < T -> mod q_j, written to output limb j
struct CrtTab { uint32_t t0, t1, t2; uint32_t inv01;  /* t0^-1 mod t1 */ uint32_t inv012; /* (t0 t1)^-1 mod t2 */ };
__global__ void crtOut_(void** acc, int j, void** out, const uint32_t qj, const CrtTab tab) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t a0 = ((const uint32_t*)acc[0])[n], a1 = ((const uint32_t*)acc[1])[n], a2 = ((const uint32_t*)acc[2])[n];
    // Garner: x = a0 + t0 * y1 + t0 t1 * y2
    const uint64_t y1 = ((a1 + tab.t1 - a0 % tab.t1) % tab.t1) * tab.inv01 % tab.t1;
    const unsigned __int128 t01 = (unsigned __int128)tab.t0 * tab.t1;
    const unsigned __int128 x01 = a0 + (unsigned __int128)tab.t0 * y1;
    const uint64_t x01m2 = (uint64_t)(x01 % tab.t2);
    const uint64_t y2 = ((a2 + tab.t2 - x01m2) % tab.t2) * tab.inv012 % tab.t2;
    const unsigned __int128 x = x01 + t01 * y2;
    // the negacyclic convolution is signed: recover the centred representative (|x| < T/2) before reducing
    const unsigned __int128 T = t01 * tab.t2;
    const uint32_t r = (x > T / 2) ? (uint32_t)((T - x) % qj) : 0;
    ((uint32_t*)out[j])[n] = (x > T / 2) ? (r ? qj - r : 0) : (uint32_t)(x % qj);
}
uint64_t powmod(uint64_t b, uint64_t e, uint64_t m) { uint64_t r = 1; b %= m; while (e) { if (e & 1) r = (unsigned __int128)r * b % m; b = (unsigned __int128)b * b % m; e >>= 1; } return r; }
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

long linearKsBench(Ciphertext& any, int iters, std::ostream& os) {
    ContextData& cc = any.cc;
    cudaSetDevice(cc.GPUid[0]);
    const int L = cc.L, N = cc.N, nq = L + 1, K = (int)cc.specialPrime.size(), nout = nq + K, nd = nq;
    std::vector<uint32_t> hq(nout);
    for (int j = 0; j < nq; ++j) hq[j] = (uint32_t)cc.prime[j].p;
    for (int k = 0; k < K; ++k) hq[nq + k] = (uint32_t)cc.specialPrime[k].p;
    uint32_t* dq; cudaMalloc(&dq, nout * 4); cudaMemcpy(dq, hq.data(), nout * 4, cudaMemcpyHostToDevice);
    const uint32_t* dt = dq;  // T = {q0, q1, q2}
    CrtTab tab{hq[0], hq[1], hq[2], (uint32_t)powmod(hq[0], hq[1] - 2, hq[1]),
               (uint32_t)powmod((uint64_t)hq[0] * hq[1] % hq[2], hq[2] - 2, hq[2])};
    const long double lT = std::log2((long double)hq[0]) + std::log2((long double)hq[1]) + std::log2((long double)hq[2]);
    const long double lBound = std::log2((long double)N) + std::log2((long double)nd) + 2 * 28;
    os << "[links] N=" << N << " Q limbs=" << nq << " specials=" << K << " digits(single-limb)=" << nd
       << " T=2^" << (double)lT << " bound=2^" << (double)lBound << (lT > lBound + 1 ? " (exact)" : " (NOT exact)") << "\n";

    // operand c1 at the top level, random residues
    RNSPoly c(cc, L);
    auto& PC = c.GPU.at(0);
    joinLimbs(PC);
    fillRandom_<<<dim3{(uint32_t)N / 128, (uint32_t)nq}, 128, 0, PC.s.ptr()>>>(PC.limbptr.data, dq, 11u);
    forkLimbs(PC);
    cudaDeviceSynchronize();

    // keys: flat v[(i*nout+j)*3+k][c0|c1][n] < t_k ; c0 half consistent with kappa (gate), c1 half random
    const size_t nkl = (size_t)nd * nout * 3 * 2;
    uint32_t* v = nullptr;
    if (cudaMalloc(&v, nkl * (size_t)N * 4) != cudaSuccess) { os << "[links] cannot allocate " << nkl * N * 4 / 1e9 << " GB of keys\n"; return -1; }
    os << "[links] key stream per switch: " << nkl * (double)N * 4 / 1e9 << " GB (hybrid: " << 2.0 * 6 * nout * N * 4 / 1e9 << " GB)\n";
    {
        dim3 g((uint32_t)N / 128, 1024, (uint32_t)((nkl + 1023) / 1024));
        fillKeys_<<<g, 128>>>(v, nkl, dt, 77u, (size_t)N);
        cudaDeviceSynchronize();
    }
    // gate inputs: kappa[i][j] := the integer the c0 key limb represents in T... for the gate we DEFINE kappa[i][j] as the
    // CRT of its three T residues only when those are consistent; simplest exact construction: overwrite the c0 half with
    // residues of a random kappa[i][j] < q_j (so the T residues agree), limb by limb, then NTT in T.
    std::vector<std::unique_ptr<RNSPoly>> kap;  // kappa per digit i as an RNSPoly over Q (limb j = kappa[i][j]), coeff form
    kap.reserve(nd);
    for (int i = 0; i < nd; ++i) {
        kap.emplace_back(std::make_unique<RNSPoly>(cc, L));
        auto& P = kap.back()->GPU.at(0);
        joinLimbs(P);
        fillRandom_<<<dim3{(uint32_t)N / 128, (uint32_t)nq}, 128, 0, P.s.ptr()>>>(P.limbptr.data, dq, 1000u + i);
        forkLimbs(P);
    }
    cudaDeviceSynchronize();
    // T-form keys for the gate (c0 half, j < nq): lift kappa[i][j] into T, NTT in T, store into v
    {
        RNSPoly tmp(cc, 2);
        auto& PT = tmp.GPU.at(0);
        std::vector<void*> hp(3);
        cudaMemcpy(hp.data(), PT.limbptr.data, 3 * sizeof(void*), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nd; ++i) {
            std::vector<void*> src(nq);
            cudaMemcpy(src.data(), kap[i]->GPU.at(0).limbptr.data, nq * sizeof(void*), cudaMemcpyDeviceToHost);
            for (int j = 0; j < nq; ++j) {
                // tmp limbs k := kappa[i][j] mod t_k   (kappa < q_j)
                void** one; cudaMalloc(&one, sizeof(void*)); cudaMemcpy(one, &src[j], sizeof(void*), cudaMemcpyHostToDevice);
                liftT_<<<dim3{(uint32_t)N / 128, 1}, 128>>>(one, PT.limbptr.data, dt);
                cudaDeviceSynchronize();
                tmp.NTT(cc.batch, true);
                cudaDeviceSynchronize();
                for (int k = 0; k < 3; ++k) {
                    const size_t base = ((size_t)(i * nout + j) * 3 + k) * 2;
                    cudaMemcpy(v + base * (size_t)N, hp[k], (size_t)N * 4, cudaMemcpyDeviceToDevice);
                }
                tmp.INTT(cc.batch, true);  // back to coefficient form for the next fill (cheap, keeps the object sane)
                cudaFree(one);
            }
        }
        cudaDeviceSynchronize();
    }
    // Q-form keys for the baseline: Kq[i] = NTT_Q(kappa[i])
    for (int i = 0; i < nd; ++i) kap[i]->NTT(cc.batch, true);
    cudaDeviceSynchronize();

    // ---------------- linear path ----------------
    std::vector<std::unique_ptr<RNSPoly>> b;   // nd polys at level 2
    for (int i = 0; i < nd; ++i) b.emplace_back(std::make_unique<RNSPoly>(cc, 2));
    std::vector<std::unique_ptr<RNSPoly>> acc0, acc1;  // nout polys at level 2
    for (int j = 0; j < nout; ++j) { acc0.emplace_back(std::make_unique<RNSPoly>(cc, 2)); acc1.emplace_back(std::make_unique<RNSPoly>(cc, 2)); }
    // device pointer tables
    std::vector<void*> hb(nd * 3), ha0(nout * 3), ha1(nout * 3);
    for (int i = 0; i < nd; ++i) cudaMemcpy(&hb[i * 3], b[i]->GPU.at(0).limbptr.data, 3 * sizeof(void*), cudaMemcpyDeviceToHost);
    for (int j = 0; j < nout; ++j) {
        cudaMemcpy(&ha0[j * 3], acc0[j]->GPU.at(0).limbptr.data, 3 * sizeof(void*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ha1[j * 3], acc1[j]->GPU.at(0).limbptr.data, 3 * sizeof(void*), cudaMemcpyDeviceToHost);
    }
    void **db, **da0, **da1;
    cudaMalloc(&db, hb.size() * sizeof(void*)); cudaMemcpy(db, hb.data(), hb.size() * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMalloc(&da0, ha0.size() * sizeof(void*)); cudaMemcpy(da0, ha0.data(), ha0.size() * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMalloc(&da1, ha1.size() * sizeof(void*)); cudaMemcpy(da1, ha1.data(), ha1.size() * sizeof(void*), cudaMemcpyHostToDevice);
    RNSPoly d0(cc, L), d1(cc, L);
    d0.generateSpecialLimbs(false, false);
    d1.generateSpecialLimbs(false, false);
    std::vector<void*> hd0(nout), hd1(nout);
    cudaMemcpy(hd0.data(), d0.GPU.at(0).limbptr.data, nq * sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd0.data() + nq, d0.GPU.at(0).SPECIALlimbptr.data, K * sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd1.data(), d1.GPU.at(0).limbptr.data, nq * sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd1.data() + nq, d1.GPU.at(0).SPECIALlimbptr.data, K * sizeof(void*), cudaMemcpyDeviceToHost);
    void **dd0, **dd1;
    cudaMalloc(&dd0, nout * sizeof(void*)); cudaMemcpy(dd0, hd0.data(), nout * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMalloc(&dd1, nout * sizeof(void*)); cudaMemcpy(dd1, hd1.data(), nout * sizeof(void*), cudaMemcpyHostToDevice);

    double tLin = 0, tLinDot = 0;
    for (int it = 0; it < iters; ++it) {
        cudaDeviceSynchronize();
        const double t0 = now_ms();
        RNSPoly cp(cc, L);
        cp.copy(c);
        cp.INTT(cc.batch, true);                                   // c' (coefficient form)
        cudaDeviceSynchronize();
        liftT_<<<dim3{(uint32_t)N / 128, (uint32_t)nd}, 128>>>(cp.GPU.at(0).limbptr.data, db, dt);
        cudaDeviceSynchronize();
        for (int i = 0; i < nd; ++i) b[i]->NTT(cc.batch, false);  // nd x 3 NTTs
        cudaDeviceSynchronize();
        const double t1 = now_ms();
        dotT_<<<dim3{(uint32_t)N / 128, (uint32_t)(nout * 3)}, 128>>>(db, v, nd, nout, da0, da1, dt, (size_t)N);
        cudaDeviceSynchronize();
        const double t2 = now_ms();
        for (int j = 0; j < nout; ++j) { acc0[j]->INTT(cc.batch, false); acc1[j]->INTT(cc.batch, false); }
        cudaDeviceSynchronize();
        for (int j = 0; j < nout; ++j) {
            crtOut_<<<N / 128, 128>>>(da0 + j * 3, j, dd0, hq[j], tab);
            crtOut_<<<N / 128, 128>>>(da1 + j * 3, j, dd1, hq[j], tab);
        }
        cudaDeviceSynchronize();
        if (it == iters - 1) break;  // keep d0/d1 in coefficient form for the gate on the last iteration
        d0.NTT(cc.batch, false); d0.GPU.at(0).nttSpecialLimbs();
        d1.NTT(cc.batch, false); d1.GPU.at(0).nttSpecialLimbs();
        d0.SetModUp(true); d1.SetModUp(true);
        d0.moddown(true, false, 0);
        d1.moddown(true, false, 1);
        cudaDeviceSynchronize();
        d0.generateSpecialLimbs(false, false); d1.generateSpecialLimbs(false, false);
        cudaDeviceSynchronize();
        const double t3 = now_ms();
        if (it > 0) { tLin += t3 - t0; tLinDot += t2 - t1; }
    }
    os << "[links] linear path: " << tLin / (iters - 2) << " ms/switch (dot " << tLinDot / (iters - 2) << " ms), c0+c1\n";

    // ---------------- baseline (paper's Algorithm 1 at alpha=1): per digit, lift to all of Q, NTT, multiply, accumulate
    RNSPoly accQ(cc, L), Ri(cc, L), tmpQ(cc, L);
    double tBase = 0;
    for (int it = 0; it < 3; ++it) {
        cudaDeviceSynchronize();
        const double t0 = now_ms();
        RNSPoly cp(cc, L);
        cp.copy(c);
        cp.INTT(cc.batch, true);
        cudaDeviceSynchronize();
        bool first = true;
        for (int i = 0; i < nd; ++i) {
            auto& PR = Ri.GPU.at(0);
            joinLimbs(PR);
            liftQ_<<<dim3{(uint32_t)N / 128, (uint32_t)nq}, 128, 0, PR.s.ptr()>>>(cp.GPU.at(0).limbptr.data, i, PR.limbptr.data, dq);
            forkLimbs(PR);
            Ri.NTT(cc.batch, false);
            tmpQ.multElement(Ri, *kap[i]);
            if (first) { accQ.copy(tmpQ); first = false; } else accQ.add(tmpQ);
        }
        accQ.INTT(cc.batch, true);
        cudaDeviceSynchronize();
        const double t1 = now_ms();
        if (it > 0) tBase += t1 - t0;
    }
    os << "[links] baseline (single-limb digits, O(l^2) NTTs, c0 half only, Q limbs only): " << tBase / 2 << " ms\n";
    // gate: d0 (coefficient form, Q limbs) == accQ
    long bad = 0;
    {
        std::vector<std::vector<uint64_t>> A, B;
        d0.store(A); accQ.store(B);
        cudaDeviceSynchronize();
        for (int j = 0; j < nq; ++j) for (int n = 0; n < N; ++n) bad += (A[j][n] != B[j][n]);
        os << "[links] gate linear vs baseline (Q limbs): mismatching residues = " << bad << "\n";
    }
    // ---------------- hybrid (FIDESlib) on a ciphertext at the top level
    {
        Ciphertext ct(any.cc_);
        ct.copy(any);
        ct.c0.grow(L); ct.c1.grow(L);
        for (RNSPoly* p : {&ct.c0, &ct.c1}) { auto& P = p->GPU.at(0); joinLimbs(P); fillRandom_<<<dim3{(uint32_t)N / 128, (uint32_t)nq}, 128, 0, P.s.ptr()>>>(P.limbptr.data, dq, 5u); forkLimbs(P); }
        cudaDeviceSynchronize();
        const KeySwitchingKey& ksk = cc.GetEvalKey(any.keyID);
        double tH = 0;
        for (int it = 0; it < iters; ++it) {
            Ciphertext w(any.cc_); w.copy(ct);
            cudaDeviceSynchronize();
            const double t0 = now_ms();
            w.keySwitch(ksk);
            cudaDeviceSynchronize();
            if (it > 0) tH += now_ms() - t0;
        }
        os << "[links] FIDESlib hybrid key switch (dnum " << 6 << ", level " << L << "): " << tH / (iters - 1) << " ms\n";
    }
    cudaFree(v); cudaFree(dq); cudaFree(db); cudaFree(da0); cudaFree(da1); cudaFree(dd0); cudaFree(dd1);
    return bad;
}
}  // namespace FIDESlib::CKKS
