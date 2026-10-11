#include "CKKS/GpuEncrypt.cuh"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KskSeedExpand.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"

namespace FIDESlib::CKKS {

namespace {

constexpr uint32_t kEncDomain = 0x47454e43u;  // "GENC": never collides with the key-expansion lanes
constexpr int kCdt = 48;                       // |x| <= 47 for sigma ~3.2: the tail past 32 is below 2^-64

std::atomic<int> g_on{[] {
    const char* e = std::getenv("FIDESLIB_GPU_ENCRYPT");
    return e ? std::atoi(e) : 1;
}()};

// cdt[k] = 2^64 * P(|x| <= k) for P(x) ~ exp(-x^2 / (2 sigma^2)), saturated; |x| = first k with u < cdt[k].
struct Cdt {
    uint64_t c[kCdt];
};

Cdt makeCdt(double sigma) {
    long double w[kCdt], sum = 0;
    for (int k = 0; k < kCdt; ++k) {
        w[k] = (k == 0 ? 1.0L : 2.0L) * std::exp(-(long double)k * k / (2.0L * sigma * sigma));
        sum += w[k];
    }
    Cdt t{};
    long double acc = 0;
    for (int k = 0; k < kCdt; ++k) {
        acc += w[k];
        const long double v = std::ldexp(acc / sum, 64);
        t.c[k] = v >= std::ldexp(1.0L, 64) ? ~0ull : (uint64_t)v;
    }
    t.c[kCdt - 1] = ~0ull;
    return t;
}

__device__ __forceinline__ int gauss(const Cdt& t, uint64_t u, uint32_t sign) {
    int k = 0;
#pragma unroll 4
    for (; k < kCdt - 1; ++k)
        if (u < t.c[k])
            break;
    return (k != 0 && (sign & 1u)) ? -k : k;
}

template <typename T>
__device__ __forceinline__ T residue(int x, uint64_t q) {
    return (T)(x >= 0 ? (uint64_t)x : q - (uint64_t)(-x));
}

// One coefficient per thread: block i of ChaCha12(key, nonce) gives v (ternary), e0 and e1 (Gaussian); their
// residues go to every limb in coefficient form.
__global__ void sampleSmall_(void** v, void** e0, void** e1, const __grid_constant__ int limbs,
                             const __grid_constant__ Cdt cdt, const __grid_constant__ uint32_t k0,
                             const __grid_constant__ uint32_t k1, const __grid_constant__ uint32_t k2,
                             const __grid_constant__ uint32_t k3, const __grid_constant__ uint32_t k4,
                             const __grid_constant__ uint32_t k5, const __grid_constant__ uint32_t k6,
                             const __grid_constant__ uint32_t k7, const __grid_constant__ uint64_t nonce) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t key[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    uint32_t r[16];
    kskexpand::chacha_block_tail(key, (uint32_t)idx, (uint32_t)nonce, (uint32_t)(nonce >> 32), kEncDomain, r);
    const uint64_t uv = ((uint64_t)r[1] << 32) | r[0];
    const int xv = (int)__umul64hi(uv, 3ull) - 1;  // uniform {-1, 0, 1}, bias 2^-64
    const int x0 = gauss(cdt, ((uint64_t)r[3] << 32) | r[2], r[6]);
    const int x1 = gauss(cdt, ((uint64_t)r[5] << 32) | r[4], r[6] >> 1);
    for (int j = 0; j < limbs; ++j) {
        const uint64_t q = C_.primes[j];
        if (ISU64(j)) {
            ((uint64_t*)v[j])[idx] = residue<uint64_t>(xv, q);
            ((uint64_t*)e0[j])[idx] = residue<uint64_t>(x0, q);
            ((uint64_t*)e1[j])[idx] = residue<uint64_t>(x1, q);
        } else {
            ((uint32_t*)v[j])[idx] = residue<uint32_t>(xv, q);
            ((uint32_t*)e0[j])[idx] = residue<uint32_t>(x0, q);
            ((uint32_t*)e1[j])[idx] = residue<uint32_t>(x1, q);
        }
    }
}

}  // namespace

bool gpuEncryptOn() {
    return g_on.load(std::memory_order_relaxed) > 0;
}

void setGpuEncrypt(int on) {
    g_on.store(on, std::memory_order_relaxed);
}

void encryptZeroDevice(Ciphertext& out, const Ciphertext& pk, const uint32_t key[8], uint64_t nonce, double sigma) {
    ContextData& cc = out.cc;
    assert(cc.GPUid.size() == 1 && "encryptZeroDevice is single-GPU only");
    const int level = out.c0.getLevel();
    assert(pk.c0.getLevel() >= level);
    static thread_local double cdt_sigma = -1;
    static thread_local Cdt cdt;
    if (sigma != cdt_sigma) {
        cdt = makeCdt(sigma);
        cdt_sigma = sigma;
    }

    RNSPoly v(cc, level), e0(cc, level), e1(cc, level);
    auto& pv = v.GPU.at(0);
    pv.s.wait(e0.GPU.at(0).s);
    pv.s.wait(e1.GPU.at(0).s);
    cudaSetDevice(pv.device);
    sampleSmall_<<<cc.N / 128, 128, 0, pv.s.ptr()>>>(pv.limbptr.data, e0.GPU.at(0).limbptr.data,
                                                     e1.GPU.at(0).limbptr.data, level + 1, cdt, key[0], key[1],
                                                     key[2], key[3], key[4], key[5], key[6], key[7], nonce);
    e0.GPU.at(0).s.wait(pv.s);
    e1.GPU.at(0).s.wait(pv.s);
    v.NTT(cc.batch, true);
    e0.NTT(cc.batch, true);
    e1.NTT(cc.batch, true);

    out.c0.multElement(pk.c0, v);
    out.c0.add(e0);
    out.c1.multElement(pk.c1, v);
    out.c1.add(e1);
}

}  // namespace FIDESlib::CKKS
