#include "CKKS/Ccmm.cuh"

#include <cublas_v2.h>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "CudaUtils.cuh"

namespace FIDESlib::CKKS {

namespace {

constexpr int NB = 1 << 16, NS = 1 << 13, SW = NB / NS, D = 128, KY = NS / D, NQ = 4, NP = 5, NBIG = D / SW;
constexpr size_t COLS = (size_t)2 * NQ * D * NS;

#define CCMM_CUDA(x)                                                                                          \
    do {                                                                                                      \
        cudaError_t e_ = (x);                                                                                 \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("[ccmm] ") + #x + ": " + cudaGetErrorString(e_)); \
    } while (0)

__device__ __forceinline__ uint32_t mulm(uint32_t a, uint32_t b, uint32_t p) {
    return (uint32_t)((uint64_t)a * b % p);
}
__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b, uint32_t p) {
    uint32_t s = a + b;
    return s >= p ? s - p : s;
}
__device__ __forceinline__ uint32_t subm(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a - b : a + p - b;
}

struct Primes {
    uint32_t p[NP];
};
struct Tables {
    const uint32_t* t[NP];
    uint32_t ninv[NP];
};

__global__ void nttStage(uint32_t* d, int n, int m, int t, Tables tb, Primes pr, int inner, int nprime) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n / 2) return;
    const size_t poly = blockIdx.y + (size_t)gridDim.y * blockIdx.z;
    const int pi = (int)((poly / inner) % nprime);
    const uint32_t p = pr.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = d + poly * n;
    const uint32_t w = tb.t[pi][m + i];
    const uint32_t u = a[j], v = mulm(a[j + t], w, p);
    a[j] = addm(u, v, p);
    a[j + t] = subm(u, v, p);
}

__global__ void inttStage(uint32_t* d, int n, int m, int t, Tables tb, Primes pr, int inner, int nprime) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n / 2) return;
    const size_t poly = blockIdx.y + (size_t)gridDim.y * blockIdx.z;
    const int pi = (int)((poly / inner) % nprime);
    const uint32_t p = pr.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = d + poly * n;
    const uint32_t w = tb.t[pi][m + i];
    const uint32_t u = a[j], v = a[j + t];
    a[j] = addm(u, v, p);
    a[j + t] = mulm(subm(u, v, p), w, p);
}

__global__ void scaleBy(uint32_t* d, int n, Tables tb, Primes pr, int inner, int nprime) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const size_t poly = blockIdx.y + (size_t)gridDim.y * blockIdx.z;
    const int pi = (int)((poly / inner) % nprime);
    uint32_t* a = d + poly * n;
    a[k] = mulm(a[k], tb.ninv[pi], pr.p[pi]);
}

dim3 polyGrid(int n, size_t count, int& threads) {
    threads = 256;
    const int gx = (n / 2 + threads - 1) / threads;
    const size_t gy = count > 65535 ? 65535 : count;
    const size_t gz = (count + gy - 1) / gy;
    if (gy * gz != count) throw std::runtime_error("[ccmm] poly count not tileable");
    return dim3(gx, (unsigned)gy, (unsigned)gz);
}

constexpr int kSmemN = 8192;
__global__ void nttSmem(uint32_t* d, int n, size_t count, Tables tb, Primes pr, int inner, int nprime, bool inverse) {
    extern __shared__ uint32_t a[];
    const size_t poly = blockIdx.x + (size_t)gridDim.x * blockIdx.y;
    if (poly >= count) return;
    const int pi = (int)((poly / inner) % nprime);
    const uint32_t p = pr.p[pi];
    uint32_t* g = d + poly * n;
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        a[k] = g[k];
    __syncthreads();
    const uint32_t* w = tb.t[pi];
    if (!inverse) {
        for (int m = 1, t = n / 2; m < n; m *= 2, t /= 2) {
            for (int b = threadIdx.x; b < n / 2; b += blockDim.x) {
                const int i = b / t, j = 2 * i * t + (b % t);
                const uint32_t u = a[j], v = mulm(a[j + t], w[m + i], p);
                a[j] = addm(u, v, p);
                a[j + t] = subm(u, v, p);
            }
            __syncthreads();
        }
    } else {
        for (int m = n / 2, t = 1; m >= 1; m /= 2, t *= 2) {
            for (int b = threadIdx.x; b < n / 2; b += blockDim.x) {
                const int i = b / t, j = 2 * i * t + (b % t);
                const uint32_t u = a[j], v = a[j + t];
                a[j] = addm(u, v, p);
                a[j + t] = mulm(subm(u, v, p), w[m + i], p);
            }
            __syncthreads();
        }
    }
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        g[k] = inverse ? mulm(a[k], tb.ninv[pi], p) : a[k];
}

void nttAny(uint32_t* d, int n, size_t count, const Tables& tb, const Primes& pr, int inner, int nprime, bool inv,
            cudaStream_t s) {
    if (n <= kSmemN) {
        const unsigned gx = (unsigned)std::min<size_t>(count, 65535);
        const unsigned gy = (unsigned)((count + gx - 1) / gx);
        nttSmem<<<dim3(gx, gy), n >= 1024 ? 512 : n / 2, n * sizeof(uint32_t), s>>>(d, n, count, tb, pr, inner, nprime,
                                                                                    inv);
        return;
    }
    int th;
    dim3 g = polyGrid(n, count, th);
    if (!inv) {
        for (int m = 1, t = n / 2; m < n; m *= 2, t /= 2)
            nttStage<<<g, th, 0, s>>>(d, n, m, t, tb, pr, inner, nprime);
    } else {
        for (int m = n / 2, t = 1; m >= 1; m /= 2, t *= 2)
            inttStage<<<g, th, 0, s>>>(d, n, m, t, tb, pr, inner, nprime);
        dim3 g2((n + 255) / 256, g.y, g.z);
        scaleBy<<<g2, 256, 0, s>>>(d, n, tb, pr, inner, nprime);
    }
}

uint64_t modinv(uint64_t a, uint64_t m) {
    int64_t t = 0, nt = 1, r = (int64_t)m, nr = (int64_t)(a % m);
    while (nr) {
        const int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) throw std::runtime_error("[ccmm] not invertible");
    return (uint64_t)(t < 0 ? t + (int64_t)m : t);
}

struct KsConst {
    uint32_t hat[NQ];
    uint32_t pinv[NQ];
};

__global__ void ksDigits(const uint32_t* x, uint32_t* Dg, int n, int polys, Primes pr, KsConst kc) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int poly = blockIdx.y;
    if (k >= n) return;
    for (int j = 0; j < NQ; ++j) {
        const uint32_t q = pr.p[j];
        const uint32_t v = mulm(x[((size_t)j * polys + poly) * n + k], kc.hat[j], q);
        const int64_t c = v > q / 2 ? (int64_t)v - q : (int64_t)v;
        for (int pi = 0; pi < NP; ++pi) {
            const int64_t p = pr.p[pi];
            Dg[(((size_t)j * NP + pi) * polys + poly) * n + k] = (uint32_t)(((c % p) + p) % p);
        }
    }
}

__global__ void ksMac(const uint32_t* Dg, const uint32_t* keys, size_t kstride, uint32_t* acc, int n, int polys,
                      Primes pr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int poly = blockIdx.y, pi = blockIdx.z;
    if (k >= n) return;
    const uint64_t p = pr.p[pi];
    const uint32_t* key = keys + (size_t)poly * kstride;
    uint64_t a = 0, b = 0;
    for (int j = 0; j < NQ; ++j) {
        const uint64_t d = Dg[(((size_t)j * NP + pi) * polys + poly) * n + k];
        const size_t kb = (((size_t)j * NP + pi) * 2) * n + k;
        a = (a + d * key[kb]) % p;
        b = (b + d * key[kb + n]) % p;
    }
    acc[(((size_t)0 * NP + pi) * polys + poly) * n + k] = (uint32_t)a;
    acc[(((size_t)1 * NP + pi) * polys + poly) * n + k] = (uint32_t)b;
}

__global__ void modDown(const uint32_t* acc, uint32_t* out, int n, int polys, Primes pr, KsConst kc) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int poly = blockIdx.y, ab = blockIdx.z;
    if (k >= n) return;
    const uint32_t P = pr.p[NQ];
    const uint32_t xp = acc[(((size_t)ab * NP + NQ) * polys + poly) * n + k];
    const int64_t c = xp > P / 2 ? (int64_t)xp - P : (int64_t)xp;
    for (int i = 0; i < NQ; ++i) {
        const int64_t q = pr.p[i];
        const uint32_t cm = (uint32_t)(((c % q) + q) % q);
        out[(((size_t)ab * NQ + i) * polys + poly) * n + k] =
            mulm(subm(acc[(((size_t)ab * NP + i) * polys + poly) * n + k], cm, (uint32_t)q), kc.pinv[i], (uint32_t)q);
    }
}

__global__ void ksApply(uint32_t* x, const uint32_t* res, size_t per, Primes pr) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= (size_t)NQ * per) return;
    const int pi = (int)(k / per);
    x[k] = addm(x[k], res[(size_t)NQ * per + k], pr.p[pi]);
    x[(size_t)NQ * per + k] = res[k];
}

__device__ __forceinline__ uint32_t monoAt(const uint32_t* v, int k, int64_t e, uint32_t p) {
    e %= 2 * NS;
    if (e < 0) e += 2 * NS;
    bool neg = false;
    if (e >= NS) { e -= NS; neg = true; }
    int src = k - (int)e;
    if (src < 0) { src += NS; neg = !neg; }
    const uint32_t x = v[src];
    return neg ? subm(0, x, p) : x;
}

__global__ void monoCols(const uint32_t* in, uint32_t* out, int mul, Primes pr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const size_t base = ((size_t)pp * D + col) * NS;
    out[base + k] = monoAt(in + base, k, (int64_t)mul * col, pr.p[pp % NQ]);
}

__global__ void bitrevCols(const uint32_t* in, uint32_t* out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const int r = __brev(col) >> (32 - 7);
    out[((size_t)pp * D + col) * NS + k] = in[((size_t)pp * D + r) * NS + k];
}

// one radix-2 level of TWEAK (CKL Alg. 2)
__global__ void tweakLevel(const uint32_t* in, uint32_t* out, int h, int kk, int sgn, Primes pr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = blockIdx.y, pp = blockIdx.z;
    if (k >= NS) return;
    const uint32_t p = pr.p[pp % NQ];
    const int g = pair / h, j = pair % h;
    const int i0 = g * 2 * h + j, i1 = i0 + h;
    const size_t b0 = ((size_t)pp * D + i0) * NS, b1 = ((size_t)pp * D + i1) * NS;
    const uint32_t u = in[b0 + k];
    const uint32_t v = monoAt(in + b1, k, (int64_t)2 * kk * j * sgn, p);
    out[b0 + k] = addm(u, v, p);
    out[b1 + k] = subm(u, v, p);
}

struct ScrConst {
    int tstar[D];
    uint32_t dinv[NQ];
};

__global__ void scrambleAuto(const uint32_t* in, uint32_t* out, const ScrConst* sc, Primes pr) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y, pp = blockIdx.z;
    if (u >= NS) return;
    const uint32_t p = pr.p[pp % NQ];
    const int h = 2 * KY * t + 1;
    const uint32_t x = mulm(in[((size_t)pp * D + sc->tstar[t]) * NS + u], sc->dinv[pp % NQ], p);
    int64_t pos = ((int64_t)u * h) % (2 * NS);
    bool neg = false;
    if (pos >= NS) { pos -= NS; neg = true; }
    out[((size_t)pp * D + t) * NS + pos] = neg ? subm(0, x, p) : x;
}

__global__ void addCols(uint32_t* x, const uint32_t* y, size_t per, Primes pr) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= (size_t)NQ * per) return;
    x[k] = addm(x[k], y[k], pr.p[k / per]);
}

__global__ void gatherU(const uint32_t* X, uint32_t* E) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= COLS) return;
    const int u = (int)(k % KY);
    const size_t rest = k / KY;
    const int i = (int)(rest % D);
    const size_t pc = rest / D;
    E[k] = X[pc * NS + i + (size_t)D * u];
}

__global__ void digitsLR(const uint32_t* E, int8_t* Xd, int8_t* Yd, int prime, Primes pr, bool left) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t per = (size_t)2 * D * D * KY;
    if (k >= per) return;
    const uint32_t p = pr.p[prime];
    int part, i, t, u;
    if (left) {
        t = (int)(k % D);
        const int r = (int)((k / D) % (2 * D));
        u = (int)(k / ((size_t)D * 2 * D));
        part = r / D; i = r % D;
    } else {
        const int c = (int)(k % (2 * D));
        t = (int)((k / (2 * D)) % D);
        u = (int)(k / ((size_t)2 * D * D));
        part = c / D; i = c % D;
    }
    int64_t v = E[((((size_t)part * NQ + prime) * D + t) * D + i) * KY + u];
    if (v > p / 2) v -= p;
    int8_t* dst = left ? Xd : Yd;
    for (int dg = 0; dg < 4; ++dg) {
        int64_t r = ((v % 256) + 256) % 256;
        if (r > 127) r -= 256;
        dst[(size_t)dg * per + k] = (int8_t)r;
        v = (v - r) / 256;
    }
}

__global__ void recombine(const int32_t* G, uint32_t* Cn, int prime, Primes pr) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t per = (size_t)KY * 2 * D * 2 * D;
    if (k >= per) return;
    const int64_t p = pr.p[prime];
    const int c = (int)(k % (2 * D));
    const int r = (int)((k / (2 * D)) % (2 * D));
    const int u = (int)(k / ((size_t)4 * D * D));
    int64_t v = 0;
    for (int e = 6; e >= 0; --e)
        v = (v * 256 + G[(size_t)e * per + k]) % p;
    if (v < 0) v += p;
    Cn[(((size_t)prime * 2 * D + r) * 2 * D + c) * KY + u] = (uint32_t)v;
}

__global__ void scatterRows(const uint32_t* Cn, uint32_t* X, int rowOff) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= COLS) return;
    const int coef = (int)(k % NS);
    const size_t rest = k / NS;
    const int j = (int)(rest % D);
    const int pp = (int)(rest / D);
    const int part = pp / NQ, prime = pp % NQ;
    const int i = coef % D, u = coef / D;
    X[k] = Cn[(((size_t)prime * 2 * D + rowOff + j) * 2 * D + part * D + i) * KY + u];
}

__global__ void splitBig(const uint32_t* big, uint32_t* X) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= COLS) return;
    const int z = (int)(k % NS);
    const size_t rest = k / NS;
    const int col = (int)(rest % D);
    const int pp = (int)(rest / D);
    const int c = col / SW, j0 = col % SW;
    X[k] = big[((size_t)pp * NBIG + c) * NB + j0 + (size_t)SW * z];
}

__global__ void composeBig(const uint32_t* X, uint32_t* big) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= COLS) return;
    const int idx = (int)(k % NB);
    const size_t rest = k / NB;
    const int c = (int)(rest % NBIG);
    const int pp = (int)(rest / NBIG);
    const int z = idx / SW, j0 = idx % SW;
    big[k] = X[((size_t)pp * D + c * SW + j0) * NS + z];
}

}  // namespace

struct CcmmKeys {
    Primes pr{};
    uint32_t* tabs = nullptr;
    Tables t[3]{}, it[3]{};
    uint32_t *fwd = nullptr, *back = nullptr, *autos = nullptr, *rlk = nullptr;
    ScrConst* scr = nullptr;
    KsConst kc{};
    cublasHandle_t blas = nullptr;
    ~CcmmKeys() {
        cudaFree(tabs); cudaFree(fwd); cudaFree(back); cudaFree(autos); cudaFree(rlk); cudaFree(scr);
        if (blas) cublasDestroy(blas);
    }
};

std::shared_ptr<CcmmKeys> ccmmUploadKeys(const CcmmKeySource& src) {
    auto k = std::make_shared<CcmmKeys>();
    for (int i = 0; i < NP; ++i) k->pr.p[i] = src.p[i];
    const int ns[3] = {NB, NS, KY};
    size_t total = 0;
    for (int s = 0; s < 3; ++s) total += 2 * (size_t)ns[s] * NP;
    CCMM_CUDA(cudaMalloc(&k->tabs, 4 * total));
    size_t off = 0;
    for (int s = 0; s < 3; ++s)
        for (int pi = 0; pi < NP; ++pi) {
            if (src.psi[s][pi].size() != (size_t)ns[s] || src.ipsi[s][pi].size() != (size_t)ns[s])
                throw std::invalid_argument("[ccmm] twiddle tables have the wrong length");
            CCMM_CUDA(cudaMemcpy(k->tabs + off, src.psi[s][pi].data(), 4 * ns[s], cudaMemcpyHostToDevice));
            k->t[s].t[pi] = k->tabs + off;
            off += ns[s];
            CCMM_CUDA(cudaMemcpy(k->tabs + off, src.ipsi[s][pi].data(), 4 * ns[s], cudaMemcpyHostToDevice));
            k->it[s].t[pi] = k->tabs + off;
            off += ns[s];
            k->it[s].ninv[pi] = src.ninv[s][pi];
        }
    auto up = [&](const std::vector<uint32_t>& v, size_t want, uint32_t*& dst, int s) {
        if (v.size() != want) throw std::invalid_argument("[ccmm] key array has the wrong size");
        CCMM_CUDA(cudaMalloc(&dst, 4 * want));
        CCMM_CUDA(cudaMemcpy(dst, v.data(), 4 * want, cudaMemcpyHostToDevice));
        nttAny(dst, ns[s], want / ns[s], k->t[s], k->pr, 2, NP, false, 0);
    };
    up(src.fwd, (size_t)NQ * NP * 2 * NB, k->fwd, 0);
    up(src.back, (size_t)NQ * NP * 2 * NB, k->back, 0);
    up(src.autos, (size_t)D * NQ * NP * 2 * NS, k->autos, 1);
    up(src.rlk, (size_t)NQ * NP * 2 * NS, k->rlk, 1);
    for (int j = 0; j < NQ; ++j) {
        uint64_t prod = 1;
        for (int i = 0; i < NQ; ++i)
            if (i != j) prod = prod * (src.p[i] % src.p[j]) % src.p[j];
        k->kc.hat[j] = (uint32_t)modinv(prod, src.p[j]);
        k->kc.pinv[j] = (uint32_t)modinv(src.p[NQ] % src.p[j], src.p[j]);
    }
    ScrConst sc{};
    for (int t = 0; t < D; ++t) {
        const uint64_t h = 2 * KY * t + 1, hinv = modinv(h, 2 * NS);
        sc.tstar[t] = (int)((hinv - 1) / (2 * KY));
    }
    for (int i = 0; i < NQ; ++i) sc.dinv[i] = (uint32_t)modinv(D, src.p[i]);
    CCMM_CUDA(cudaMalloc(&k->scr, sizeof(ScrConst)));
    CCMM_CUDA(cudaMemcpy(k->scr, &sc, sizeof(ScrConst), cudaMemcpyHostToDevice));
    CCMM_CUDA(cudaDeviceSynchronize());
    if (cublasCreate(&k->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("[ccmm] cublasCreate");
    return k;
}

namespace {

struct Buf {
    void* p = nullptr;
    size_t n = 0;
    template <class T> T* get(size_t count) {
        const size_t want = count * sizeof(T);
        if (want > n) {
            if (p) cudaFree(p);
            p = nullptr;
            CCMM_CUDA(cudaMalloc(&p, want));
            n = want;
        }
        return static_cast<T*>(p);
    }
};
struct Bufs {
    Buf big, scratch, res, c[6], E, Xd, Yd, G, Cn;
};
Bufs g_buf;

inline unsigned blocks(size_t n) { return (unsigned)((n + 255) / 256); }

void keySwitch(CcmmKeys& K, const uint32_t* x, uint32_t* res, int n, int polys, const uint32_t* keys, size_t kstride,
               int tab, cudaStream_t s) {
    uint32_t* Dg = g_buf.scratch.get<uint32_t>((size_t)(NQ * NP + 2 * NP) * polys * n);
    uint32_t* acc = Dg + (size_t)NQ * NP * polys * n;
    ksDigits<<<dim3((n + 255) / 256, polys), 256, 0, s>>>(x, Dg, n, polys, K.pr, K.kc);
    nttAny(Dg, n, (size_t)NQ * NP * polys, K.t[tab], K.pr, polys, NP, false, s);
    ksMac<<<dim3((n + 255) / 256, polys, NP), 256, 0, s>>>(Dg, keys, kstride, acc, n, polys, K.pr);
    nttAny(acc, n, (size_t)2 * NP * polys, K.it[tab], K.pr, polys, NP, true, s);
    modDown<<<dim3((n + 255) / 256, polys, 2), 256, 0, s>>>(acc, res, n, polys, K.pr, K.kc);
}

uint32_t* tweak(uint32_t* in, uint32_t* tmp, int sgn, CcmmKeys& K, cudaStream_t s) {
    const dim3 g(NS / 256, D, 2 * NQ);
    bitrevCols<<<g, 256, 0, s>>>(in, tmp);
    uint32_t *a = tmp, *b = in;
    for (int h = 1; h < D; h *= 2) {
        tweakLevel<<<dim3(NS / 256, D / 2, 2 * NQ), 256, 0, s>>>(a, b, h, KY * D / (2 * h), sgn, K.pr);
        std::swap(a, b);
    }
    return a;
}

// CKL Alg. 3: transpose a column-wise encrypted matrix
void cmt(uint32_t* X, uint32_t* t1, uint32_t* t2, CcmmKeys& K, cudaStream_t s) {
    const dim3 g(NS / 256, D, 2 * NQ);
    const size_t per = (size_t)D * NS;
    monoCols<<<g, 256, 0, s>>>(X, t1, 1, K.pr);
    uint32_t* r = tweak(t1, t2, 1, K, s);
    uint32_t* o = r == t1 ? t2 : t1;
    scrambleAuto<<<g, 256, 0, s>>>(r, o, K.scr, K.pr);
    uint32_t* res = g_buf.res.get<uint32_t>(2 * (size_t)NQ * per);
    keySwitch(K, o + (size_t)NQ * per, res, NS, D, K.autos, (size_t)NQ * NP * 2 * NS, 1, s);
    ksApply<<<blocks(NQ * per), 256, 0, s>>>(o, res, per, K.pr);
    uint32_t* r2 = tweak(o, r, -1, K, s);
    monoCols<<<g, 256, 0, s>>>(r2, X, -1, K.pr);
}

__global__ void moveLimbs(uint32_t* const* ptrs, uint32_t* big, bool toBig) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= NB) return;
    uint32_t* bj = big + (size_t)blockIdx.y * NB + k;
    if (toBig) *bj = ptrs[blockIdx.y][k];
    else ptrs[blockIdx.y][k] = *bj;
}

constexpr int NPOLY = 2 * NBIG;

// limb j = (part * NQ + i) * NBIG + c of the staging layout
// FIDESLIB_CCMM_ASYNC=1: no host fences or timing marks inside the CCMM (s is the legacy stream, which orders against
// FIDESlib's blocking streams both ways, as FIDESLIB_PCMM_ASYNC), the limb pointer tables uploaded without blocking
bool ccmmAsync() {
    static const bool on = [] { const char* e = std::getenv("FIDESLIB_CCMM_ASYNC"); return e && std::atoi(e) > 0; }();
    return on;
}
void fence() {
    if (!ccmmAsync()) cudaDeviceSynchronize();
}

uint32_t** limbTable(const std::vector<RNSPoly*>& polys, int which) {
    static uint32_t** dev[2] = {nullptr, nullptr};
    if (!dev[which]) CCMM_CUDA(cudaMalloc(&dev[which], sizeof(uint32_t*) * NQ * NPOLY));
    std::vector<uint32_t*> h((size_t)NQ * NPOLY);
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < NBIG; ++c) {
            auto& LP = polys[part * NBIG + c]->GPU.at(0);
            for (int i = 0; i < NQ; ++i)
                h[((size_t)part * NQ + i) * NBIG + c] = std::get<Limb<uint32_t>>(LP.limb[i]).v.data;
        }
    if (ccmmAsync())    // pageable source: staged before the call returns; ordered on the legacy stream
        CCMM_CUDA(cudaMemcpyAsync(dev[which], h.data(), sizeof(uint32_t*) * h.size(), cudaMemcpyHostToDevice, 0));
    else
        CCMM_CUDA(cudaMemcpy(dev[which], h.data(), sizeof(uint32_t*) * h.size(), cudaMemcpyHostToDevice));
    return dev[which];
}

void ringIn(const std::vector<Ciphertext*>& ins, uint32_t* X, CcmmKeys& K, cudaStream_t s) {
    ContextData& cc = ins[0]->cc;
    static std::vector<std::unique_ptr<RNSPoly>> stage;
    static ContextData* stage_cc = nullptr;
    if (stage_cc != &cc) {
        stage.clear();
        for (int j = 0; j < NPOLY; ++j) stage.push_back(std::make_unique<RNSPoly>(cc, 3));
        stage_cc = &cc;
    }
    uint32_t* big = g_buf.big.get<uint32_t>((size_t)NBIG * 2 * NQ * NB);
    fence();
    std::vector<RNSPoly*> polys(NPOLY);
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < NBIG; ++c) {
            RNSPoly* st = stage[part * NBIG + c].get();
            st->copy(part ? ins[c]->c1 : ins[c]->c0);
            st->INTT(1, true);
            polys[part * NBIG + c] = st;
        }
    fence();
    moveLimbs<<<dim3(NB / 256, NQ * NPOLY), 256, 0, s>>>(limbTable(polys, 0), big, true);
    const size_t per = (size_t)NBIG * NB;
    uint32_t* res = g_buf.res.get<uint32_t>(2 * (size_t)NQ * per);
    keySwitch(K, big + (size_t)NQ * per, res, NB, NBIG, K.fwd, 0, 0, s);
    ksApply<<<blocks(NQ * per), 256, 0, s>>>(big, res, per, K.pr);
    splitBig<<<blocks(COLS), 256, 0, s>>>(big, X);
}

void ringOut(uint32_t* X, const std::vector<Ciphertext*>& outs, CcmmKeys& K, cudaStream_t s) {
    uint32_t* big = g_buf.big.get<uint32_t>((size_t)NBIG * 2 * NQ * NB);
    composeBig<<<blocks(COLS), 256, 0, s>>>(X, big);
    const size_t per = (size_t)NBIG * NB;
    uint32_t* res = g_buf.res.get<uint32_t>(2 * (size_t)NQ * per);
    keySwitch(K, big + (size_t)NQ * per, res, NB, NBIG, K.back, 0, 0, s);
    ksApply<<<blocks(NQ * per), 256, 0, s>>>(big, res, per, K.pr);
    std::vector<RNSPoly*> polys(NPOLY);
    for (int part = 0; part < 2; ++part)
        for (int c = 0; c < NBIG; ++c) polys[part * NBIG + c] = part ? &outs[c]->c1 : &outs[c]->c0;
    fence();
    moveLimbs<<<dim3(NB / 256, NQ * NPOLY), 256, 0, s>>>(limbTable(polys, 1), big, false);
    fence();
    for (RNSPoly* p : polys) p->NTT(1, true);
    fence();
}

void check(const std::vector<Ciphertext*>& v, CcmmKeys& K, const char* what) {
    if (v.size() != (size_t)NBIG) throw std::invalid_argument(std::string("[ccmm] ") + what + ": need 16 ciphertexts");
    for (auto* c : v) {
        if (c->cc.N != NB) throw std::runtime_error("[ccmm] ring degree must be 2^16");
        if (c->getLevel() != 3) throw std::runtime_error(std::string("[ccmm] ") + what + " must hold 4 limbs");
        for (int i = 0; i < NQ; ++i)
            if (c->cc.prime.at(i).p != K.pr.p[i]) throw std::runtime_error("[ccmm] moduli differ from the keys'");
    }
}

struct Timer {
    double* t;
    std::chrono::steady_clock::time_point tick = std::chrono::steady_clock::now();
    void mark(int slot) {
        if (!t) return;
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        t[slot] += std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
    }
};

void ppmm(uint32_t* XM, uint32_t* XMp, CcmmKeys& K, cudaStream_t s) {
    uint32_t* E = g_buf.E.get<uint32_t>(2 * COLS);
    uint32_t* Em = E;
    uint32_t* Ep = E + COLS;
    gatherU<<<blocks(COLS), 256, 0, s>>>(XM, Em);
    gatherU<<<blocks(COLS), 256, 0, s>>>(XMp, Ep);
    nttAny(E, KY, 2 * COLS / KY, K.t[2], K.pr, D * D, NQ, false, s);
    const size_t per = (size_t)2 * D * D * KY;
    int8_t* Xd = g_buf.Xd.get<int8_t>(4 * per);
    int8_t* Yd = g_buf.Yd.get<int8_t>(4 * per);
    const size_t gper = (size_t)KY * 2 * D * 2 * D;
    int32_t* G = g_buf.G.get<int32_t>(7 * gper);
    uint32_t* Cn = g_buf.Cn.get<uint32_t>((size_t)NQ * gper);
    cublasSetStream(K.blas, s);
    const int32_t one = 1, zero = 0;
    for (int pi = 0; pi < NQ; ++pi) {
        digitsLR<<<blocks(per), 256, 0, s>>>(Em, Xd, Yd, pi, K.pr, true);
        digitsLR<<<blocks(per), 256, 0, s>>>(Ep, Xd, Yd, pi, K.pr, false);
        bool seen[7] = {};
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                const int e = a + b;
                const cublasStatus_t st = cublasGemmStridedBatchedEx(
                    K.blas, CUBLAS_OP_N, CUBLAS_OP_N, 2 * D, 2 * D, D, &one,
                    Yd + (size_t)b * per, CUDA_R_8I, 2 * D, (long long)D * 2 * D,
                    Xd + (size_t)a * per, CUDA_R_8I, D, (long long)2 * D * D,
                    seen[e] ? &one : &zero, G + (size_t)e * gper, CUDA_R_32I, 2 * D, (long long)4 * D * D, KY,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                if (st != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("[ccmm] cublasGemmStridedBatchedEx " + std::to_string(st));
                seen[e] = true;
            }
        recombine<<<blocks(gper), 256, 0, s>>>(G, Cn, pi, K.pr);
    }
    nttAny(Cn, KY, (size_t)NQ * gper / KY, K.it[2], K.pr, 4 * D * D, NQ, true, s);
}

}  // namespace

void ccmmTranspose(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, CcmmKeys& K,
                   double* times) {
    check(ins, K, "ins");
    check(outs, K, "outs");
    cudaStream_t s = 0;
    if (times) for (int i = 0; i < 5; ++i) times[i] = 0;
    if (ccmmAsync()) times = nullptr;
    fence();
    Timer tm{times};
    uint32_t* X = g_buf.c[0].get<uint32_t>(COLS);
    uint32_t* t1 = g_buf.c[1].get<uint32_t>(COLS);
    uint32_t* t2 = g_buf.c[2].get<uint32_t>(COLS);
    ringIn(ins, X, K, s);
    tm.mark(0);
    cmt(X, t1, t2, K, s);
    tm.mark(1);
    ringOut(X, outs, K, s);
    tm.mark(4);
    fence();
}

void ccmmMul(const std::vector<Ciphertext*>& insM, const std::vector<Ciphertext*>& insMp,
             const std::vector<Ciphertext*>& outs, CcmmKeys& K, double* times, bool mpTransposed) {
    check(insM, K, "insM");
    check(insMp, K, "insMp");
    check(outs, K, "outs");
    cudaStream_t s = 0;
    if (times) for (int i = 0; i < 5; ++i) times[i] = 0;
    if (ccmmAsync()) times = nullptr;
    fence();
    Timer tm{times};
    uint32_t* XM = g_buf.c[0].get<uint32_t>(COLS);
    uint32_t* XMp = g_buf.c[1].get<uint32_t>(COLS);
    uint32_t* t1 = g_buf.c[2].get<uint32_t>(COLS);
    uint32_t* t2 = g_buf.c[3].get<uint32_t>(COLS);
    uint32_t* X1 = g_buf.c[4].get<uint32_t>(COLS);
    ringIn(insM, XM, K, s);
    ringIn(insMp, XMp, K, s);
    tm.mark(0);
    if (!mpTransposed)
        cmt(XMp, t1, t2, K, s);
    tm.mark(1);
    ppmm(XM, XMp, K, s);
    tm.mark(2);
    uint32_t* Cn = static_cast<uint32_t*>(g_buf.Cn.p);
    scatterRows<<<blocks(COLS), 256, 0, s>>>(Cn, XM, 0);
    scatterRows<<<blocks(COLS), 256, 0, s>>>(Cn, X1, D);
    cmt(XM, t1, t2, K, s);
    cmt(X1, t1, t2, K, s);
    const size_t per = (size_t)D * NS;
    uint32_t* res = g_buf.res.get<uint32_t>(2 * (size_t)NQ * per);
    keySwitch(K, X1 + (size_t)NQ * per, res, NS, D, K.rlk, 0, 1, s);
    addCols<<<blocks(NQ * per), 256, 0, s>>>(XM, res + (size_t)NQ * per, per, K.pr);
    addCols<<<blocks(NQ * per), 256, 0, s>>>(XM + (size_t)NQ * per, X1, per, K.pr);
    addCols<<<blocks(NQ * per), 256, 0, s>>>(XM + (size_t)NQ * per, res, per, K.pr);
    tm.mark(3);
    ringOut(XM, outs, K, s);
    tm.mark(4);
    fence();
}

}  // namespace FIDESlib::CKKS
