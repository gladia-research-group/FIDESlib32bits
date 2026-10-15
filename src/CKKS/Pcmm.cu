#include "CKKS/Pcmm.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
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

constexpr int N = 1 << 16, N1 = 1 << 12, S = 16, L = 2, B = 3;

// MLWE shape for t lanes: piece degree np = 2t, rank r = 4096 / np, nch = 16 r
struct Lay {
    int np, r, nch;
};

#define PCMM_CUDA(x)                                                                                          \
    do {                                                                                                      \
        cudaError_t e_ = (x);                                                                                 \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("[pcmm] ") + #x + ": " + cudaGetErrorString(e_)); \
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
    uint32_t p[B];          // q0, q1, P
};

struct Tables {
    const uint32_t* t[B];   // per prime: psi^brev or psi^-brev, length n
    uint32_t ninv[B];
};

// Cooley-Tukey stage (natural -> bit-reversed); poly k uses prime (k / inner) % nprime
__global__ void nttStage(uint32_t* d, int n, int m, int t, Tables tb, Primes pr, int inner, int nprime) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n / 2) return;
    const int poly = blockIdx.y + gridDim.y * blockIdx.z;
    const int pi = (poly / inner) % nprime;
    const uint32_t p = pr.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = d + (size_t)poly * n;
    const uint32_t w = tb.t[pi][m + i];
    const uint32_t u = a[j], v = mulm(a[j + t], w, p);
    a[j] = addm(u, v, p);
    a[j + t] = subm(u, v, p);
}

// Gentleman-Sande stage (bit-reversed -> natural)
__global__ void inttStage(uint32_t* d, int n, int m, int t, Tables tb, Primes pr, int inner, int nprime) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n / 2) return;
    const int poly = blockIdx.y + gridDim.y * blockIdx.z;
    const int pi = (poly / inner) % nprime;
    const uint32_t p = pr.p[pi];
    const int i = b / t, j = 2 * i * t + (b % t);
    uint32_t* a = d + (size_t)poly * n;
    const uint32_t w = tb.t[pi][m + i];
    const uint32_t u = a[j], v = a[j + t];
    a[j] = addm(u, v, p);
    a[j + t] = mulm(subm(u, v, p), w, p);
}

__global__ void scaleBy(uint32_t* d, int n, Tables tb, Primes pr, int inner, int nprime) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int poly = blockIdx.y + gridDim.y * blockIdx.z;
    const int pi = (poly / inner) % nprime;
    uint32_t* a = d + (size_t)poly * n;
    a[k] = mulm(a[k], tb.ninv[pi], pr.p[pi]);
}

dim3 polyGrid(int n, int count, int& threads) {
    threads = 256;
    const int gx = (n / 2 + threads - 1) / threads;
    const int gy = count > 65535 ? 65535 : count;
    const int gz = (count + gy - 1) / gy;
    if (gy * gz != count) throw std::runtime_error("[pcmm] poly count not tileable");
    return dim3(gx, gy, gz);
}

// whole transform of one poly (n <= 4096) in shared memory
constexpr int kSmemN = 4096;
__global__ void nttSmem(uint32_t* d, int n, Tables tb, Primes pr, int inner, int nprime, bool inverse) {
    __shared__ uint32_t a[kSmemN];
    const int poly = blockIdx.x;
    const int pi = (poly / inner) % nprime;
    const uint32_t p = pr.p[pi];
    uint32_t* g = d + (size_t)poly * n;
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
    const uint32_t sc = inverse ? tb.ninv[pi] : 1u;
    for (int k = threadIdx.x; k < n; k += blockDim.x)
        g[k] = inverse ? mulm(a[k], sc, p) : a[k];
}

void ntt(uint32_t* d, int n, int count, const Tables& tb, const Primes& pr, int inner, int nprime, cudaStream_t s) {
    if (n <= kSmemN) {
        nttSmem<<<count, 512, 0, s>>>(d, n, tb, pr, inner, nprime, false);
        return;
    }
    int th;
    dim3 g = polyGrid(n, count, th);
    for (int m = 1, t = n / 2; m < n; m *= 2, t /= 2)
        nttStage<<<g, th, 0, s>>>(d, n, m, t, tb, pr, inner, nprime);
}

void intt(uint32_t* d, int n, int count, const Tables& tb, const Primes& pr, int inner, int nprime, cudaStream_t s) {
    if (n <= kSmemN) {
        nttSmem<<<count, 512, 0, s>>>(d, n, tb, pr, inner, nprime, true);
        return;
    }
    int th;
    dim3 g = polyGrid(n, count, th);
    for (int m = n / 2, t = 1; m >= 1; m /= 2, t *= 2)
        inttStage<<<g, th, 0, s>>>(d, n, m, t, tb, pr, inner, nprime);
    dim3 g2((n + 255) / 256, g.y, g.z);
    scaleBy<<<g2, 256, 0, s>>>(d, n, tb, pr, inner, nprime);
}

uint64_t modinv(uint64_t a, uint64_t m) {
    int64_t t = 0, nt = 1, r = (int64_t)m, nr = (int64_t)(a % m);
    while (nr) {
        const int64_t q = r / nr;
        int64_t tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    if (r != 1) throw std::runtime_error("[pcmm] not invertible");
    return (uint64_t)(t < 0 ? t + (int64_t)m : t);
}

// RNS helpers (coefficient domain)

// exact rescale by the last prime
__global__ void rescaleTop(uint32_t* x, int last, Primes pr, uint32_t pl, Primes inv) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const uint32_t xl = x[(size_t)last * N + k];
    const int64_t c = xl > pl / 2 ? (int64_t)xl - pl : (int64_t)xl;
    for (int i = 0; i < last; ++i) {
        const uint32_t p = pr.p[i];
        const uint32_t cm = (uint32_t)(((c % (int64_t)p) + p) % p);
        x[(size_t)i * N + k] = mulm(subm(x[(size_t)i * N + k], cm, p), inv.p[i], p);
    }
}

// D[j][prime][poly][n] = centered(x[j] * hat_j) mod prime
__global__ void ksDigits(const uint32_t* x, uint32_t* D, int n, int polys, Primes pr, uint32_t hat0, uint32_t hat1) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int poly = blockIdx.y;
    if (k >= n) return;
    for (int j = 0; j < L; ++j) {
        const uint32_t q = pr.p[j];
        const uint32_t v = mulm(x[((size_t)j * polys + poly) * n + k], j ? hat1 : hat0, q);
        const int64_t c = v > q / 2 ? (int64_t)v - q : (int64_t)v;
        for (int pi = 0; pi < B; ++pi) {
            const uint32_t p = pr.p[pi];
            D[(((size_t)j * B + pi) * polys + poly) * n + k] = (uint32_t)(((c % (int64_t)p) + p) % p);
        }
    }
}

// acc[ab][prime][out][n] = sum_j sum_l D[j][prime][out * nl + l][n] * key[l][j][prime][ab][n]   (NTT form)
__global__ void ksMac(const uint32_t* D, const uint32_t* key, uint32_t* acc, int n, int outs, int nl, Primes pr) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y, pi = blockIdx.z;
    if (k >= n) return;
    const uint32_t p = pr.p[pi];
    const int polys = outs * nl;
    unsigned long long a = 0, b = 0;
    for (int j = 0; j < L; ++j)
        for (int l = 0; l < nl; ++l) {
            const unsigned long long d = D[(((size_t)j * B + pi) * polys + (size_t)o * nl + l) * n + k];
            const size_t kb = ((((size_t)l * L + j) * B + pi) * 2) * n + k;
            a = (a + d * key[kb]) % p;
            b = (b + d * key[kb + n]) % p;
        }
    acc[(((size_t)0 * B + pi) * outs + o) * n + k] = (uint32_t)a;
    acc[(((size_t)1 * B + pi) * outs + o) * n + k] = (uint32_t)b;
}

// ModDown by P
__global__ void modDown(const uint32_t* acc, uint32_t* out, int n, int outs, Primes pr, Primes pinv) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y, ab = blockIdx.z;
    if (k >= n) return;
    const uint32_t P = pr.p[2];
    const uint32_t xp = acc[(((size_t)ab * B + 2) * outs + o) * n + k];
    const int64_t c = xp > P / 2 ? (int64_t)xp - P : (int64_t)xp;
    for (int i = 0; i < L; ++i) {
        const uint32_t q = pr.p[i];
        const uint32_t cm = (uint32_t)(((c % (int64_t)q) + q) % q);
        const uint32_t v = mulm(subm(acc[(((size_t)ab * B + i) * outs + o) * n + k], cm, q), pinv.p[i], q);
        out[(((size_t)ab * L + i) * outs + o) * n + k] = v;
    }
}

__global__ void addInto(uint32_t* x, const uint32_t* y, size_t count, int per, Primes pr) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= count) return;
    const uint32_t p = pr.p[k / per];
    x[k] = addm(x[k], y[k], p);
}

// MLWE rows
// A_rows[i_prime][row = sub*R + i][col = l*NP + w]: cpa[sub][c][w] = ka[(w*R + c)*16 + sub];
//   l <= i: cpa[sub][i-l][w];  l > i: zc[sub][i-l+R][w], zc[c][w] = (w == 0) ? -cpa[c][NP-1] : cpa[c][w-1]
__global__ void mlweA(const uint32_t* ka, uint32_t* A, Primes pr, int rowsTotal, int rowOff, Lay ly) {
    const int R = ly.r, NP = ly.np;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;      // < 4096
    const int row = blockIdx.y, pi = blockIdx.z;                // row < NCH
    if (col >= N1) return;
    const uint32_t p = pr.p[pi];
    const int sub = row / R, i = row % R, l = col / NP, w = col % NP;
    const uint32_t* a = ka + (size_t)pi * N;
    uint32_t v;
    if (l <= i) {
        v = a[(w * R + (i - l)) * S + sub];
    } else {
        const int c = i - l + R;
        v = w == 0 ? subm(0, a[((NP - 1) * R + c) * S + sub], p) : a[((w - 1) * R + c) * S + sub];
    }
    A[((size_t)pi * rowsTotal + rowOff + row) * N1 + col] = v;
}

// B_rows[i_prime][row = sub*R + i][w] = b[(w*R + i)*16 + sub]
__global__ void mlweB(const uint32_t* b, uint32_t* Bm, int rowsTotal, int rowOff, Lay ly) {
    const int R = ly.r, NP = ly.np;
    const int w = threadIdx.x, row = blockIdx.x, pi = blockIdx.y;
    const int sub = row / R, i = row % R;
    Bm[((size_t)pi * rowsTotal + rowOff + row) * NP + w] = b[(size_t)pi * N + (w * R + i) * S + sub];
}

// exact GEMM mod p: balanced base-256 digits, int8 tensor cores
// d[t][row][prime * cols + col] = balanced digit t of centered X[prime][row][col]
__global__ void digits4both(const uint32_t* X, int8_t* d, int rows, int cols, Primes pr) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t c2 = 2 * (size_t)cols, count = (size_t)rows * c2;
    if (k >= count) return;
    const size_t row = k / c2, cc = k % c2;
    const int pi = (int)(cc / cols);
    const uint32_t p = pr.p[pi];
    int64_t v = X[((size_t)pi * rows + row) * cols + cc % cols];
    if (v > p / 2) v -= p;                      // centered: |v| < 2^30
    for (int t = 0; t < 4; ++t) {
        int64_t r = ((v % 256) + 256) % 256;    // balanced digit in [-128, 127]
        if (r > 127) r -= 256;
        d[(size_t)t * count + k] = (int8_t)r;
        v = (v - r) / 256;
    }
}

// Y = sum_e G_e 256^e mod p (Horner)
__global__ void recombineE(const int32_t* G, uint32_t* Y, int rc, int rowOff, int rowsOut, int cols, Primes pr,
                           int E) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t c2 = 2 * (size_t)cols, count = (size_t)rc * c2;
    if (k >= count) return;
    const size_t r = k / c2, cc = k % c2;
    const int pi = (int)(cc / cols);
    const int64_t p = pr.p[pi];
    int64_t v = 0;
    for (int e = E - 1; e >= 0; --e)
        v = (v * 256 + G[(size_t)e * count + k]) % p;
    if (v < 0) v += p;
    Y[((size_t)pi * rowsOut + rowOff + r) * cols + cc % cols] = (uint32_t)v;
}

__global__ void digits2(const int16_t* w, int8_t* d, size_t count) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= count) return;
    int v = w[k];
    int r = ((v % 256) + 256) % 256;
    if (r > 127) r -= 256;
    d[k] = (int8_t)r;
    d[count + k] = (int8_t)((v - r) / 256);
}


// pack / compose
// C[i_prime][sub][l][w*R + i] = Ap[i_prime][sub*R + i][l*NP + w]
__global__ void packC(const uint32_t* Ap, uint32_t* C, int rowsTotal, int rowOff, Lay ly) {
    const int R = ly.r, NP = ly.np, NCH = ly.nch;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // < 4096
    const int sl = blockIdx.y, pi = blockIdx.z;                 // sl = sub*R + l
    if (idx >= N1) return;
    const int sub = sl / R, l = sl % R, w = idx / R, i = idx % R;
    C[((size_t)pi * NCH + sl) * N1 + idx] = Ap[((size_t)pi * rowsTotal + rowOff + sub * R + i) * N1 + l * NP + w];
}

// pb[i_prime][sub][w*R + i] += Bp[i_prime][sub*R + i][w]
__global__ void addBpk(uint32_t* pb, const uint32_t* Bp, Primes pr, int rowsTotal, int rowOff, Lay ly) {
    const int R = ly.r, NP = ly.np;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sub = blockIdx.y, pi = blockIdx.z;
    if (idx >= N1) return;
    const int w = idx / R, i = idx % R;
    uint32_t* x = pb + ((size_t)pi * S + sub) * N1 + idx;
    *x = addm(*x, Bp[((size_t)pi * rowsTotal + rowOff + sub * R + i) * NP + w], pr.p[pi]);
}

// RX[i_prime][z*16 + sub] = px[i_prime][sub][z]
__global__ void compose(const uint32_t* px, uint32_t* RX) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int pi = blockIdx.y;
    if (k >= N) return;
    const int z = k / S, sub = k % S;
    RX[(size_t)pi * N + k] = px[((size_t)pi * S + sub) * N1 + z];
}

// limb i < 2 of the output: f[i] * lift[i]; limbs 2, 3: 0
__global__ void liftOut(const uint32_t* f, uint32_t* const* dst, Primes pr, Primes lift) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (k >= N) return;
    dst[i][k] = i < L ? mulm(f[(size_t)i * N + k], lift.p[i], pr.p[i]) : 0u;
}

}  // namespace

struct PcmmKeys {
    Primes pr{};
    Lay ly{};
    uint32_t split[4]{};
    uint32_t* tabs = nullptr;            // psi16 / ipsi16 / psi12 / ipsi12 per prime
    Tables t16{}, it16{}, t12{}, it12{};
    uint32_t* fwd = nullptr;             // [j][prime][ab][N] NTT form (nl = 1)
    uint32_t* back = nullptr;
    uint32_t* pack = nullptr;            // [l][j][prime][ab][N1] NTT form
    cublasHandle_t blas = nullptr;
    ~PcmmKeys() {
        cudaFree(tabs); cudaFree(fwd); cudaFree(back); cudaFree(pack);
        if (blas) cublasDestroy(blas);
    }
};

std::shared_ptr<PcmmKeys> pcmmUploadKeys(const PcmmKeySource& src) {
    auto k = std::make_shared<PcmmKeys>();
    k->pr.p[0] = src.q[0]; k->pr.p[1] = src.q[1]; k->pr.p[2] = src.P;
    for (int i = 0; i < 4; ++i) k->split[i] = src.split[i];
    const size_t per = 2 * (size_t)N + 2 * (size_t)N1;
    PCMM_CUDA(cudaMalloc(&k->tabs, sizeof(uint32_t) * per * B));
    for (int pi = 0; pi < B; ++pi) {
        if (src.psi16[pi].size() != N || src.ipsi16[pi].size() != N || src.psi12[pi].size() != N1 ||
            src.ipsi12[pi].size() != N1)
            throw std::invalid_argument("[pcmm] twiddle tables have the wrong length");
        uint32_t* base = k->tabs + per * pi;
        PCMM_CUDA(cudaMemcpy(base, src.psi16[pi].data(), 4 * N, cudaMemcpyHostToDevice));
        PCMM_CUDA(cudaMemcpy(base + N, src.ipsi16[pi].data(), 4 * N, cudaMemcpyHostToDevice));
        PCMM_CUDA(cudaMemcpy(base + 2 * N, src.psi12[pi].data(), 4 * N1, cudaMemcpyHostToDevice));
        PCMM_CUDA(cudaMemcpy(base + 2 * N + N1, src.ipsi12[pi].data(), 4 * N1, cudaMemcpyHostToDevice));
        k->t16.t[pi] = base; k->it16.t[pi] = base + N; k->t12.t[pi] = base + 2 * N; k->it12.t[pi] = base + 2 * N + N1;
        k->it16.ninv[pi] = src.ninv16[pi]; k->it12.ninv[pi] = src.ninv12[pi];
    }
    auto up = [&](const std::vector<uint32_t>& v, size_t want, uint32_t*& dst, int n, int keys) {
        if (v.size() != want) throw std::invalid_argument("[pcmm] key array has the wrong size");
        PCMM_CUDA(cudaMalloc(&dst, 4 * want));
        PCMM_CUDA(cudaMemcpy(dst, v.data(), 4 * want, cudaMemcpyHostToDevice));
        ntt(dst, n, (int)(want / n), n == N ? k->t16 : k->t12, k->pr, 2, B, 0);
    };
    up(src.fwd, (size_t)L * B * 2 * N, k->fwd, N, 1);
    up(src.back, (size_t)L * B * 2 * N, k->back, N, 1);
    const int R = (int)(src.pack.size() / ((size_t)L * B * 2 * N1));
    if (R < 1 || N1 % R || (size_t)R * L * B * 2 * N1 != src.pack.size())
        throw std::invalid_argument("[pcmm] pack keys: expected [r][2][3][2][4096]");
    k->ly = Lay{N1 / R, R, S * R};
    up(src.pack, (size_t)R * L * B * 2 * N1, k->pack, N1, R);
    PCMM_CUDA(cudaDeviceSynchronize());
    if (cublasCreate(&k->blas) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("[pcmm] cublasCreate");
    return k;
}

namespace {

// sum of nl key switches per output: x [j][outs*nl][n], keys [l][j][prime][ab][n] -> res [ab][j][outs][n]
void keySwitch(const uint32_t* x, uint32_t* res, int n, int outs, int nl, const uint32_t* keys, PcmmKeys& K,
               const Tables& t, const Tables& it, uint32_t* scratch, cudaStream_t s) {
    const int polys = outs * nl;
    const uint32_t Q0 = K.pr.p[0], Q1 = K.pr.p[1], P = K.pr.p[2];
    const uint32_t hat0 = (uint32_t)modinv(Q1 % Q0, Q0), hat1 = (uint32_t)modinv(Q0 % Q1, Q1);
    uint32_t* D = scratch;                                  // [j][prime][polys][n]
    uint32_t* acc = D + (size_t)L * B * polys * n;          // [ab][prime][outs][n]
    ksDigits<<<dim3((n + 255) / 256, polys), 256, 0, s>>>(x, D, n, polys, K.pr, hat0, hat1);
    ntt(D, n, L * B * polys, t, K.pr, polys, B, s);
    ksMac<<<dim3((n + 255) / 256, outs, B), 256, 0, s>>>(D, keys, acc, n, outs, nl, K.pr);
    intt(acc, n, 2 * B * outs, it, K.pr, outs, B, s);
    Primes pinv{};
    pinv.p[0] = (uint32_t)modinv(P % Q0, Q0); pinv.p[1] = (uint32_t)modinv(P % Q1, Q1);
    modDown<<<dim3((n + 255) / 256, outs, 2), 256, 0, s>>>(acc, res, n, outs, K.pr, pinv);
}

// cached device buffers, grown on demand; on OOM trim the runtime pools and retry once
struct Buf {
    void* p = nullptr;
    size_t n = 0;
    template <class T> T* get(size_t count) {
        const size_t want = count * sizeof(T);
        if (want > n) {
            if (p) cudaFree(p);
            p = nullptr;
            n = 0;
            if (cudaMalloc(&p, want) != cudaSuccess) {
                cudaGetLastError();
                int dev = 0;
                cudaGetDevice(&dev);
                GPUpoolTrim(dev);
                cudaMemPool_t pool;
                if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) cudaMemPoolTrimTo(pool, 0);
                cudaDeviceSynchronize();
                p = nullptr;
                PCMM_CUDA(cudaMalloc(&p, want));
            }
            n = want;
        }
        return static_cast<T*>(p);
    }
};
struct Bufs {
    Buf x, scratch, ks, A, Bm, Ap, Bp, C, pk, RA, RB, fb, Xd, Wd, G, Wdev, dst, dstAll;
};
Bufs g_buf;

// Y[pi] = W X[pi] mod p: digit products summed per exponent t + u in int32 (exact), both primes in one GEMM
void modGemm(PcmmKeys& K, const int8_t* Wd, int nW, const uint32_t* X, uint32_t* Y, int rowsIn, int rowsOut, int cols,
             Buf& xdBuf, Buf& gBuf, cudaStream_t s) {
    const int E = 3 + nW;
    if ((int64_t)std::min(4, nW) * (1 << 14) * rowsIn >= ((int64_t)1 << 31))
        throw std::runtime_error("[pcmm] rowsIn too large for exact int32 accumulation");
    cublasSetStream(K.blas, s);
    const int c2 = 2 * cols;
    const size_t cin = (size_t)rowsIn * c2, cw = (size_t)rowsOut * rowsIn;
    int8_t* Xd = xdBuf.get<int8_t>(4 * cin);
    digits4both<<<(cin + 255) / 256, 256, 0, s>>>(X, Xd, rowsIn, cols, K.pr);
    const size_t budget = (size_t)1 << 30;
    int rc = (int)std::min<size_t>(rowsOut, std::max<size_t>(512, budget / ((size_t)E * 4 * (size_t)c2)));
    int32_t* G = gBuf.get<int32_t>((size_t)E * rc * c2);
    const int32_t one = 1, zero = 0;
    for (int r0 = 0; r0 < rowsOut; r0 += rc) {
        const int rcur = std::min(rc, rowsOut - r0);
        bool seen[8] = {false, false, false, false, false, false, false, false};
        for (int t = 0; t < 4; ++t)
            for (int u = 0; u < nW; ++u) {
                const int e = t + u;
                // column-major: G_e^T (c2 x rcur) (+)= X^T (c2 x rowsIn) * W^T[:, r0:r0+rcur] (rowsIn x rcur)
                const cublasStatus_t st = cublasGemmEx(
                    K.blas, CUBLAS_OP_N, CUBLAS_OP_N, c2, rcur, rowsIn, &one, Xd + (size_t)t * cin, CUDA_R_8I, c2,
                    Wd + (size_t)u * cw + (size_t)r0 * rowsIn, CUDA_R_8I, rowsIn, seen[e] ? &one : &zero,
                    G + (size_t)e * rcur * c2, CUDA_R_32I, c2, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
                if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("[pcmm] cublasGemmEx " + std::to_string(st));
                seen[e] = true;
            }
        const size_t cnt = (size_t)rcur * c2;
        recombineE<<<(cnt + 255) / 256, 256, 0, s>>>(G, Y, rcur, r0, rowsOut, cols, K.pr, E);
    }
}

}  // namespace

namespace {
void pcmmCore(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, PcmmKeys& K,
              const int16_t* W16, const int8_t* Wplanes, int nW, double* times) {
    if (nW < 1 || nW > 4) throw std::invalid_argument("[pcmm] 1..4 weight digit planes");
    if (ins.empty() || outs.empty()) throw std::invalid_argument("[pcmm] no ciphertexts");
    ContextData& cc = ins[0]->cc;
    if (cc.N != N) throw std::runtime_error("[pcmm] ring degree must be 2^16");
    for (auto* c : ins) if (c->getLevel() != 3) throw std::runtime_error("[pcmm] the split ciphertexts must hold 4 limbs");
    for (auto* c : outs) if (c->getLevel() != 3) throw std::runtime_error("[pcmm] the outputs must hold 4 limbs");
    for (int i = 0; i < 4; ++i)
        if (cc.prime.at(i).p != K.split[i]) throw std::runtime_error("[pcmm] moduli differ from the keys'");
    const int nIn = (int)ins.size(), nOut = (int)outs.size();
    const int R = K.ly.r, NP = K.ly.np, NCH = K.ly.nch;
    const int rowsIn = NCH * nIn, rowsOut = NCH * nOut;
    cudaStream_t s = 0;
    for (int i = 0; i < 6 && times; ++i) times[i] = 0;
    // FIDESLIB_PCMM_ASYNC=1: no host fences inside (s is the legacy stream, which orders against FIDESlib's blocking
    // streams both ways), no timing marks, outputs lifted straight into their limbs
    static const bool async = [] { const char* e = std::getenv("FIDESLIB_PCMM_ASYNC"); return e && std::atoi(e) > 0; }();
    if (async) times = nullptr;
    cudaDeviceSynchronize();
    auto tick = std::chrono::steady_clock::now();
    auto mark = [&](int slot) {
        if (!times) return;
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        times[slot] += std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
    };
    cudaDeviceSynchronize();
    Bufs& G = g_buf;
    uint32_t* x = G.x.get<uint32_t>(8 * (size_t)N);
    uint32_t* scratch = G.scratch.get<uint32_t>((size_t)L * B * NCH * N1 + 2 * (size_t)B * S * N1);
    uint32_t* ks = G.ks.get<uint32_t>(2 * L * (size_t)S * N1);
    uint32_t* A = G.A.get<uint32_t>((size_t)L * rowsIn * N1);
    uint32_t* Bm = G.Bm.get<uint32_t>((size_t)L * rowsIn * NP);
    uint32_t* Ap = G.Ap.get<uint32_t>((size_t)L * rowsOut * N1);
    uint32_t* Bp = G.Bp.get<uint32_t>((size_t)L * rowsOut * NP);
    uint32_t* C = G.C.get<uint32_t>((size_t)L * NCH * N1);
    uint32_t* pk = G.pk.get<uint32_t>(2 * L * (size_t)S * N1);
    uint32_t* RA = G.RA.get<uint32_t>((size_t)L * N);
    uint32_t* RB = G.RB.get<uint32_t>((size_t)L * N);
    uint32_t* fb = G.fb.get<uint32_t>(2 * (size_t)L * N);
    const size_t nw = (size_t)rowsOut * rowsIn;
    int8_t* Wd = G.Wd.get<int8_t>((size_t)nW * nw);
    uint32_t** ddst = G.dst.get<uint32_t*>(4);
    if (W16) {
        int16_t* Wdev = G.Wdev.get<int16_t>(nw);
        PCMM_CUDA(cudaMemcpy(Wdev, W16, sizeof(int16_t) * nw, cudaMemcpyHostToDevice));
        digits2<<<(nw + 255) / 256, 256, 0, s>>>(Wdev, Wd, nw);
    } else {
        PCMM_CUDA(cudaMemcpy(Wd, Wplanes, (size_t)nW * nw, cudaMemcpyHostToDevice));
    }

    // staging poly kept for the process (never freed: it would outlive its context)
    static RNSPoly* g_stage = nullptr;
    static ContextData* g_stage_cc = nullptr;
    if (g_stage_cc != &cc) {
        g_stage = new RNSPoly(cc, 3);
        g_stage_cc = &cc;
    }
    RNSPoly& m = *g_stage;

    // per input: coefficient form, rescale by q3 then q2, ring switch, MLWE rows
    for (int c = 0; c < nIn; ++c) {
        for (int part = 0; part < 2; ++part) {
            m.copy(part ? ins[c]->c0 : ins[c]->c1);
            m.INTT(1, true);
            if (!async) cudaDeviceSynchronize();
            auto& LP = m.GPU.at(0);
            for (int i = 0; i < 4; ++i) {
                uint32_t* d = x + ((size_t)part * 4 + i) * N;
                const uint32_t* v = std::get<Limb<uint32_t>>(LP.limb[i]).v.data;
                if (async)
                    PCMM_CUDA(cudaMemcpyAsync(d, v, 4 * N, cudaMemcpyDeviceToDevice, s));
                else
                    PCMM_CUDA(cudaMemcpy(d, v, 4 * N, cudaMemcpyDeviceToDevice));
            }
        }
        for (int part = 0; part < 2; ++part) {
            uint32_t* y = x + (size_t)part * 4 * N;
            for (int last = 3; last >= 2; --last) {
                Primes pr{}, inv{};
                for (int i = 0; i < last && i < 3; ++i) {
                    pr.p[i] = K.split[i];
                    inv.p[i] = (uint32_t)modinv(K.split[last] % K.split[i], K.split[i]);
                }
                rescaleTop<<<(N + 255) / 256, 256, 0, s>>>(y, last, pr, K.split[last], inv);
            }
        }
        mark(0);
        uint32_t* a = x;
        uint32_t* b = x + 4 * N;
        keySwitch(a, ks, N, 1, 1, K.fwd, K, K.t16, K.it16, scratch, s);
        addInto<<<(L * N + 255) / 256, 256, 0, s>>>(b, ks + (size_t)L * N, (size_t)L * N, N, K.pr);
        mark(1);
        mlweA<<<dim3(N1 / 256, NCH, L), 256, 0, s>>>(ks, A, K.pr, rowsIn, c * NCH, K.ly);
        mlweB<<<dim3(NCH, L), NP, 0, s>>>(b, Bm, rowsIn, c * NCH, K.ly);
        mark(2);
    }

    // GEMMs mod q
    modGemm(K, Wd, nW, A, Ap, rowsIn, rowsOut, N1, G.Xd, G.G, s);
    modGemm(K, Wd, nW, Bm, Bp, rowsIn, rowsOut, NP, G.Xd, G.G, s);
    mark(3);

    // per output: pack, compose, switch back, lift
    Primes lift{};
    const uint64_t q1 = (uint64_t)K.split[2] * K.split[3];
    lift.p[0] = (uint32_t)(q1 % K.split[0]); lift.p[1] = (uint32_t)(q1 % K.split[1]);
    uint32_t** ddstAll = nullptr;
    if (async) {
        ddstAll = G.dstAll.get<uint32_t*>(8 * (size_t)nOut);
        std::vector<uint32_t*> tab(8 * (size_t)nOut);
        for (int o = 0; o < nOut; ++o)
            for (int part = 0; part < 2; ++part) {
                auto& LP = (part ? outs[o]->c0 : outs[o]->c1).GPU.at(0);
                for (int i = 0; i < 4; ++i) tab[((size_t)o * 2 + part) * 4 + i] = std::get<Limb<uint32_t>>(LP.limb[i]).v.data;
            }
        PCMM_CUDA(cudaMemcpyAsync(ddstAll, tab.data(), sizeof(uint32_t*) * tab.size(), cudaMemcpyHostToDevice, s));
    }
    for (int o = 0; o < nOut; ++o) {
        packC<<<dim3(N1 / 256, NCH, L), 256, 0, s>>>(Ap, C, rowsOut, o * NCH, K.ly);
        keySwitch(C, pk, N1, S, R, K.pack, K, K.t12, K.it12, scratch, s);
        addBpk<<<dim3(N1 / 256, S, L), 256, 0, s>>>(pk + (size_t)L * S * N1, Bp, K.pr, rowsOut, o * NCH, K.ly);
        mark(4);
        compose<<<dim3(N / 256, L), 256, 0, s>>>(pk, RA);
        compose<<<dim3(N / 256, L), 256, 0, s>>>(pk + (size_t)L * S * N1, RB);
        keySwitch(RA, fb, N, 1, 1, K.back, K, K.t16, K.it16, scratch, s);
        addInto<<<(L * N + 255) / 256, 256, 0, s>>>(fb + (size_t)L * N, RB, (size_t)L * N, N, K.pr);
        if (async) {
            for (int part = 0; part < 2; ++part) {
                RNSPoly& P = part ? outs[o]->c0 : outs[o]->c1;
                liftOut<<<dim3(N / 256, 4), 256, 0, s>>>(fb + (size_t)(part ? 1 : 0) * L * N,
                                                          ddstAll + ((size_t)o * 2 + part) * 4, K.pr, lift);
                P.NTT(1, true);
            }
            mark(5);
            continue;
        }
        for (int part = 0; part < 2; ++part) {
            m.copy(part ? outs[o]->c0 : outs[o]->c1);       // the shape; overwritten
            auto& LP = m.GPU.at(0);
            uint32_t* dst[4];
            for (int i = 0; i < 4; ++i) dst[i] = std::get<Limb<uint32_t>>(LP.limb[i]).v.data;
            PCMM_CUDA(cudaMemcpy(ddst, dst, sizeof(dst), cudaMemcpyHostToDevice));
            liftOut<<<dim3(N / 256, 4), 256, 0, s>>>(fb + (size_t)(part ? 1 : 0) * L * N, ddst, K.pr, lift);
            cudaDeviceSynchronize();
            m.NTT(1, true);
            cudaDeviceSynchronize();
            (part ? outs[o]->c0 : outs[o]->c1).copy(m);
        }
        mark(5);
    }
    cudaDeviceSynchronize();
}

}  // namespace

void pcmmMulti(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, PcmmKeys& K,
               const int16_t* W, double* times) {
    pcmmCore(ins, outs, K, W, nullptr, 2, times);
}

void pcmmMultiPlanes(const std::vector<Ciphertext*>& ins, const std::vector<Ciphertext*>& outs, PcmmKeys& K,
                     const int8_t* Wd, int nW, double* times) {
    pcmmCore(ins, outs, K, nullptr, Wd, nW, times);
}

int pcmmChannels(const PcmmKeys& K) {
    return K.ly.nch;
}

void pcmmAtSplit(Ciphertext& ct, PcmmKeys& K, const int16_t* Wp, double* times) {
    std::vector<Ciphertext*> v{&ct};
    pcmmMulti(v, v, K, Wp, times);
}

}  // namespace FIDESlib::CKKS
