// CKKS packed encoding on the GPU (see DeviceEncode.cuh).
#include <cmath>
#include <cstring>
#include <complex>
#include <map>
#include <mutex>
#include "CKKS/Context.cuh"
#include "CKKS/DeviceEncode.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {

namespace {

// slot j sits at zeta^{5^j}, its conjugate at zeta^{-5^j}: FFT inputs (t-1)/2 and (2N-t-1)/2 (Spru.cu)
__global__ void place_(const double2* v, const int* uPos, const int* uNeg, double2* w, int half) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    const double2 x = v[j];
    w[uPos[j]] = x;
    w[uNeg[j]] = make_double2(x.x, -x.y);
}

// one radix-2 Stockham pass (forward, exp(-2 pi i ...)), as spruFftPass_
__global__ void fftPass_(const double2* in, double2* out, int N, int Ns) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = N / 2;
    if (j >= half) return;
    const double2 v0 = in[j], v1 = in[j + half];
    const int k = j % Ns;
    double sn, cs;
    sincospi(-(double)k / (double)Ns, &sn, &cs);
    const double2 t = make_double2(v1.x * cs - v1.y * sn, v1.x * sn + v1.y * cs);
    const int idx = (j / Ns) * Ns * 2 + k;
    out[idx] = make_double2(v0.x + t.x, v0.y + t.y);
    out[idx + Ns] = make_double2(v0.x - t.x, v0.y - t.y);
}

// c_k = Re(Y_k zeta^{-k} / N) * scale, and max |c_k| (non-negative doubles order as their bit patterns)
__global__ void coeffs_(const double2* Y, int N, double scale, double* c, unsigned long long* cmax) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 0.0;
    if (k < N) {
        double sn, cs;
        sincospi(-(double)k / (double)N, &sn, &cs);
        const double re = (Y[k].x * cs - Y[k].y * sn) / (double)N * scale;
        c[k] = re;
        a = fabs(re);
    }
    for (int o = 16; o > 0; o >>= 1) a = fmax(a, __shfl_down_sync(0xffffffff, a, o));
    if ((threadIdx.x & 31) == 0) atomicMax(cmax, (unsigned long long)__double_as_longlong(a));
}

// OpenFHE's split: logc = max ceil(log2|c_k|), exact through frexp (|c| = f 2^e, f in [0.5, 1)); coefficients past
// 2^61 are rounded at c / 2^(logc - 61) and scaled back by that power of two mod each prime
__device__ int splitOf(unsigned long long cmaxBits) {
    const double m = __longlong_as_double((long long)cmaxBits);
    if (!(m > 0.0)) return 0;
    int e;
    const double f = frexp(m, &e);
    const int logc = (f == 0.5) ? e - 1 : e;
    return logc > 61 ? logc - 61 : 0;
}

// m_k = llround(c_k / 2^split), reduced into every limb, times deg_factor[l] * 2^split mod p (coefficient form)
__global__ void scatter_(const double* c, int N, const unsigned long long* cmax, void** limbs, const uint32_t* primes,
                         const uint32_t* degFactor, int nl) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const int split = splitOf(*cmax);
    const long long m = llround(ldexp(c[k], -split));
    for (int l = 0; l < nl; ++l) {
        const uint64_t p = primes[l];
        uint64_t f = degFactor[l];
        for (int i = 0; i < split; ++i) f = (f << 1) % p;
        long long r = m % (long long)p;
        if (r < 0) r += p;
        ((uint32_t*)limbs[l])[k] = (uint32_t)((uint64_t)r * f % p);
    }
}

// row r of the staged raw rows -> double2 slot values (zero past `width`)
__global__ void widen_(const void* raw, int dtype, size_t width, size_t row, double2* v, int half) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    double2 x = make_double2(0.0, 0.0);
    if ((size_t)j < width) {
        const size_t i = row * width + j;
        switch (dtype) {
            case 0: x.x = ((const float*)raw)[i]; break;
            case 1: x.x = ((const double*)raw)[i]; break;
            case 2: x.x = ((const float*)raw)[2 * i]; x.y = ((const float*)raw)[2 * i + 1]; break;
            default: x.x = ((const double*)raw)[2 * i]; x.y = ((const double*)raw)[2 * i + 1]; break;
        }
    }
    v[j] = x;
}

uint32_t powmod(uint64_t b, uint64_t e, uint64_t p) {
    uint64_t r = 1 % p;
    b %= p;
    while (e) {
        if (e & 1) r = r * b % p;
        b = b * b % p;
        e >>= 1;
    }
    return (uint32_t)r;
}

struct Scratch {
    void* pinned = nullptr;           // the raw rows' staging, grown on demand
    size_t pinnedBytes = 0;
    void* raw = nullptr;              // their device copy
    size_t rawBytes = 0;
    int N = 0;
    int* uPos = nullptr;
    int* uNeg = nullptr;
    double2* v = nullptr;
    double2* a = nullptr;
    double2* b = nullptr;
    double* c = nullptr;
    unsigned long long* cmax = nullptr;
    uint32_t* primes = nullptr;
    std::map<int, uint32_t*> degFactor;   // round(scale)^(deg-1) mod p per prime, per (deg, scale)
    cudaStream_t stream = nullptr;        // every encode runs here: the scratch is reused in order
    cudaEvent_t limbsReady = nullptr, encoded = nullptr;
};

std::mutex g_mutex;
Scratch g_s;

void ensure(Scratch& s, ContextData& C) {
    const int N = C.N;
    if (s.N == N) return;
    std::vector<int> up(N / 2), un(N / 2);
    uint64_t t = 1;
    for (int j = 0; j < N / 2; ++j) {
        up[j] = (int)((t - 1) / 2);
        un[j] = (int)((2ull * N - t - 1) / 2);
        t = (t * 5) % (2ull * N);
    }
    std::vector<uint32_t> pr(C.L + 1);
    for (int l = 0; l <= C.L; ++l) pr[l] = (uint32_t)C.prime[l].p;
    cudaMalloc(&s.uPos, sizeof(int) * N / 2);
    cudaMalloc(&s.uNeg, sizeof(int) * N / 2);
    cudaMalloc(&s.v, sizeof(double2) * N / 2);
    cudaMalloc(&s.a, sizeof(double2) * N);
    cudaMalloc(&s.b, sizeof(double2) * N);
    cudaMalloc(&s.c, sizeof(double) * N);
    cudaMalloc(&s.cmax, sizeof(unsigned long long));
    cudaMalloc(&s.primes, sizeof(uint32_t) * pr.size());
    cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&s.limbsReady, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&s.encoded, cudaEventDisableTiming);
    cudaMemcpy(s.uPos, up.data(), sizeof(int) * N / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(s.uNeg, un.data(), sizeof(int) * N / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(s.primes, pr.data(), sizeof(uint32_t) * pr.size(), cudaMemcpyHostToDevice);
    s.N = N;
}

}  // namespace

namespace {
// the encode of whatever sits in s.v (on stream E): the FFT, the coefficients, the limb scatter and the NTT
cudaStream_t encodeFromScratch(Scratch& s, Plaintext& pt, int limbs, double scale, int deg, double* cmaxPinned = nullptr) {
    ContextData& C = pt.cc;
    const int N = C.N;
    const cudaStream_t E = s.stream;
    // round(scale)^(deg-1) mod each prime, cached per (deg, scale bits): deg 1 is all ones
    const uint64_t powP = (uint64_t)std::llround(scale);
    const int key = deg * 1000 + (int)std::log2(scale);
    auto it = s.degFactor.find(key);
    if (it == s.degFactor.end()) {
        std::vector<uint32_t> f(C.L + 1);
        for (int l = 0; l <= C.L; ++l) f[l] = powmod(powP, deg - 1, C.prime[l].p);
        uint32_t* d = nullptr;
        cudaMalloc(&d, sizeof(uint32_t) * f.size());
        cudaMemcpy(d, f.data(), sizeof(uint32_t) * f.size(), cudaMemcpyHostToDevice);
        it = s.degFactor.emplace(key, d).first;
    }
    place_<<<(N / 2 + 255) / 256, 256, 0, E>>>(s.v, s.uPos, s.uNeg, s.a, N / 2);
    double2* in = s.a;
    double2* out = s.b;
    for (int Ns = 1; Ns < N; Ns <<= 1) {
        fftPass_<<<(N / 2 + 255) / 256, 256, 0, E>>>(in, out, N, Ns);
        std::swap(in, out);
    }
    cudaMemsetAsync(s.cmax, 0, sizeof(unsigned long long), E);
    coeffs_<<<(N + 255) / 256, 256, 0, E>>>(in, N, scale, s.c, s.cmax);
    // the max |coefficient| read back before `encoded` is recorded: a host wait on the plaintext's stream covers it
    if (cmaxPinned)   // the bit pattern of a non-negative double, read back as one
        cudaMemcpyAsync(cmaxPinned, s.cmax, sizeof(double), cudaMemcpyDeviceToHost, E);
    pt.c0.grow(limbs - 1);   // with the NTT's aux buffers (constant limbs have none), as Spru.cu
    const cudaStream_t ps = pt.c0.GPU.at(0).s.ptr();
    cudaEventRecord(s.limbsReady, ps);
    cudaStreamWaitEvent(E, s.limbsReady, 0);
    scatter_<<<(N + 255) / 256, 256, 0, E>>>(s.c, N, s.cmax, pt.c0.GPU.at(0).limbptr.data, s.primes, it->second, limbs);
    cudaEventRecord(s.encoded, E);
    cudaStreamWaitEvent(ps, s.encoded, 0);
    pt.c0.NTT(C.batch, true);
    pt.NoiseFactor = std::pow(scale, deg);
    pt.NoiseLevel = deg;
    pt.slots = N / 2;
    return ps;
}
}  // namespace

cudaStream_t encodeOnDevice(Plaintext& pt, const std::vector<std::complex<double>>& values, int limbs, double scale,
                            int deg, double* cmaxPinned) {
    ContextData& C = pt.cc;
    const int N = C.N;
    std::lock_guard<std::mutex> g(g_mutex);
    Scratch& s = g_s;
    ensure(s, C);
    const cudaStream_t E = s.stream;
    cudaMemsetAsync(s.v, 0, sizeof(double2) * N / 2, E);
    cudaMemcpyAsync(s.v, values.data(), sizeof(double2) * values.size(), cudaMemcpyHostToDevice, E);
    return encodeFromScratch(s, pt, limbs, scale, deg, cmaxPinned);
}

std::vector<cudaStream_t> encodeRowsOnDevice(const std::vector<Plaintext*>& pts, const void* rows, int dtype, size_t n,
                                             size_t width, int limbs, double scale, int deg) {
    std::vector<cudaStream_t> ready;
    if (pts.empty()) return ready;
    ContextData& C = pts[0]->cc;
    const int N = C.N;
    std::lock_guard<std::mutex> g(g_mutex);
    Scratch& s = g_s;
    ensure(s, C);
    const cudaStream_t E = s.stream;
    const size_t elem = (dtype == 0 ? 4 : dtype == 1 || dtype == 2 ? 8 : 16);
    const size_t bytes = n * width * elem;
    // the previous call's upload has to be done before the staging is overwritten
    cudaStreamSynchronize(E);
    if (s.pinnedBytes < bytes) {
        if (s.pinned) cudaFreeHost(s.pinned);
        cudaMallocHost(&s.pinned, bytes);
        s.pinnedBytes = bytes;
    }
    if (s.rawBytes < bytes) {
        if (s.raw) cudaFree(s.raw);
        cudaMalloc(&s.raw, bytes);
        s.rawBytes = bytes;
    }
    std::memcpy(s.pinned, rows, bytes);
    cudaMemcpyAsync(s.raw, s.pinned, bytes, cudaMemcpyHostToDevice, E);
    ready.reserve(n);
    for (size_t r = 0; r < n; ++r) {
        widen_<<<(N / 2 + 255) / 256, 256, 0, E>>>(s.raw, dtype, width, r, s.v, N / 2);
        ready.push_back(encodeFromScratch(s, *pts[r], limbs, scale, deg));
    }
    return ready;
}

}  // namespace FIDESlib::CKKS
