// Compact storage of NTT-structured plaintexts (see PtCompact.cuh).
#include "CKKS/PtCompact.cuh"

#include <cstdlib>
#include <iostream>
#include <variant>

#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {

namespace {

template <typename T>
__global__ void ptCompactCheck_(const T* v, int n, uint32_t mask, unsigned int* bad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && v[i] != v[(uint32_t)i & mask])
        atomicAdd(bad, 1u);
}

template <typename T>
__global__ void ptCompactGather_(T* dst, const T* src, int m, int shift) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m)
        dst[j] = shift >= 0 ? src[(size_t)j << shift] : src[j];
}

template <typename T>
__global__ void ptCompactExpand_(T* dst, const T* src, int n, uint32_t enc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[ptCompactIdx(i, enc)];
}

constexpr int kThreads = 256;
int blocksFor(int n) { return (n + kThreads - 1) / kThreads; }

// (compact length, gather shift: -1 = keep the first `m` values) of an n-element limb under a stage mask
std::pair<int, int> compactShape(uint32_t mask, int n) {
    if ((mask & 0x80000000u) != 0) {  // ~(L - 1): constant blocks of L
        const uint32_t L = (~mask) + 1u;
        int shift = 0;
        while ((1u << shift) < L) ++shift;
        return {n >> shift, shift};
    }
    return {(int)mask + 1, -1};       // P - 1: period P
}

template <typename F>
void forEachLimb(RNSPoly& p, F&& f) {
    for (auto& part : p.GPU) {
        for (auto& l : part.limb)
            std::visit([&](auto& x) { f(part, x); }, l);
        for (auto& l : part.SPECIALlimb)
            std::visit([&](auto& x) { f(part, x); }, l);
    }
}

// re-point the partition's device pointer tables at the limbs' current buffers
void repointTables(RNSPoly& p) {
    for (auto& part : p.GPU) {
        auto put = [&](std::vector<LimbImpl>& limbs, VectorGPU<void*>& table) {
            if (limbs.empty() || table.data == nullptr) return;
            std::vector<void*> h(limbs.size());
            for (size_t i = 0; i < limbs.size(); ++i)
                h[i] = std::visit([](auto& x) { return (void*)x.v.data; }, limbs[i]);
            cudaMemcpyAsync(table.data, h.data(), h.size() * sizeof(void*), cudaMemcpyHostToDevice, part.s.ptr());
            cudaStreamSynchronize(part.s.ptr());
        };
        put(part.limb, part.limbptr);
        put(part.SPECIALlimb, part.SPECIALlimbptr);
    }
}

}  // namespace

namespace {
// every limb, the special ones included, reproduces itself through `mask` (ltStageMask sampled three limbs)
bool verifyCompact(Plaintext& pt, uint32_t mask) {
    unsigned int* bad = nullptr;
    cudaMallocManaged(&bad, sizeof(unsigned int));
    *bad = 0;
    forEachLimb(pt.c0, [&](LimbPartition& part, auto& x) {
        const int n = x.v.size;
        ptCompactCheck_<<<blocksFor(n), kThreads, 0, part.s.ptr()>>>(x.v.data, n, mask, bad);
    });
    cudaDeviceSynchronize();
    const bool ok = *bad == 0;
    cudaFree(bad);
    return ok;
}

void doCompact(Plaintext& pt, uint32_t mask) {
    forEachLimb(pt.c0, [&](LimbPartition& part, auto& x) {
        using T = std::remove_reference_t<decltype(*x.v.data)>;
        const auto [m, shift] = compactShape(mask, x.v.size);
        VectorGPU<T> nv(part.s, m, x.v.device, nullptr, m);
        ptCompactGather_<<<blocksFor(m), kThreads, 0, part.s.ptr()>>>(nv.data, x.v.data, m, shift);
        x.replaceStorage(std::move(nv), part.s);
    });
    repointTables(pt.c0);
    pt.compactMask = (mask & 0x80000000u) ? (0x7F000000u | (uint32_t)compactShape(mask, 2).second) : mask;
}
}  // namespace

bool compactPlaintext(Plaintext& pt, uint32_t mask) {
    if (mask == 0xFFFFFFFFu || pt.compactMask != 0xFFFFFFFFu || !verifyCompact(pt, mask)) return false;
    doCompact(pt, mask);
    return true;
}

Plaintext expandCompact(Context& cc, const Plaintext& src) {
    Plaintext& pt = const_cast<Plaintext&>(src);  // read-only use of its limbs and streams
    Plaintext np(cc);
    np.c0.grow(pt.c0.getLevel());
    if (pt.c0.isModUp()) {
        np.c0.generateSpecialLimbs(false, false);
        np.c0.SetModUp(true);
    }
    np.copyMetadata(pt);
    for (size_t g = 0; g < pt.c0.GPU.size(); ++g) {
        auto& sp = pt.c0.GPU[g];
        auto& dp = np.c0.GPU[g];
        auto expand = [&](std::vector<LimbImpl>& s, std::vector<LimbImpl>& d) {
            for (size_t i = 0; i < s.size() && i < d.size(); ++i)
                std::visit(
                    [&](auto& a) {
                        using L = std::remove_reference_t<decltype(a)>;
                        auto& b = std::get<L>(d[i]);
                        const int n = b.v.size;
                        ptCompactExpand_<<<blocksFor(n), kThreads, 0, dp.s.ptr()>>>(b.v.data, a.v.data, n,
                                                                                   pt.compactMask);
                    },
                    s[i]);
        };
        dp.s.wait(sp.s);
        expand(sp.limb, dp.limb);
        expand(sp.SPECIALlimb, dp.SPECIALlimb);
    }
    cudaDeviceSynchronize();
    return np;
}

std::pair<size_t, size_t> compactStages(BootstrapPrecomputation& pre) {
    if (const char* e = std::getenv("FIDESLIB_PT_COMPACT"); e && e[0] == '0') return {0, 0};
    size_t n = 0, freed = 0;
    for (auto* v : {&pre.CtS, &pre.StC})
        for (auto& st : *v) {
            if (st.ptMask == 0xFFFFFFFFu || (st.ptMask >> 24) == 0x7Fu || st.A.empty()) continue;
            // a stage reads every diagonal through one mask: compact all of them or none
            bool all = true;
            for (auto& pt : st.A)
                if (pt.compactMask != 0xFFFFFFFFu || !verifyCompact(pt, st.ptMask)) { all = false; break; }
            if (!all) {
                std::cerr << "[pt_compact] a stage's diagonals do not all verify under its mask: kept full\n";
                continue;
            }
            size_t before = 0, after = 0;
            for (auto& pt : st.A) {
                forEachLimb(pt.c0, [&](LimbPartition&, auto& x) { before += (size_t)x.v.size * sizeof(*x.v.data); });
                doCompact(pt, st.ptMask);
                forEachLimb(pt.c0, [&](LimbPartition&, auto& x) { after += (size_t)x.v.size * sizeof(*x.v.data); });
            }
            st.ptMask = st.A.front().compactMask;
            n += st.A.size();
            freed += before - after;
        }
    for (auto& [drop, var] : pre.raise_variants)
        if (var) {
            auto [a, b] = compactStages(*var);
            n += a;
            freed += b;
        }
    return {n, freed};
}

}  // namespace FIDESlib::CKKS
