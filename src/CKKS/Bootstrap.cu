#include <set>
#include <cstdio>
//
// Created by carlosad on 4/12/24.
//

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <utility>
#include <vector>
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include <cmath>
#include <iostream>
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapStages.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/AksKeys.cuh"
#include "CKKS/SmallInt.cuh"
#include "CKKS/SparseB.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

// Stage-divergence harness (default off): when a caller installs a stash vector, every
// btsStageProbe checkpoint (pre-CtS / post-CtS / pre-StC / post-StC / end) also deposits a
// full ciphertext clone the caller can download+decrypt offline. Zero cost when null.
std::vector<std::pair<std::string, std::shared_ptr<FIDESlib::CKKS::Ciphertext>>>*
    FIDESlib::CKKS::g_btsStageStash = nullptr;


// Effective correction factor for this bootstrap call: the ContextData override (armed by
// the wrapper's CorrectionScope) wins over the per-slots precomputation value.
// StC-first's folded last stage is built at this correction factor (the planner's cf_max); a call with a smaller one
// scales its input by the exact integer 2^(ref - cf) first.
static constexpr int kStcFirstCfRef = 20;

static uint32_t effCorrectionFactor(FIDESlib::CKKS::ContextData& cc, int slots) {
    return cc.correctionFactorOverride >= 0
               ? (uint32_t)cc.correctionFactorOverride
               : cc.GetBootPrecomputation(slots).correctionFactor;
}

static void btsStageProbe(const char* stage, FIDESlib::CKKS::Ciphertext& ctxt) {
    if (FIDESlib::CKKS::g_btsStageStash) {
        cudaDeviceSynchronize();
        auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
        c->copy(ctxt);
        FIDESlib::CKKS::g_btsStageStash->emplace_back(stage, std::move(c));
    }
}

void FIDESlib::CKKS::BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // deg = q0_bits - scale_bits must not exceed the correction factor, or the uint32
    // subtraction below underflows and corFactor = 1 << garbage silently poisons the bootstrap.
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L - cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop) / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////

    {

        ModRaise(ctxt, slots, correction, prescaled);

        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootPrecomputation(slots).bootK > 0 ? cc.GetBootPrecomputation(slots).bootK : cc.GetBootK();

        double constantEvalMult = pre * (1.0 / (k * cc.N));
        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        constantEvalMult *= cc.getBtsPreScale();

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    btsStageProbe("pre-CtS", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();

        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (auto& rp = cc.GetBootPrecomputation(slots); !rp.evalModCheb.empty())
            approxModReductionSparse(ctxt, scalar, rp.evalModCheb, rp.evalModDaIts);
        else
            approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    uint64_t corFactor = (uint64_t)1 << std::llround(correction);

    btsStageProbe("pre-StC", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (corFactor != 1)
        multIntScalar(ctxt, corFactor);
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
}

namespace FIDESlib::CKKS {

// FIDESLIB_BTS_RAISE_NOSTC: a raise variant built without StC tables serves the staged StC-first finishes only; any
// path about to read its StC fails here instead of running a bootstrap without its SlotsToCoeffs
static void requireStC(const BootstrapPrecomputation& pre, const char* where) {
    if (pre.StC.empty())
        throw std::runtime_error(std::string(where) + ": the bootstrap precomputation in scope (raise drop " +
                                 std::to_string(pre.raise_drop) +
                                 ") has no StC tables (FIDESLIB_BTS_RAISE_NOSTC: StC-first finishes only)");
}

// StC-first last-stage cache entries (FIDESLIB_STC_LAST_CACHE, default 24): ~0.2 GB sparse, 1-2 GB dense each.
// FIDESLIB_STC_CACHE_ASYNC=1: cache misses re-level and evict under GPU-side barriers instead of device syncs
static bool stcCacheAsync() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_STC_CACHE_ASYNC"); return e && std::atoi(e) > 0; }();
    return v;
}
// FIDESLIB_STC_CACHE_STATS=1: hit / miss / evict counts of the StC-first and stc_high relevel caches, at exit
struct StcCacheStats {
    long first_hit = 0, first_miss = 0, first_evict = 0, high_hit = 0, high_miss = 0, high_evict = 0;
    std::set<std::string> first_keys, high_keys;
    bool on = [] { const char* e = std::getenv("FIDESLIB_STC_CACHE_STATS"); return e && *e && *e != '0'; }();
    ~StcCacheStats() {
        if (on)
            std::fprintf(stderr, "[stc_cache] first: hit %ld miss %ld evict %ld keys %zu | high: hit %ld miss %ld evict %ld keys %zu\n",
                         first_hit, first_miss, first_evict, first_keys.size(), high_hit, high_miss, high_evict, high_keys.size());
    }
};
static StcCacheStats g_stcStats;

static size_t lastStageCacheCap() {
    static const size_t cap = [] {
        const char* e = std::getenv("FIDESLIB_STC_LAST_CACHE");
        return (e && *e) ? std::max<size_t>(1, std::strtoull(e, nullptr, 10)) : size_t(24);
    }();
    // under memory pressure (free < FIDESLIB_STC_LAST_CACHE_FREE_GB, default 8) keep one entry
    static const double floor_gb = [] {
        const char* e = std::getenv("FIDESLIB_STC_LAST_CACHE_FREE_GB");
        return (e && *e) ? std::atof(e) : 8.0;
    }();
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess)
        return cap;
    // FIDESLIB_STC_FREE_COUNTS_POOL=1: idle memory held by the allocators (FIDESlib's slabs and the stream-ordered pool,
    // which the cache's plaintexts draw from) counts as free, so the per-block pool trim is not needed to keep entries
    static const bool countPool = [] {   // FIDESLIB_STC_CACHE_POOLFREE: dev's name
        const char* e = std::getenv("FIDESLIB_STC_FREE_COUNTS_POOL");
        if (!(e && *e)) e = std::getenv("FIDESLIB_STC_CACHE_POOLFREE");
        return e && std::atoi(e) > 0;
    }();
    if (countPool && fr < floor_gb * (1ull << 30)) {
        int dev = 0;
        cudaGetDevice(&dev);
        const FIDESlib::GPUpoolStats st = FIDESlib::GPUpoolStatsOf(dev);
        fr += st.idle + (st.cuda_reserved > st.cuda_used ? st.cuda_reserved - st.cuda_used : 0);
    }
    if (fr < floor_gb * (1ull << 30))
        return 1;
    return cap;
}


// FIDESLIB_STC_CACHE_RELEASE=1: the StC-first / stc_high cache entries drop their NTT scratch once built (read-only from
// then on, as bootstrapImpl's entries already do): about half of each entry's device memory
// FIDESLIB_RELEVEL_NOSYNC=1: a cache miss relevels its stage without a device fence per plaintext (one after the batch)
// FIDESLIB_RELEVEL_NOSYNC=1: each cache-miss relevel skips its own fence (the miss path fences once after the batch)
static bool relevelEachSync() {
    static const bool nosync = [] { const char* e = std::getenv("FIDESLIB_RELEVEL_NOSYNC"); return e && std::atoi(e) > 0; }();
    return !nosync;
}
static bool stcHighPow2() {
    static const bool on = [] { const char* e = std::getenv("FIDESLIB_STC_HIGH_POW2"); return e && std::atoi(e) > 0; }();
    return on;
}
static bool stcPow2Key() {
    static const bool on = [] { const char* e = std::getenv("FIDESLIB_STC_POW2_KEY"); return e && std::atoi(e) > 0; }();
    return on;
}
static bool stcCacheRelease() {
    static const bool on = [] { const char* e = std::getenv("FIDESLIB_STC_CACHE_RELEASE"); return e && std::atoi(e) > 0; }();
    return on;
}
// FIDESLIB_STC_CACHE_STATS=2: one line per miss (cache, key, entries, free device memory)
static void stcMissLog(const char* which, const std::string& key, size_t entries) {
    static const bool on = [] { const char* e = std::getenv("FIDESLIB_STC_CACHE_STATS"); return e && std::atoi(e) >= 2; }();
    if (!on) return;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    std::fprintf(stderr, "[stc_miss] %s key %s entries %zu free %.2f GB\n", which, key.c_str(), entries, fr / 1073741824.0);
}

namespace {
// A LIVE ciphertext polynomial's data put in the buffers the host calls `v` (readers of Limb::v.data, e.g. the PCMM
// ring switch, need it there; after an odd number of out-of-place passes it may sit in `aux`, which only the device
// tables follow): copy where limbptr points at aux, then limbptr = v and auxptr = aux. The scratch stays allocated.
// RNSPoly::releaseAux is for read-only plaintexts: on a ciphertext it left auxptr pointing at freed memory, which the
// next out-of-place pass wrote into and swapped in, so the ciphertext shared memory with later allocations (StC-first
// cache entries among them): wrong results on every path after the first alternative-StC call.
struct LiveFix {
    void* v[MAXP];
    void* aux[MAXP];
    uint32_t words[MAXP];
    int n;
};
__global__ void liveCopy_(void* const* lp, LiveFix f) {
    const int i = blockIdx.y;
    if (i >= f.n || f.aux[i] == nullptr || lp[i] != f.aux[i])
        return;
    uint32_t* dst = static_cast<uint32_t*>(f.v[i]);
    const uint32_t* src = static_cast<const uint32_t*>(f.aux[i]);
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < f.words[i]; k += (size_t)gridDim.x * blockDim.x)
        dst[k] = src[k];
}
__global__ void liveTables_(void** lp, void** ap, LiveFix f) {
    const int i = threadIdx.x;
    if (i >= f.n || f.aux[i] == nullptr || (lp[i] != f.v[i] && lp[i] != f.aux[i]))
        return;
    lp[i] = f.v[i];
    if (ap)
        ap[i] = f.aux[i];
}
void liveInV(RNSPoly& p) {
    for (auto& part : p.GPU) {
        cudaSetDevice(part.device);
        auto each = [&](auto&& fn) {
            for (auto& l : part.limb)
                std::visit(fn, l);
            for (auto& l : part.SPECIALlimb)
                std::visit(fn, l);
        };
        each([&](auto& x) { part.s.wait(x.stream); });
        auto fix = [&](std::vector<LimbImpl>& limbs, VectorGPU<void*>& lp, VectorGPU<void*>& ap) {
            const int n = (int)limbs.size();
            if (n == 0 || n > MAXP || lp.data == nullptr || lp.size < n)
                return;
            LiveFix f{};
            f.n = n;
            for (int i = 0; i < n; ++i)
                std::visit(
                    [&](auto& x) {
                        f.v[i] = (void*)x.v.data;
                        f.aux[i] = x.auxReleased ? nullptr : (void*)x.aux.data;
                        f.words[i] = (uint32_t)((size_t)x.v.size * sizeof(*x.v.data) / sizeof(uint32_t));
                    },
                    limbs[i]);
            liveCopy_<<<dim3(32, n), 256, 0, part.s.ptr()>>>(lp.data, f);
            liveTables_<<<1, MAXP, 0, part.s.ptr()>>>(lp.data, (ap.data != nullptr && ap.size >= n) ? ap.data : nullptr, f);
        };
        fix(part.limb, part.limbptr, part.auxptr);
        fix(part.SPECIALlimb, part.SPECIALlimbptr, part.SPECIALauxptr);
        each([&](auto& x) { x.stream.wait(part.s); });
    }
}
// FIDESLIB_STC_ALT_SYNC=bits (diagnostic, default 0): fences in the alternative branch. Host fences (device
// synchronize): 1 after its first stages, 2 after its last stage, 4 at its end. GPU-only barriers (an empty kernel on
// the legacy stream, which every blocking FIDESlib stream orders against both ways; the host does not wait): 8 after
// its first stages, 16 after its last stage. Bisects the missing dependency between back-to-back StC-first inputs.
int stcAltSync() {
    static const int v = [] { const char* e = std::getenv("FIDESLIB_STC_ALT_SYNC"); return e && *e ? std::atoi(e) : 0; }();
    return v;
}
__global__ void legacyBarrier_() {}
// FIDESLIB_STC_ALT_SYNC bits 32 / 64 / 128 (diagnostic): targeted event edges between this ciphertext and the shared
// key-switch machinery, no host wait and no legacy-stream barrier. 32 (tail): the shared digit streams wait on this
// call; 64 (tail): the key-switch scratch polynomials' streams wait on this call; 128 (entry): this call waits on both.
void sharedStreams(ContextData& cc, int g, const std::function<void(Stream&)>& fn, bool digits, bool aux,
                   bool auxS = true, bool auxLimbs = true) {
    if (digits)
        for (auto* dsv : {&cc.digitStreamL(), &cc.digitStream2L()})
            for (auto& per : *dsv)
                if (g < (int)per.size())
                    fn(per[g]);
    if (aux)
        for (RNSPoly* a : {&cc.getKeySwitchAux(), &cc.getKeySwitchAux2()}) {
            if (g >= (int)a->GPU.size())
                continue;
            auto& P = a->GPU[g];
            if (auxS)
                fn(P.s);
            if (!auxLimbs)
                continue;
            for (auto& l : P.limb)
                std::visit([&](auto& x) { fn(x.stream); }, l);
            for (auto& l : P.SPECIALlimb)
                std::visit([&](auto& x) { fn(x.stream); }, l);
        }
}
void joinCt(Ciphertext& ct) {   // every limb stream of the ciphertext joins its partition streams
    for (RNSPoly* p : {&ct.c0, &ct.c1})
        for (auto& part : p->GPU) {
            for (auto& l : part.limb)
                std::visit([&](auto& x) { part.s.wait(x.stream); }, l);
            for (auto& l : part.SPECIALlimb)
                std::visit([&](auto& x) { part.s.wait(x.stream); }, l);
        }
}
void forkCt(Ciphertext& ct) {
    for (RNSPoly* p : {&ct.c0, &ct.c1})
        for (auto& part : p->GPU) {
            for (auto& l : part.limb)
                std::visit([&](auto& x) { x.stream.wait(part.s); }, l);
            for (auto& l : part.SPECIALlimb)
                std::visit([&](auto& x) { x.stream.wait(part.s); }, l);
        }
}
void stcAltEdgeTail(Ciphertext& ct) {
    const int b = stcAltSync();
    if (!(b & (32 | 64)))
        return;
    joinCt(ct);
    for (int g = 0; g < (int)ct.c0.GPU.size(); ++g) {
        cudaSetDevice(ct.c0.GPU[g].device);
        Stream& s0 = ct.c0.GPU[g].s;
        Stream& s1 = ct.c1.GPU[g].s;
        sharedStreams(ct.cc, g, [&](Stream& x) { x.wait(s0); x.wait(s1); }, b & 32, b & 64);
    }
}
// bits 256 / 512 / 1024 (entry, subsets of 128): wait on the digit streams only / the scratch polys' partition streams
// only / the scratch polys' limb and special-limb streams only
void stcAltEdgeEntry(Ciphertext& ct) {
    const int b = stcAltSync();
    if (!(b & (128 | 256 | 512 | 1024)))
        return;
    const bool all = b & 128;
    for (int g = 0; g < (int)ct.c0.GPU.size(); ++g) {
        cudaSetDevice(ct.c0.GPU[g].device);
        Stream& s0 = ct.c0.GPU[g].s;
        Stream& s1 = ct.c1.GPU[g].s;
        sharedStreams(ct.cc, g, [&](Stream& x) { s0.wait(x); s1.wait(x); }, all || (b & 256), all || (b & (512 | 1024)),
                      all || (b & 512), all || (b & 1024));
    }
    forkCt(ct);
}
void stcAltFence(int hostBit, int gpuBit) {
    if (stcAltSync() & hostBit)
        cudaDeviceSynchronize();
    if (stcAltSync() & gpuBit)
        legacyBarrier_<<<1, 1, 0, cudaStreamLegacy>>>();
}
// FIDESLIB_STC_ALT_PROBE=n (diagnostic): for the first n StC-first inputs, after a device fence, which buffer each
// limb's device tables name: limbptr / auxptr vs the host v / aux ('v', 'a', '?'), per polynomial and step
void probeLive(const char* step, Ciphertext& ct, bool alt) {
    static int left = [] { const char* e = std::getenv("FIDESLIB_STC_ALT_PROBE"); return e && *e ? std::atoi(e) : 0; }();
    if (left <= 0)
        return;
    cudaDeviceSynchronize();
    std::string out = std::string("[stc_probe] ") + (alt ? "alt " : "ship ") + step + " lvl " + std::to_string(ct.getLevel());
    for (RNSPoly* p : {&ct.c0, &ct.c1}) {
        auto& part = p->GPU.at(0);
        const int n = (int)part.limb.size();
        std::vector<void*> lp(n, nullptr), ap(n, nullptr);
        if (part.limbptr.data && part.limbptr.size >= n)
            cudaMemcpy(lp.data(), part.limbptr.data, n * sizeof(void*), cudaMemcpyDeviceToHost);
        if (part.auxptr.data && part.auxptr.size >= n)
            cudaMemcpy(ap.data(), part.auxptr.data, n * sizeof(void*), cudaMemcpyDeviceToHost);
        out += " |";
        for (int i = 0; i < n; ++i)
            std::visit(
                [&](auto& x) {
                    auto code = [&](void* t) { return t == (void*)x.v.data ? 'v' : t == (void*)x.aux.data ? 'a' : '?'; };
                    out += ' ';
                    out += code(lp[i]);
                    out += code(ap[i]);
                },
                part.limb[i]);
    }
    std::cerr << out << "\n";
    if (std::string(step) == "end")
        --left;
}
}  // namespace

static void bootstrapImpl(Ciphertext& ctxt, const int slots, const bool prescaled, const bool allowStcFirst);

std::function<void(Ciphertext&, int, bool)> g_bootstrapOverride;

// FIDESLIB_BTS_GRAPH=1: every bootstrap shape (input level/degree/scale, slots, key, the per-call context state) is
// run eagerly once (it warms the plaintext caches and the scratch buffers), captured as a CUDA graph of the staged
// bootstrap the second time, and replayed from then on: input copied into the graph's own ciphertext, a legacy-stream
// barrier on each side (shared scratch buffers are ordered across the boundary), output copied back. The s = 1
// route stays with the runtime's override (SPRU). FIDESLIB_BTS_GRAPH_LOG=1 reports captures and fallbacks.
namespace {
bool btsGraphOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_BTS_GRAPH"); return e && std::atoi(e) > 0; }();
    return v;
}
bool btsGraphLog() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_BTS_GRAPH_LOG"); return e && std::atoi(e) > 0; }();
    return v;
}
uint64_t bitsOf(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}
struct BtsGraphKey {
    int slots, prescaled, ctSlots, l0, l1, deg, modUp, raiseDrop, real, cfOverride, rescaleOne;
    uint64_t nf, preScale;
    std::string keyID;
    auto tie() const {
        return std::tie(slots, prescaled, ctSlots, l0, l1, deg, modUp, raiseDrop, real, cfOverride, rescaleOne, nf,
                        preScale, keyID);
    }
    bool operator<(const BtsGraphKey& o) const { return tie() < o.tie(); }
};
struct BtsGraph {
    int seen = 0;
    bool failed = false;
    cudaGraphExec_t exec = nullptr;
    FIDESlib::CaptureRegion* region = nullptr;   // the pinned snapshots its H2D nodes read
    // the output's metadata (the shared output ciphertext's level is re-seated before every copy-out)
    int outLevel = 0, outDeg = 1, outSlots = 0;
    double outNF = 0;
    KeyHash outKey;
};
// One input and one output ciphertext for all graphs, grown to the top level once: their limb addresses are what the
// graphs read and write, and they never move (dropToLevel only lowers the level, grow only appends).
struct BtsGraphIO {
    std::unique_ptr<Ciphertext> in, out;
    FIDESlib::Stream origin;
    cudaEvent_t pre = nullptr, post = nullptr;
};
std::map<BtsGraphKey, BtsGraph>& btsGraphs() {
    static auto* m = new std::map<BtsGraphKey, BtsGraph>;   // never destroyed: graph memory is never released
    return *m;
}
BtsGraphIO& btsGraphIO(Context& cc_) {
    static auto* io = [&] {
        auto* p = new BtsGraphIO;
        p->in = std::make_unique<Ciphertext>(cc_);
        p->out = std::make_unique<Ciphertext>(cc_);
        for (RNSPoly* q : {&p->in->c0, &p->in->c1, &p->out->c0, &p->out->c1})
            q->grow(cc_->L);
        p->origin.init();
        cudaEventCreateWithFlags(&p->pre, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&p->post, cudaEventDisableTiming);
        cudaDeviceSynchronize();
        return p;
    }();
    return *io;
}
__global__ void btsGraphFence() {}

bool btsGraphCaptureFn(BtsGraph& e, Ciphertext& ctxt, const std::function<void(Ciphertext&)>& body);
bool btsGraphCapture(BtsGraph& e, Ciphertext& ctxt, const int slots, const bool prescaled) {
    return btsGraphCaptureFn(e, ctxt, [&](Ciphertext& w) { BootstrapStaged(w, slots, prescaled); });
}
bool btsGraphCaptureFn(BtsGraph& e, Ciphertext& ctxt, const std::function<void(Ciphertext&)>& body) {
    Context& cc_ = ctxt.cc_;
    BtsGraphIO& io = btsGraphIO(cc_);
    io.in->copy(ctxt);
    cudaDeviceSynchronize();
    e.region = FIDESlib::captureRegionNew();
    cudaStream_t o = io.origin.ptr();
    if (!FIDESlib::captureBegin(o, e.region)) {
        FIDESlib::captureRegionFree(e.region);
        e.region = nullptr;
        return false;
    }
    try {
        Ciphertext w(cc_);   // from the graph pool
        w.copy(*io.in);
        body(w);
        io.out->copy(w);
    } catch (const std::exception& ex) {
        FIDESlib::captureInvalidate("exception during capture");
        if (btsGraphLog()) std::cerr << "[bts_graph] exception during capture: " << ex.what() << "\n";
    }
    cudaGraph_t g = FIDESlib::captureEnd();
    if (!g) {
        if (btsGraphLog())
            std::cerr << "[bts_graph] capture failed (in level " << ctxt.getLevel() << "): "
                      << (FIDESlib::g_cap.why ? FIDESlib::g_cap.why : "?") << "\n";
        FIDESlib::captureRegionFree(e.region);
        e.region = nullptr;
        return false;
    }
    size_t nodes = 0;
    cudaGraphGetNodes(g, nullptr, &nodes);
    const cudaError_t err = cudaGraphInstantiateWithFlags(&e.exec, g, 0);
    cudaGraphDestroy(g);
    if (err != cudaSuccess) {
        cudaGetLastError();
        e.exec = nullptr;
        if (btsGraphLog()) std::cerr << "[bts_graph] instantiate failed: " << cudaGetErrorString(err) << "\n";
        FIDESlib::captureRegionFree(e.region);
        e.region = nullptr;
        return false;
    }
    Ciphertext& out = *io.out;
    e.outLevel = out.getLevel();
    e.outDeg = out.NoiseLevel;
    e.outSlots = out.slots;
    e.outNF = out.NoiseFactor;
    e.outKey = out.keyID;
    if (btsGraphLog())
        std::cerr << "[bts_graph] captured in level " << ctxt.getLevel() << " deg "
                  << ctxt.NoiseLevel << " -> out level " << e.outLevel << " deg " << e.outDeg << ", " << nodes
                  << " nodes (" << btsGraphs().size() << " shapes)\n";
    return true;
}

void btsGraphReplay(BtsGraph& e, Ciphertext& ctxt) {
    BtsGraphIO& io = btsGraphIO(ctxt.cc_);
    io.in->copy(ctxt);
    cudaStream_t o = io.origin.ptr();
    btsGraphFence<<<1, 1, 0, 0>>>();   // legacy stream: after every earlier op on the (blocking) streams
    cudaEventRecord(io.pre, 0);
    cudaStreamWaitEvent(o, io.pre, 0);
    cudaGraphLaunch(e.exec, o);
    cudaEventRecord(io.post, o);
    cudaStreamWaitEvent(0, io.post, 0);
    btsGraphFence<<<1, 1, 0, 0>>>();   // every later op on the blocking streams waits for the graph
    Ciphertext& out = *io.out;
    for (RNSPoly* p : {&out.c0, &out.c1}) {
        p->dropToLevel(e.outLevel);
        p->grow(e.outLevel);
        for (auto& part : p->GPU)
            cudaStreamWaitEvent(part.s.ptr(), io.post, 0);
    }
    out.NoiseLevel = e.outDeg;
    out.slots = e.outSlots;
    out.NoiseFactor = e.outNF;
    out.keyID = e.outKey;
    ctxt.copy(out);
}

bool btsGraphRun(Ciphertext& ctxt, const int slots, const bool prescaled) {
    ContextData& cc = ctxt.cc;
    if (slots == 1 || cc.GPUid.size() != 1)
        return false;
    const BtsGraphKey k{slots, prescaled, ctxt.slots, ctxt.c0.getLevel(), ctxt.c1.getLevel(), ctxt.NoiseLevel,
                        (int)ctxt.c0.isModUp() + 2 * (int)ctxt.c1.isModUp(), cc.btsRaiseDropEffective(),
                        (int)cc.getBtsRealPayload(), cc.getCorrectionFactorOverride(), (int)g_rescaleOnePrime,
                        bitsOf(ctxt.NoiseFactor), bitsOf(cc.getBtsPreScale()), ctxt.keyID};
    BtsGraph& e = btsGraphs()[k];
    if (e.failed)
        return false;
    if (!e.exec) {
        if (e.seen++ < 1)
            return false;
        if (!btsGraphCapture(e, ctxt, slots, prescaled)) {
            e.failed = true;
            return false;
        }
    }
    btsGraphReplay(e, ctxt);
    return true;
}
}  // namespace

void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    if (btsGraphOn() && !FIDESlib::g_cap.active && btsGraphRun(ctxt, slots, prescaled))
        return;
    static thread_local bool inOverride = false;
    if (g_bootstrapOverride && !inOverride) {
        inOverride = true;
        try {
            g_bootstrapOverride(ctxt, slots, prescaled);
        } catch (...) {
            inOverride = false;
            throw;
        }
        inOverride = false;
        return;
    }
    // EvalRound lives in the staged path (the one the Python bootstrap drives)
    static const bool staged = [] {
        const char* e = std::getenv("FIDESLIB_EVALROUND");
        return e && std::atoi(e) > 0;
    }();
    if (staged) {
        BootstrapStaged(ctxt, slots, prescaled);
        return;
    }
    bootstrapImpl(ctxt, slots, prescaled, true);
}

// FIDESLIB_BTS_STC_FIRST_AB=<dir>: in-model A/B of the StC-first order — the first few bootstraps that take it also run the
// shipped order on a copy of the SAME input; input and both outputs are exact-decrypted (FIDESLIB_DIAG_SK=1) into <dir>
// for the offline comparison (logs/bts_traffic/stcfirst_ab.py).
static void bootstrapImpl(Ciphertext& ctxt, const int slots, const bool prescaled, const bool allowStcFirst) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    assert(slots >= ctxt.slots);
    int old_slots = ctxt.slots;

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // deg = q0_bits - scale_bits must not exceed the correction factor, or the uint32
    // subtraction below underflows and corFactor = 1 << garbage silently poisons the bootstrap.
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L - cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop) / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;
    bool shiftedFlow = false;  // FIDESLIB_BTS_SHIFT path taken (exact scaling, +d level)

    // FIDESLIB_BTS_STC_FIRST (lever A, slim order): SlotsToCoeffs on the INPUT ciphertext, at <= entry + 1 limbs,
    // before the raise. The raised polynomial then carries (Re z ; Im z) as coefficients and EvalMod's output is the
    // bootstrap output. The StC levels are paid from the input (entry = 2d-1 limbs for ModRaise's adjust+rescale plus
    // d per StC stage), the landing moves up by the same amount. Sparse routes: the shipped StC_sparse is
    // diag(U0, i U0); the fold turns its output into (1+Y^s) Q(Y) / 2, i.e. the halves (a-b ; a+b)/2 come out of
    // EvalMod, and the mask (1-i ; 1+i) + rotate(slots) recombines z = a + ib (see BootstrapPrecomputation.cuh).
    // lever B route (standard order); FIDESLIB_BTS_SPARSE_B_RUN=0 keeps the precomputation but runs the shipped path
    // (calibration: the same context, both flows)
    const bool sparseB = cc.GetBootPrecomputation(slots).sparseB != nullptr &&
                         [] { const char* e = std::getenv("FIDESLIB_BTS_SPARSE_B_RUN"); return !(e && std::atoi(e) == 0); }();
    // an input without room for the StC levels takes the shipped order (as btsBegin)
    const bool stcFirst = cc.GetBootPrecomputation(slots).stc_first_mode > 0 && !sparseB && allowStcFirst && !cc.stcFirstForbid &&
                          !stcFirstStagedOnly() &&
                          (int)ctxt.getLevel() - (ctxt.NoiseLevel == 2 ? cc.compositeDegree() : 0) >=
                              cc.GetBootPrecomputation(slots).stc_first_entry;
    static int abCount = 0;
    std::unique_ptr<Ciphertext> abRef;
    std::string abDir;
    if (stcFirst) {
        if (const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_AB"); e && *e && abCount < 8) {
            abDir = e;
            const std::string base = abDir + "/" + std::to_string(abCount) + "-s" + std::to_string(slots);
            cudaDeviceSynchronize();
            exactDecryptDump(ctxt, (base + "-in.ct").c_str());
            abRef = std::make_unique<Ciphertext>(cc_);
            abRef->copy(ctxt);
            bootstrapImpl(*abRef, slots, prescaled, false);  // the shipped order on the same input
            cudaDeviceSynchronize();
            exactDecryptDump(*abRef, (base + "-std.ct").c_str());
            std::cerr << "[stc_first_ab] #" << abCount << " slots=" << slots << " in level " << ctxt.getLevel() << " deg "
                      << ctxt.NoiseLevel << " NF " << ctxt.NoiseFactor << " cf " << effCorrectionFactor(cc, slots)
                      << " -> std out level " << abRef->getLevel() << " deg " << abRef->NoiseLevel << "\n";
        }
    }
    const bool stcFolded = cc.GetBootPrecomputation(slots).stc_first_mode == 2;  // last stage carries the adjust
    if (stcFirst) {
        auto& pre = cc.GetBootPrecomputation(slots);
        if (ctxt.NoiseLevel == 2)
            ctxt.rescale();
        const int entry = pre.stc_first_entry;
        if (ctxt.getLevel() < entry)
            throw std::runtime_error("[stc_first] bootstrap input at limb index " + std::to_string(ctxt.getLevel()) +
                                     " is below the StC-first entry level " + std::to_string(entry) +
                                     " (the input must keep " + std::to_string(entry + 1) + " limbs)");
        if (ctxt.getLevel() > entry)
            ctxt.dropToLevel(entry);
        ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;  // the 2*slots view the sparse StC stages are indexed in
        btsStageProbe("pre-StC1", ctxt);
        if (!stcFolded) {
            if (isLT)
                EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, pre.LT_first);
            else
                EvalLTStages(ctxt, pre.StC_first, "StC1-stage-");
        } else {
            // The last stage's plaintexts carry 2^-deg (scaleDec compensation), 2^-correction and the FLEXIBLEAUTO
            // re-nominalization, so that after multPt + ModRaise's prescaled rescale the integers are v*targetSF*2^-c:
            // factor = 2^-CF * (targetSF / sf(d-1)) * (sf(entry) / NF_in)   [NF_in = true scale after the drop; the
            // stages before the last multiply NF by sf(l)/sf(entry), their rescales telescope]
            const int d = cc.compositeDegree();
            const int lastL = 2 * d - 1;
            const double nfIn = ctxt.NoiseFactor;
            const double targetSF = cc.sfAtLimb(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) * d - d * pre.raise_drop);
            const int cf = (int)effCorrectionFactor(cc, slots);
            // The input-scale ratio MUST ride the plaintext (before the raise): folding it into the exact post-raise
            // scaling also scales the q0*I term, which EvalMod needs as integer multiples (chain96/97: garbage).
            const double ratio = cc.sfAtLimb(entry) / nfIn;
            // built at a reference CF, keyed by the input scale alone (see btsStcFirstInput)
            const int cfStage = std::max(cf, kStcFirstCfRef);
            if (cfStage > cf)
                multIntScalar(ctxt, uint64_t{1} << (cfStage - cf));
            const double factor = std::ldexp(1.0, -cfStage) * (targetSF / cc.sfAtLimb(d - 1)) * ratio;
            const std::pair<int, long long> key{cfStage, std::llround(std::log2(nfIn) * 1e6)};  // nominal scales per level
            // The cached last stages must keep their special limbs (the Q-only variant lost 15 bits, chain93); the
            // cache is bounded (each entry ~0.2 GB; one per input scale).
            const size_t kLastStageCache = lastStageCacheCap();
            if (isLT) {
                auto it = pre.lt_first_last.find(key);
                if (it == pre.lt_first_last.end()) {
                    if (pre.lt_first_last.size() >= kLastStageCache) {
                        cudaDeviceSynchronize();  // the evicted plaintexts may still be read by the previous bootstrap's kernels
                        while (pre.lt_first_last.size() >= kLastStageCache)
                            pre.lt_first_last.erase(pre.lt_first_last.begin());
                    }
                    std::vector<Plaintext> v;
                    v.reserve(pre.LT.invA.size());
                    for (auto& pt : pre.LT.invA)
                        v.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, factor));
                    cudaDeviceSynchronize();
                    for (auto& pt : v)
                        pt.c0.releaseAux();   // read-only from here on
                    it = pre.lt_first_last.emplace(key, std::move(v)).first;
                }
                EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, it->second);
            } else {
                if (!pre.StC_first.empty())
                    EvalLTStages(ctxt, pre.StC_first, "StC1-stage-");
                auto it = pre.stc_first_last.find(key);
                if (it == pre.stc_first_last.end()) {
                    if (pre.stc_first_last.size() >= kLastStageCache) {
                        cudaDeviceSynchronize();  // the evicted plaintexts may still be read by the previous bootstrap's kernels
                        while (pre.stc_first_last.size() >= kLastStageCache)
                            pre.stc_first_last.erase(pre.stc_first_last.begin());
                    }
                    requireStC(pre, "stc_first");
                    auto& st = pre.StC.back();
                    BootstrapPrecomputation::LTstep o;
                    o.slots = st.slots; o.bStep = st.bStep; o.gStep = st.gStep; o.rotIn = st.rotIn; o.rotOut = st.rotOut;
                    o.A.reserve(st.A.size());
                    for (auto& pt : st.A)
                        o.A.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, factor));
                    std::vector<BootstrapPrecomputation::LTstep> v;
                    v.push_back(std::move(o));
                    cudaDeviceSynchronize();
                    for (auto& stp : v)
                        for (auto& pt : stp.A)
                            pt.c0.releaseAux();   // read-only from here on
                    it = pre.stc_first_last.emplace(key, std::move(v)).first;
                }
                EvalLTStages(ctxt, it->second, "StC1-last-");
            }
            // ModRaise's prescaled path expects the deg-2 ciphertext at limb index 2d-1
            if (ctxt.getLevel() != lastL || ctxt.NoiseLevel != 2)
                throw std::runtime_error("[stc_first] folded last stage left the ciphertext at limb index " +
                                         std::to_string(ctxt.getLevel()) + " deg " + std::to_string(ctxt.NoiseLevel));
        }
        btsStageProbe("post-StC1", ctxt);
        if (const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_DIAG"); e && std::atoi(e) > 0) {
            static int once = 0;
            if (once++ < 2)
                std::cerr << "[stc_first] diag: entry limb " << entry << " sf(entry) " << cc.sfAtLimb(entry) << " sf(L) "
                          << cc.sfAtLimb(cc.L) << " post-StC level " << ctxt.getLevel() << " NoiseLevel " << ctxt.NoiseLevel
                          << " NF " << ctxt.NoiseFactor << " sf(level) " << cc.sfAtLimb(ctxt.getLevel())
                          << " sf(level-d) " << cc.sfAtLimb(ctxt.getLevel() - cc.compositeDegree()) << " q0 "
                          << qDouble << " deg " << deg << " corr " << correction << "\n";
        }
    }

    {
        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootPrecomputation(slots).bootK > 0 ? cc.GetBootPrecomputation(slots).bootK : cc.GetBootK();
        double constantEvalMult = pre * (1.0 / (k * cc.N));
        constantEvalMult *= cc.getBtsPreScale();
        // FIDESLIB_BTS_SHIFT: apply it exactly inside the raise (level kept) when the precomputation was built for it
        const double baked = cc.GetBootPrecomputation(slots).cts0_const;
        // diagnostic: FIDESLIB_BTS_EXACT_ONLY=1 = exact scaling inside the raise, then an exact LevelReduce to the
        // standard level, with the standard (unshifted) plaintexts — isolates the scaling step from the +d flow
        const bool exactOnly = [] { const char* e = std::getenv("FIDESLIB_BTS_EXACT_ONLY"); return e && std::atoi(e) > 0; }();
        const bool exactConst = exactOnly || (baked != 0 && std::fabs(baked / constantEvalMult - 1.0) < 1e-9);
        if (baked != 0 && !exactConst)
            std::cerr << "[bts_shift] WARNING: baked constant " << baked << " != runtime " << constantEvalMult
                      << " (or sparse slots): falling back to multScalar, the level gain is lost\n";
        const bool aksOn = exactConst && baked != 0 && cc.GetBootPrecomputation(slots).aks0 != nullptr;
        ModRaise(ctxt, slots, correction, prescaled || (stcFirst && stcFolded), sparse_encaps, exactConst ? constantEvalMult : 0.0, aksOn);
        shiftedFlow = exactConst && baked != 0 && !aksOn;
        if (exactOnly && baked == 0 && exactConst)
            ctxt.dropToLevel(ctxt.getLevel() - cc.compositeDegree());  // standard plaintexts expect L - d
        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)

        // TO-DO: The 1/32 scale will be pre-applied with OpenFHE v1.4, so remove it from here

        /*
        if (sparse_encaps) {
            constantEvalMult = pre * (1.0 / (k * cc.N) / 32);
        }
        */

        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        if (!exactConst)
            ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    btsStageProbe("pre-CtS", ctxt);
    if (g_btsStageStash && shiftedFlow && !cc.GetBootPrecomputation(slots).CtS_orig.empty() &&
        cc.GetBootPrecomputation(slots).cts0_t == 0) {
        // within-run reference: the same raised ciphertext through the STANDARD flow (level dropped exactly by d,
        // original plaintexts swapped in); stages probed as R-*
        auto& pre = cc.GetBootPrecomputation(slots);
        Ciphertext ref(cc_);
        ref.copy(ctxt);
        ref.dropToLevel(ref.getLevel() - cc.compositeDegree());
        std::swap(pre.CtS, pre.CtS_orig);
        std::swap(pre.StC, pre.StC_orig);
        btsStageProbe("R-pre-CtS", ref);
        EvalCoeffsToSlots(ref, slots, false);
        btsStageProbe("R-post-CtS", ref);
        {
            Ciphertext raux(cc_);
            raux.conjugate(ref);
            Ciphertext encI(cc_);
            encI.sub(ref, raux);
            ref.add(raux);
            encI.multMonomial(3 * 2 * cc.N / 4);
            approxModReduction(ref, encI, cc.GetEvalKey(ref.keyID), scalar);
        }
        if (ref.NoiseLevel == 2)
            ref.rescale();
        btsStageProbe("R-post-EvalMod", ref);
        EvalCoeffsToSlots(ref, slots, true);
        btsStageProbe("R-post-StC", ref);
        if (const uint64_t cf = (uint64_t)1 << std::llround(correction); cf != 1)
            multIntScalar(ref, cf);
        btsStageProbe("R-end", ref);
        std::swap(pre.CtS, pre.CtS_orig);
        std::swap(pre.StC, pre.StC_orig);
    }
    if (sparseB) {
        if (g_btsStageStash) {  // reference: the shipped sparse CtS (+ conj) and EvalMod on the SAME folded ciphertext
            Ciphertext ref(cc_), raux(cc_);
            ref.copy(ctxt);
            ref.slots = 2 * slots;
            EvalCoeffsToSlots(ref, slots, false);
            raux.conjugate(ref);
            ref.add(raux);
            btsStageProbe("post-CtS-std", ref);
            approxModReductionSparse(ref, scalar);
            if (ref.NoiseLevel == 2)
                ref.rescale();
            btsStageProbe("post-EvalMod-std", ref);
        }
        SparseBCoeffsToSlots(ctxt, slots, cc.GetBootPrecomputation(slots));
    } else if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
        auto& preS = cc.GetBootPrecomputation(slots);  // FIDESLIB_CTS_SINGLE window reaching the last CtS stage
        if (preS.cts_single && preS.cts_single_lo + preS.cts_single == (int)preS.CtS.size() && ctxt.NoiseLevel == 2) {
            RescaleOneScope one(true);
            ctxt.rescale();
        }
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else if (sparseB) {
        auto& sb = *cc.GetBootPrecomputation(slots).sparseB;
        approxModReductionSparse(ctxt, scalar, sb.cheb, sb.daIts);  // the conj-add is part of SparseBCoeffsToSlots
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, scalar);
    }

    // StC-first dense output stays a lazy deg-2 (the landing convention of every other route: the planner models a
    // deg-2 landing realized to deg-1 one level lower; a deg-1 output here broke the plan's degree pins)
    if (ctxt.NoiseLevel == 2 && !(stcFirst && cc.N / 2 == slots)) {
        ctxt.rescale();
    }

    btsStageProbe("post-EvalMod", ctxt);
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);

    if (stcFirst) {  // slim order: EvalMod's output is the result; the post-factor is 2^(correction + deg) = 2^CF
        const uint64_t cfAll = corFactor << cc.GetBootPrecomputation(slots).stc_first_deg;
        if (cfAll != 1)
            multIntScalar(ctxt, cfAll);
        if (cc.N / 2 != slots) {
            auto& pre = cc.GetBootPrecomputation(slots);
            const int lv = ctxt.getLevel();
            auto it = pre.stc_first_mask_at.find(lv);
            if (it == pre.stc_first_mask_at.end()) {
                const int sh = (lv - pre.stc_first_mask->c0.getLevel()) / cc.compositeDegree();
                it = pre.stc_first_mask_at.emplace(lv, relevelPlaintext(cc_, cc, *pre.stc_first_mask, sh, 1.0)).first;
            }
            ctxt.multPt(it->second, false);
            aux.rotate(ctxt, slots);
            ctxt.add(aux);
        }
        if (mixedChain && ctxt.NoiseLevel == 2)
            ctxt.rescale();
        btsStageProbe("end", ctxt);
        if (abRef) {
            cudaDeviceSynchronize();
            exactDecryptDump(ctxt, (abDir + "/" + std::to_string(abCount) + "-s" + std::to_string(slots) + "-slim.ct").c_str());
            std::cerr << "[stc_first_ab] #" << abCount << " slim out level " << ctxt.getLevel() << " deg " << ctxt.NoiseLevel
                      << " NF " << ctxt.NoiseFactor << "\n";
            ++abCount;
        }
        ctxt.slots = old_slots;
        return;
    }

    // A ciphertext above the StC plaintexts' level (FIDESLIB_BTS_SHIFT with FIDESLIB_BTS_SHIFT_STC=0) is dropped to
    // it first: an exact LevelReduce, instead of the plaintext-above-ciphertext adjust path.
    if (!isLT && !sparseB)
        requireStC(cc.GetBootPrecomputation(slots), "bootstrap");
    if (!isLT && !sparseB && !cc.GetBootPrecomputation(slots).StC.empty()) {
        const int stcL = cc.GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
        if (ctxt.getLevel() > stcL) {
            if (ctxt.NoiseLevel == 2)
                ctxt.rescale();
            if (ctxt.getLevel() > stcL)
                ctxt.dropToLevel(stcL);
        }
    }
    btsStageProbe("pre-StC", ctxt);
    if (sparseB) {
        SparseBSlotsToCoeffs(ctxt, slots, cc.GetBootPrecomputation(slots));  // includes the final rot(n/2) + add
    } else if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots && !sparseB) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (corFactor != 1)
        multIntScalar(ctxt, corFactor);
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }

    ctxt.slots = old_slots;
}
}  // namespace FIDESlib::CKKS

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
    ContextData& cc = *cc_;
    SetCurrentContext(cc_);
    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
    #if NATIVEINT != 128
        if (deg > static_cast<int32_t>(m_correctionFactor)) {
            OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                          std::to_string(m_correctionFactor) + "].");
        }
    #endif
        */
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;

    double res = 0.0;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        const int d_ = cc.compositeDegree();
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        double targetSF = cc.sfAtLimb(cc.L - lvl * d_);
        // composite: the pre-raise ciphertext sits at 2 LEVELS = 2d limbs; its scale lives at
        // limb 2d-1 and the adjust's rescale drops the top d primes (their product).
        double sourceSF = cc.sfAtLimb(2 * d_ - 1);  // ciphertext->GetScalingFactor();
        uint32_t numTowers = 2 * d_;                // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = cc.modReduceProduct(2 * d_ - 1);
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        res = adjustmentFactor;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        res = std::pow((double)2.0, (double)-1.0 * (double)correction);
    }

    return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled,
                              const bool sparse_encaps, const double exactScale, const bool aksStage0) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    ContextData& cc = ctxt.cc;
    btsStageProbe("MR-entry", ctxt);
    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    if (!prescaled) {
        assert(ctxt.getLevel() + 1 - cc.compositeDegree() * ctxt.NoiseLevel >= 1);
    } else {  // deg-1 at the bottom, or deg-2 one composite level above it
        assert(ctxt.getLevel() + 1 == cc.compositeDegree() * ctxt.NoiseLevel);
    }
    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial 2 ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        std::cout << correction << std::endl;
        std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }

    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        // the raise target's scale (FIDESLIB_BTS_RAISE_DROP: the target is raise_drop composite levels below the top)
        double targetSF = cc.sfAtLimb(cc.L - lvl * cc.compositeDegree() -
                                      cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);
        double sourceSF = ctxt.NoiseFactor;        // ciphertext->GetScalingFactor();
        uint32_t numTowers = ctxt.getLevel() + 1;  // ciphertext->GetElements()[0].GetNumOfElements();
        // composite: the adjust's rescale drops the top d primes — divide by their product
        double modToDrop = cc.modReduceProduct(ctxt.getLevel());
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        // multScalar encodes the scalar at the level's NOMINAL scale sf(level); the input's true scale is sourceSF.
        // integers v*sourceSF -> v*sourceSF*adj*sf(level)/modToDrop = v*targetSF*2^-c for ANY sourceSF (the original
        // (targetSF/sourceSF)*(modToDrop/sourceSF) is the sourceSF == sf(level) case; a ciphertext level-reduced
        // from a higher level, e.g. the StC-first input, is not nominal).
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / cc.sfAtLimb(ctxt.getLevel()));
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;

        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(adjustmentFactor);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            //cc->EvalMultInPlace(ciphertext, adjustmentFactor);
            ctxt.rescale();
            ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
        ctxt.NoiseFactor = targetSF;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.rescale();
            ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
    }

    btsStageProbe("MR-bottom", ctxt);
    if (sparse_encaps) {
        if (cc.compositeDegree() > 1) {
            // COMPOSITESCALING: M-4 as a STANDARD hybrid keyswitch in the MAIN context at the
            // composite bottom — the single-tower helper context cannot host a d-limb ct
            // (its digit tables stop at k=0; see BootstrapPrecomputation::sparse_atob).
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_atob);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;
            auto sparse_context_use = sparse_context.lock();
            Ciphertext sparse_ctxt(sparse_context_use);
            auto& atob = CKKS::GetSecretSwitchingKey(ctxt.cc_, sparse_context_use, ctxt.keyID);

            sparse_ctxt.reinterpretContext(ctxt);
            sparse_ctxt.keySwitch(atob);
            ctxt.reinterpretContext(sparse_ctxt);
        }
    }
    btsStageProbe("MR-atob", ctxt);

    //   std::cout << "Boot start " << std::endl;
    // auto ctxtDCRT = raised->GetElements();
    if constexpr (PRINT) {
        std::cout << "Adjustment 1: ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    // The exact post-raise scaling below (smallIntScalarMultiply) reads only the first 3 limbs back in coefficient form
    // and rewrites every limb before its own NTT, so the NTT of all raised limbs is dead work: the raised polynomials stay
    // in coefficient form and are handed over as such (bit-exact). FIDESLIB_MODRAISE_COEFF=0 restores the NTT.
    static const bool kModRaiseCoeff = [] { const char* e = std::getenv("FIDESLIB_MODRAISE_COEFF"); return !(e && *e && std::atoi(e) == 0); }();
    const bool shiftFused = sparse_encaps && cc.compositeDegree() > 1 && cc.GetBootPrecomputation(slots).ghs_btoa &&
                            [] { const char* e = std::getenv("FIDESLIB_BTS_SHIFT_FUSED"); return e && std::atoi(e) > 0; }();
    const bool raiseCoeff = kModRaiseCoeff && !PRINT && exactScale > 0 && !aksStage0 && !shiftFused &&
                            !g_btsStageStash && cc.compositeDegree() > 1;
    ctxt.c0.INTT(cc.batch, true);

    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //   std::cout << "Grow" << std::endl;
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) -
                  cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);  // FIDESLIB_BTS_RAISE_DROP
    //   std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c0.compositeModRaise();
    else
        ctxt.c0.broadcastLimb0();
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (!raiseCoeff)
        ctxt.c0.NTT(cc.batch, true);
    // std::cout << cc.batch << std::endl;
    if constexpr (PRINT) {
        std::cout << "ModRaise ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.INTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1 ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //  std::cout << "Grow" << std::endl;
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) -
                  cc.compositeDegree() * cc.GetBootPrecomputation(slots).raise_drop);  // FIDESLIB_BTS_RAISE_DROP
    //  std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        std::cout << "Adjustment c1  ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c1.compositeModRaise();
    else
        ctxt.c1.broadcastLimb0();
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (!raiseCoeff)
        ctxt.c1.NTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    btsStageProbe("MR-raised", ctxt);
    // FIDESLIB_BTS_SHIFT: the EvalMod constant as an exact scaling of the small-coefficient ciphertext, before the
    // encapsulation switch makes the coefficients uniform (CKKS/SmallInt.cuh). Replaces multScalar + rescale.
    bool fusedSwitched = false;
    if (exactScale > 0 && !aksStage0) {  // lever 1b: the aggregated stage 0 scales, switches and transforms at once
        const int tt = cc.GetBootPrecomputation(slots).cts0_t;
        const double tf = std::ldexp(1.0, tt);
        const double exactScaleT = exactScale * tf;  // the 2^t rides the scale bookkeeping (stage-0 plaintexts carry 2^-t)
        if (g_btsStageStash && sparse_encaps && cc.compositeDegree() > 1) {
            // trace (small-valued => faithful decode): the shipped path and the two scaled-then-switched variants
            auto& pre = cc.GetBootPrecomputation(slots);
            {
                Ciphertext w(ctxt.cc_);
                w.copy(ctxt);
                w.keySwitch(*pre.sparse_btoa);
                w.multScalar(exactScale, false);
                w.rescale();
                btsStageProbe("X-std", w);
            }
            Ciphertext sc(ctxt.cc_);
            sc.copy(ctxt);
            smallIntScalarMultiply(cc, sc, exactScaleT);
            sc.NoiseFactor *= tf;
            if (const char* td = std::getenv("BTS_TRACE_DIR")) {  // raw polynomials before/after the exact scaling
                exactCtPolyDump(ctxt, (std::string(td) + "/P-raw").c_str());
                exactCtPolyDump(sc, (std::string(td) + "/P-scaled").c_str());
            }
            btsStageProbe("X-scaled", sc);
            {
                Ciphertext h(ctxt.cc_);
                h.copy(sc);
                h.keySwitch(*pre.sparse_btoa);
                btsStageProbe("X-hyb", h);
            }
            if (pre.ghs_btoa) {
                Ciphertext g(ctxt.cc_);
                g.copy(sc);
                ghsSwitchSmall(g, *pre.ghs_btoa);
                btsStageProbe("X-ghs", g);
            }
        }
        // fused: c1 is scaled straight into Q+P (one reconstruction, one NTT pass) and switched with the one-digit
        // GHS key — no second lift, no 6-digit hybrid ModUp. FIDESLIB_BTS_SHIFT_FUSED=0 restores the two-step path.
        const bool fused = sparse_encaps && cc.compositeDegree() > 1 && cc.GetBootPrecomputation(slots).ghs_btoa &&
                           [] { const char* e = std::getenv("FIDESLIB_BTS_SHIFT_FUSED"); return e && std::atoi(e) > 0; }();  // measured +0.14 ms vs the hybrid two-step: default off
        if (fused) {
            const long double D = 1.0L / (long double)exactScaleT;
            smallIntDivideKeepLevel(cc, ctxt.c0, 3, D, false);
            smallIntDivideKeepLevel(cc, ctxt.c1, 3, D, true);  // Q + specials, NTT form, isModUp
            ctxt.NoiseFactor *= tf;
            ghsSwitchExt(ctxt, *cc.GetBootPrecomputation(slots).ghs_btoa);
        } else {
            smallIntScalarMultiply(cc, ctxt, exactScaleT, raiseCoeff);
            ctxt.NoiseFactor *= tf;
        }
        fusedSwitched = fused;
    }
    if (sparse_encaps && !aksStage0 && !fusedSwitched) {
        if (cc.compositeDegree() > 1) {
            if (exactScale > 0 && cc.GetBootPrecomputation(slots).ghs_btoa &&
                [] { const char* e = std::getenv("FIDESLIB_BTS_SHIFT_GHS"); return e && std::atoi(e) > 0; }()) {
                // default: the library hybrid switch (−0.67 ms vs GHS at t=12; FIDESLIB_BTS_SHIFT_GHS=1 opts in)
                // The hybrid switch's noise would land undivided on the already-scaled (2^22 smaller) signal; the
                // GHS switch on the small ciphertext only adds the ModDown rounding (CKKS/AksKeys.cuh).
                ghsSwitchSmall(ctxt, *cc.GetBootPrecomputation(slots).ghs_btoa);
            } else
            // COMPOSITESCALING: M-2 back to the dense key, MAIN-context standard hybrid key.
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_btoa);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;

            auto sparse_context_use = sparse_context.lock();

            auto& btoa = CKKS::GetSecretSwitchingKey(sparse_context_use, ctxt.cc_, ctxt.keyID);

            ctxt.keySwitch(btoa);
        }
    }

    btsStageProbe("MR-btoa", ctxt);
    ctxt.slots = cc.N / 2;
}

// Kept for API compatibility: the cached-bootstrap-graph machinery was removed, so this is a no-op.
int FIDESlib::CKKS::BootstrapPrecapture(FIDESlib::CKKS::Context& /*cc*/) {
    return 0;
}


// ======================================================================================================================
// Staged bootstrap (BootstrapStages.cuh): the production path of bootstrapImpl, one function per stage.
// ======================================================================================================================
namespace FIDESlib::CKKS {

bool stcFirstStagedOnly() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_BTS_STC_FIRST_STAGED_ONLY"); return e && std::atoi(e) > 0; }();
    return v;
}

static thread_local bool g_stcFirstRaiseScope = false;
bool stcFirstRaiseScope() { return g_stcFirstRaiseScope; }
void setStcFirstRaiseScope(bool on) { g_stcFirstRaiseScope = on; }

int btsStcFirstEntry(Ciphertext& ctxt, int slots) {
    auto& pre = ctxt.cc.GetBootPrecomputation(slots);
    return pre.stc_first_mode > 0 ? pre.stc_first_entry : -1;
}

BtsState btsBegin(Ciphertext& ctxt, int slots, bool prescaled, bool allowStcFirst) {
    ContextData& cc = ctxt.cc;
    BtsState st;
    assert(slots >= ctxt.slots);
    st.slots = slots;
    st.oldSlots = ctxt.slots;
    st.prescaled = prescaled;
    auto& pre = cc.GetBootPrecomputation(slots);
    st.isLT = pre.LT.slots == slots;
    double qDouble = (double)cc.prime[0].p;
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;
    const double powP = std::pow(2, cc.param.raw->p);
    const int32_t deg = std::round(std::log2(qDouble / powP));
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots)))
        throw std::runtime_error("Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) + " exceeds correctionFactor=" +
                                 std::to_string(effCorrectionFactor(cc, slots)));
    st.correction = effCorrectionFactor(cc, slots) - deg;
    const double post = std::pow(2, static_cast<double>(deg));
    double preF = 1. / post;
    st.scalar = std::llround(post);
    st.mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    if (st.mixedChain || cc.compositeDegree() > 1) {
        preF = cc.sfAtLimb(cc.L - cc.compositeDegree() * pre.raise_drop) / qDouble;
        st.scalar = 1;
    }
    st.sparseEncaps = pre.sparse_encaps;
    st.sparseB = pre.sparseB != nullptr &&
                 [] { const char* e = std::getenv("FIDESLIB_BTS_SPARSE_B_RUN"); return !(e && std::atoi(e) == 0); }();
    // StC-first pays its StC levels on the input: a deeper input (a site the runtime routes dense although the plan
    // priced it sparse) takes the shipped order instead, which lands where the sparse route would have.
    const int inLimb = (int)ctxt.getLevel() - (ctxt.NoiseLevel == 2 ? cc.compositeDegree() : 0);
    st.stcFirst = pre.stc_first_mode > 0 && !st.sparseB && allowStcFirst && !cc.stcFirstForbid && inLimb >= pre.stc_first_entry;
    st.stcFolded = pre.stc_first_mode == 2;
    if (static const bool dbg = [] { const char* e = std::getenv("FIDESLIB_BTS_DEBUG_STC"); return e && std::atoi(e) > 0; }(); dbg && pre.stc_first_mode > 0)
        std::cerr << "[stc_dbg] begin slots=" << slots << " inLimb=" << inLimb << " entry=" << pre.stc_first_entry
                  << " allow=" << allowStcFirst << " forbid=" << cc.stcFirstForbid << " cc=" << (void*)&cc
                  << " real=" << cc.getBtsRealPayload() << " cf=" << effCorrectionFactor(cc, slots)
                  << " -> stcFirst=" << st.stcFirst << "\n";
    const double k = pre.bootK > 0 ? pre.bootK : cc.GetBootK();
    st.constantEvalMult = preF * (1.0 / (k * cc.N)) * cc.getBtsPreScale();
    st.baked = pre.cts0_const;
    st.exactOnly = [] { const char* e = std::getenv("FIDESLIB_BTS_EXACT_ONLY"); return e && std::atoi(e) > 0; }();
    st.exactConst = st.exactOnly || (st.baked != 0 && std::fabs(st.baked / st.constantEvalMult - 1.0) < 1e-9);
    st.aksOn = st.exactConst && st.baked != 0 && pre.aks0 != nullptr;
    st.corFactor = (uint64_t)1 << std::llround(st.correction);
    // A real payload on the dense route enters as x + conj x = 2 Re x: one more bit of raise correction keeps the
    // EvalMod input where it was, and corFactor (taken above) keeps the output at Re x. In the StC-first order the
    // input StC already put (Re x ; Im x) in the coefficients, so EvalMod's real chain alone returns Re x: no input
    // conjugate, no correction, no StC variant.
    st.real = cc.getBtsRealPayload() && cc.N / 2 == slots && !st.isLT && !st.sparseB;
    // EvalRound cancels against the raised copy's full integer part, the real route reduces half the coefficients:
    // they cannot share a bootstrap, and EvalRound wins (single-prime CtS needs it).
    if (st.real && [] { const char* e = std::getenv("FIDESLIB_EVALROUND"); return e && std::atoi(e) > 0; }()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[bts_real] real-payload route disabled under FIDESLIB_EVALROUND (dense bootstraps run complex)\n";
        }
        st.real = false;
    }
    if (st.real && !st.stcFirst) {
        if (pre.stcRealA0.empty())
            throw std::runtime_error("real-payload bootstrap without its StC variant (FIDESLIB_BTS_REAL)");
        st.correction += 1;
    }
    if (!st.stcFirst && !st.isLT && !st.sparseB)
        requireStC(pre, "bootstrap");
    st.nCtS = (st.isLT || st.sparseB) ? 1 : (int)pre.CtS.size();
    st.nStC = (st.isLT || st.sparseB) ? 1 : (int)pre.StC.size();
    // FIDESLIB_EVALROUND=1: dense full-StC bootstraps only (the sparse routes fold between the raise and CtS)
    if (const char* e = std::getenv("FIDESLIB_EVALROUND"); e && std::atoi(e) > 0)
        st.evalRound = cc.N / 2 == st.slots && !st.isLT && !st.sparseB && !st.stcFirst && !st.aksOn;
    return st;
}

void btsStcFirstInput(Ciphertext& ctxt, const BtsState& st, bool ownCf) {
    CudaNvtxRange r(std::string{"bts::stc_first_input"});
    if (!st.stcFirst)
        return;
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    const int slots = st.slots;
    auto& pre = cc.GetBootPrecomputation(slots);
    const int inLevel0 = ctxt.getLevel(), inNL0 = ctxt.NoiseLevel;
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    // FIDESLIB_STC_ALT_BUDGET: the alternative (cheaper, deeper-entered) StC when the input has its entry level
    const bool alt = st.stcFolded && !st.isLT && pre.stc_alt_entry >= 0 && ctxt.getLevel() >= pre.stc_alt_entry;
    const int entry = alt ? pre.stc_alt_entry : pre.stc_first_entry;
    auto& firstStages = alt ? pre.StC_alt_first : pre.StC_first;
    auto& lastCache = alt ? pre.stc_alt_last : pre.stc_first_last;
    if (ctxt.getLevel() < entry)
        throw std::runtime_error("[stc_first] bootstrap input at limb index " + std::to_string(ctxt.getLevel()) +
                                 " is below the StC-first entry level " + std::to_string(entry));
    if (ctxt.getLevel() > entry)
        ctxt.dropToLevel(entry);
    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;
    if (!st.stcFolded) {
        if (st.isLT)
            EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, pre.LT_first);
        else
            EvalLTStages(ctxt, pre.StC_first, "StC1-stage-");
        return;
    }
    const int d = cc.compositeDegree();
    const int lastL = 2 * d - 1;
    const double nfIn = ctxt.NoiseFactor;
    const double targetSF = cc.sfAtLimb(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT) * d - d * pre.raise_drop);
    const int cf = (int)effCorrectionFactor(cc, slots);
    const double ratio = cc.sfAtLimb(entry) / nfIn;
    // The last stage is built at a reference correction factor and keyed by the input scale alone: a smaller CF is the
    // exact integer 2^(cfStage - cf) put on the ciphertext first (level-free, before the raise), so one entry per input
    // scale serves every site.
    // ownCf (the PCMM): the stage at the route's CF, whose plaintexts keep the precision its integer products need
    const int cfStage = ownCf ? cf : std::max(cf, kStcFirstCfRef);
    if (cfStage > cf)
        multIntScalar(ctxt, uint64_t{1} << (cfStage - cf));
    const double factor = std::ldexp(1.0, -cfStage) * (targetSF / cc.sfAtLimb(d - 1)) * ratio;
    const std::pair<int, long long> key{cfStage, std::llround(std::log2(nfIn) * 1e6)};
    // the cache cap (a cudaMemGetInfo under memory pressure) is read on a miss only
    if (st.isLT) {
        auto it = pre.lt_first_last.find(key);
        if (it == pre.lt_first_last.end()) {
            RelevelAsyncScope async(stcCacheAsync());
            const size_t kLastStageCache = lastStageCacheCap();
            if (pre.lt_first_last.size() >= kLastStageCache) {
                relevelSync();
                while (pre.lt_first_last.size() >= kLastStageCache)
                    pre.lt_first_last.erase(pre.lt_first_last.begin());
            }
            std::vector<Plaintext> v;
            for (auto& pt : pre.LT.invA)
                v.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, factor));
            relevelSync();
            it = pre.lt_first_last.emplace(key, std::move(v)).first;
        }
        EvalLinearTransformPts(ctxt, slots, pre.LT.bStep, it->second);
    } else {
        const double nfBefore = ctxt.NoiseFactor;
        // FIDESLIB_STC_ALT_NORM (diagnostic, default 1): the alternative stages read a normalized input (a rescale
        // just before them may leave the live data in the aux buffers)
        static const bool altNorm = [] { const char* e = std::getenv("FIDESLIB_STC_ALT_NORM"); return !(e && *e == '0'); }();
        if (alt && altNorm) {
            liveInV(ctxt.c0);
            liveInV(ctxt.c1);
        }
        if (alt)
            stcAltEdgeEntry(ctxt);
        probeLive("in", ctxt, alt);
        if (!firstStages.empty())
            EvalLTStages(ctxt, firstStages, "StC1-stage-");
        probeLive("first", ctxt, alt);
        if (alt)
            stcAltFence(1, 8);
        if (static int dbg = [] { const char* e = std::getenv("FIDESLIB_STC_ALT_DEBUG"); return e && *e ? std::atoi(e) : 0; }(); dbg > 0) {
            --dbg;
            std::cerr << "[stc_alt_dbg] alt=" << alt << " entry=" << entry << " nfIn=" << nfIn << " nfBefore=" << nfBefore
                      << " nfAfterFirst=" << ctxt.NoiseFactor << " level=" << ctxt.getLevel() << " NL=" << ctxt.NoiseLevel
                      << " sf(entry)=" << cc.sfAtLimb(entry) << " sf(level)=" << cc.sfAtLimb(ctxt.getLevel())
                      << " ratio=" << ratio << " actual=" << cc.sfAtLimb(ctxt.getLevel()) / ctxt.NoiseFactor << "\n";
        }
        // FIDESLIB_STC_POW2_KEY=1: input scales that differ by a power of two share one entry, built for the largest
        // exponent seen (per CF and fractional part of log2 nfIn); a smaller one is first multiplied by the exact integer
        // 2^(E - Ein), which the entry's factor undoes (the same mechanism as the 2^(cfStage - cf) above). A larger
        // exponent rebuilds the entry. Its plaintext precision is that of the block with the largest scale today.
        std::pair<int, long long> lkey = key;
        double lfactor = factor;
        int shiftK = 0;
        if (stcPow2Key()) {
            const double l2 = std::log2(nfIn);
            const long long ein = (long long)std::floor(l2 + 1e-9);
            const long long fk = std::llround((l2 - (double)ein) * 1e6);
            // FIDESLIB_STC_POW2_MAXK=k: exponents bucketed by ein / (k + 1), one entry per bucket, so an entry serves gaps
            // of at most k (each bit of gap is a bit of last-stage plaintext precision); default 63 (one bucket in practice)
            static const long long maxK = [] { const char* e = std::getenv("FIDESLIB_STC_POW2_MAXK"); return (long long)((e && *e) ? std::max(0, std::atoi(e)) : 63); }();
            const long long bucket = ein >= 0 ? ein / (maxK + 1) : -((-ein + maxK) / (maxK + 1));
            const long long sub = fk + (bucket + 1024) * (1LL << 24);
            lkey = {cfStage, sub + (1LL << 40)};
            static std::map<std::tuple<const void*, int, long long>, long long> eOf;
            auto eit = eOf.find({&lastCache, cfStage, sub});
            auto cur = lastCache.find(lkey);
            // FIDESLIB_STC_POW2_KEEP=1: an entry flushed under memory pressure is rebuilt at the largest exponent seen
            // so far, not the current one, so the next (larger) input scales hit instead of rebuilding once each
            static const bool keep = [] { const char* e = std::getenv("FIDESLIB_STC_POW2_KEEP"); return e && *e && *e != '0'; }();
            if ((cur != lastCache.end() || keep) && eit != eOf.end() && eit->second >= ein) {
                shiftK = (int)(eit->second - ein);
            } else {
                if (cur != lastCache.end()) {
                    cudaDeviceSynchronize();  // its plaintexts may still be read by the previous call's kernels
                    lastCache.erase(cur);
                }
                eOf[{&lastCache, cfStage, sub}] = ein;
            }
            lfactor = std::ldexp(factor, -shiftK);
        }
        auto it = lastCache.find(lkey);
        if (g_stcStats.on) {
            g_stcStats.first_keys.insert(std::to_string(lkey.first) + ":" + std::to_string(lkey.second) + ":" +
                                         std::to_string(pre.raise_drop));
            if (it != lastCache.end()) ++g_stcStats.first_hit; else ++g_stcStats.first_miss;
        }
        if (it == lastCache.end()) {
            RelevelAsyncScope async(stcCacheAsync());
            const size_t kLastStageCache = lastStageCacheCap();
            if (lastCache.size() >= kLastStageCache) {
                relevelSync();
                while (lastCache.size() >= kLastStageCache) {
                    lastCache.erase(lastCache.begin());
                    ++g_stcStats.first_evict;
                }
            }
            stcMissLog("first", std::to_string(lkey.first) + ":" + std::to_string(key.second) + ":" +
                                    std::to_string(pre.raise_drop) + " factor 2^" + std::to_string(std::log2(lfactor)),
                       lastCache.size());
            if (!alt)
                requireStC(pre, "stc_first_input");
            auto& s0 = alt ? pre.stc_alt_lastSrc[0] : pre.StC.back();
            BootstrapPrecomputation::LTstep o;
            o.slots = s0.slots; o.bStep = s0.bStep; o.gStep = s0.gStep; o.rotIn = s0.rotIn; o.rotOut = s0.rotOut;
            for (auto& pt : s0.A)
                o.A.push_back(relevelPlaintext(cc_, cc, pt, (lastL - pt.c0.getLevel()) / d, lfactor, relevelEachSync()));
            std::vector<BootstrapPrecomputation::LTstep> v;
            v.push_back(std::move(o));
            relevelSync();
            if (stcCacheRelease())
                for (auto& stp : v)
                    for (auto& pt : stp.A)
                        pt.c0.releaseAux();   // read-only from here on
            it = lastCache.emplace(lkey, std::move(v)).first;
        }
        if (static int dbg2 = [] { const char* e = std::getenv("FIDESLIB_STC_ALT_DEBUG"); return e && *e ? std::atoi(e) : 0; }(); dbg2 > 0) {
            --dbg2;
            std::cerr << "[stc_alt_dbg2] alt=" << alt << " cf=" << cf << " cfStage=" << cfStage << " nfIn=" << nfIn
                      << " ein-key=" << lkey.second << " shiftK=" << shiftK << " factor=" << factor << " lfactor=" << lfactor
                      << " entries=" << lastCache.size() << " lvl=" << ctxt.getLevel() << " in=" << inLevel0 << "/" << inNL0 << "\n";
        }
        if (shiftK)
            multIntScalar(ctxt, uint64_t{1} << shiftK);
        EvalLTStages(ctxt, it->second, "StC1-last-");
        probeLive("last", ctxt, alt);
        if (alt) {
            stcAltFence(2, 16);
            stcAltEdgeTail(ctxt);
        }
        // the alternative stages leave the shipped path's coefficients under a scale record larger by
        // sf(shipped entry) / sf(alternative entry) (the last stage's factor is keyed to the entry): restore it
        if (alt) {
            ctxt.NoiseFactor *= cc.sfAtLimb(pre.stc_alt_entry) / cc.sfAtLimb(pre.stc_first_entry);
            // one more stage than the shipped path flips the aux-swap parity: readers of a limb's own buffer (the
            // PCMM ring switch) must find the live data there
            liveInV(ctxt.c0);
            liveInV(ctxt.c1);
        }
    }
    if (st.stcFolded && !st.isLT) {
        probeLive("end", ctxt, alt);
        if (st.stcFolded && pre.stc_alt_entry >= 0 && (stcAltSync() & 4) && ctxt.NoiseFactor > 0 &&
            entry == pre.stc_alt_entry)
            cudaDeviceSynchronize();
    }
    if (ctxt.getLevel() != lastL || ctxt.NoiseLevel != 2)
        throw std::runtime_error("[stc_first] folded last stage left the ciphertext at limb index " +
                                 std::to_string(ctxt.getLevel()));
}

void btsStcHigh(Ciphertext& ctxt, int slots, int outLimb, double factor, int skipMask, double nfRef) {
    CudaNvtxRange r(std::string{"bts::stc_high"});
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    auto& pre = cc.GetBootPrecomputation(slots);
    if (slots != cc.N / 2 || pre.StC.empty())
        throw std::runtime_error("[stc_high] dense full-slot route only");
    const int d = cc.compositeDegree();
    std::vector<int> keep;
    for (int si = 0; si < (int)pre.StC.size(); ++si)
        if (!(skipMask >> si & 1))
            keep.push_back(si);
    const int nS = (int)keep.size();
    if (nS == 0)
        throw std::runtime_error("[stc_high] every stage skipped");
    const int entry = outLimb + nS * d;
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if (ctxt.getLevel() < entry)
        throw std::runtime_error("[stc_high] input at limb index " + std::to_string(ctxt.getLevel()) + " below entry " +
                                 std::to_string(entry));
    if (ctxt.getLevel() > entry)
        ctxt.dropToLevel(entry);
    if (nfRef > 0)
        factor *= nfRef / ctxt.NoiseFactor;
    ctxt.slots = slots;
    static std::map<std::tuple<int, int, int, long long>, std::vector<BootstrapPrecomputation::LTstep>> cache;
    // FIDESLIB_STC_HIGH_POW2=1: factors that differ by a power of two share one entry, built for the smallest exponent
    // seen (per slots / outLimb / skipMask and fractional part of log2 factor); a larger one first multiplies the
    // ciphertext by the exact integer 2^(e - e_entry) (the last stage's factor is linear in it). A smaller exponent
    // rebuilds the entry. Plaintext precision: that of the smallest factor in use today.
    long long k3 = std::llround(std::log2(factor) * 1e6);
    double lfactor = factor;
    int shiftK = 0;
    if (stcHighPow2()) {
        const double l2 = std::log2(factor);
        const long long ef = (long long)std::floor(l2 + 1e-9);
        const long long fk = std::llround((l2 - (double)ef) * 1e6);
        k3 = fk + (1LL << 40);
        static std::map<std::tuple<int, int, int, long long>, long long> eMin;
        const auto ek = std::make_tuple(slots, outLimb, skipMask, fk);
        auto eit = eMin.find(ek);
        auto cur = cache.find(std::make_tuple(slots, outLimb, skipMask, k3));
        if (cur != cache.end() && eit != eMin.end() && eit->second <= ef && ef - eit->second <= 16) {
            shiftK = (int)(ef - eit->second);
        } else {
            if (cur != cache.end()) {
                cudaDeviceSynchronize();  // its plaintexts may still be read by the previous call's kernels
                cache.erase(cur);
            }
            eMin[ek] = ef;
        }
        lfactor = std::ldexp(factor, -shiftK);
    }
    const auto key = std::make_tuple(slots, outLimb, skipMask, k3);
    auto it = cache.find(key);
    if (g_stcStats.on) {
        g_stcStats.high_keys.insert(std::to_string(outLimb) + ":" + std::to_string(skipMask) + ":" +
                                    std::to_string(std::get<3>(key)));
        if (it != cache.end()) ++g_stcStats.high_hit; else ++g_stcStats.high_miss;
    }
    if (it == cache.end()) {
        RelevelAsyncScope async(stcCacheAsync());
        if (cache.size() >= lastStageCacheCap()) {
            relevelSync();
            cache.erase(cache.begin());
            ++g_stcStats.high_evict;
        }
        std::vector<BootstrapPrecomputation::LTstep> v;
        for (int ki = 0; ki < nS; ++ki) {
            auto& s0 = pre.StC.at(keep[ki]);
            BootstrapPrecomputation::LTstep o;
            o.slots = s0.slots; o.bStep = s0.bStep; o.gStep = s0.gStep; o.rotIn = s0.rotIn; o.rotOut = s0.rotOut;
            const int lv = entry - ki * d;
            const double fac = ki == nS - 1 ? lfactor : 1.0;
            for (auto& pt : s0.A)
                o.A.push_back(relevelPlaintext(cc_, cc, pt, (lv - pt.c0.getLevel()) / d, fac, relevelEachSync()));
            v.push_back(std::move(o));
        }
        relevelSync();
        if (stcCacheRelease())
            for (auto& stp : v)
                for (auto& pt : stp.A)
                    pt.c0.releaseAux();   // read-only from here on
        stcMissLog("high", std::to_string(outLimb) + ":" + std::to_string(skipMask) + " factor 2^" +
                               std::to_string(std::log2(lfactor)), cache.size());
        it = cache.emplace(key, std::move(v)).first;
    }
    if (shiftK)
        multIntScalar(ctxt, uint64_t{1} << shiftK);
    EvalLTStages(ctxt, it->second, "StCH-stage-");
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if (ctxt.getLevel() != outLimb)
        throw std::runtime_error("[stc_high] output at limb index " + std::to_string(ctxt.getLevel()) + ", expected " +
                                 std::to_string(outLimb));
    // FIDESLIB_STC_HIGH_SYNC=1: the old trailing device fence; the output is stream-ordered for every consumer
    static const bool fence = [] { const char* e = std::getenv("FIDESLIB_STC_HIGH_SYNC"); return e && *e && *e != '0'; }();
    if (fence)
        cudaDeviceSynchronize();
}

void btsModRaise(Ciphertext& ctxt, BtsState& st) {
    CudaNvtxRange r(std::string{"bts::mod_raise"});
    ContextData& cc = ctxt.cc;
    if (st.real && !st.stcFirst) {  // an imaginary residue would leak into the real half's coefficients
        Ciphertext aux(ctxt.cc_);
        aux.conjugate(ctxt);
        ctxt.add(aux);
    }
    ModRaise(ctxt, st.slots, st.correction, st.prescaled || (st.stcFirst && st.stcFolded), st.sparseEncaps,
             st.exactConst ? st.constantEvalMult : 0.0, st.aksOn);
    st.shiftedFlow = st.exactConst && st.baked != 0 && !st.aksOn;
    if (st.exactOnly && st.baked == 0 && st.exactConst)
        ctxt.dropToLevel(ctxt.getLevel() - cc.compositeDegree());
    if (!st.exactConst)
        ctxt.multScalar(st.constantEvalMult, false);
    if (st.evalRound) {
        // the raised copy carries the EvalMod normalization c: the output's value is R / c - StC(K x - EvalMod x)
        if (st.constantEvalMult != 0)
            st.erGamma = 1.0 / st.constantEvalMult;
        st.erRaised = std::make_shared<Ciphertext>(ctxt.cc_);
        st.erRaised->copy(ctxt);
        btsStageProbe("ER-R", ctxt);
    }
}

void btsFold(Ciphertext& ctxt, const BtsState& st) {
    CudaNvtxRange r(std::string{"bts::fold"});
    ContextData& cc = ctxt.cc;
    Accumulate(ctxt, cc.GetBootPrecomputation(st.slots).accumulate_bStep, st.slots, cc.N / 2 / st.slots);
    ctxt.slots = cc.N / 2 == st.slots ? st.slots : 2 * st.slots;
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
}

static void ltStep(Ciphertext& ctxt, BootstrapPrecomputation::LTstep& step, std::vector<Plaintext>* A = nullptr) {
    std::vector<Plaintext>& diag = A ? *A : step.A;
    assert(step.slots == diag.size());
    std::vector<Plaintext*> Aptr(step.slots, nullptr);
    for (int j = 0; j < step.gStep; ++j)
        for (int i = 0; i < step.bStep; ++i)
            if (step.bStep * j + i < step.slots)
                Aptr[step.bStep * j + i] = &(diag[step.bStep * j + i]);
    const int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
    LtPtMaskScope ptMaskScope(A ? 0xFFFFFFFFu : step.ptMask);  // a substitute lacks the stage's periodic layout
    LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, step.rotOut[0]);
}

void btsCtSStage(Ciphertext& ctxt, const BtsState& st, int k) {
    CudaNvtxRange r(std::string{"bts::cts_stage"});
    ContextData& cc = ctxt.cc;
    auto& pre = cc.GetBootPrecomputation(st.slots);
    if (st.sparseB) {
        SparseBCoeffsToSlots(ctxt, st.slots, pre);
        return;
    }
    if (st.isLT) {
        EvalLinearTransform(ctxt, st.slots, false);
        return;
    }
    if (k == 0 && ctxt.NoiseLevel == 2)  // EvalCoeffsToSlots' entry rescale
        ctxt.rescale();
    auto& step = pre.CtS.at(k);
    if (k == 0 && pre.aks0) {
        LinearTransformAKS(ctxt, step, *pre.aks0);
        return;
    }
    const int cLo = pre.cts_single_lo, cHi = pre.cts_single_lo + pre.cts_single;   // single CtS stages [cLo, cHi)
    if (k > cLo && k <= cHi && ctxt.NoiseLevel == 2) {  // stage k-1 had one-prime diagonals
        RescaleOneScope one(true);
        ctxt.rescale();
    }
    const int fuseMode = [] { const char* e = std::getenv("FIDESLIB_LT_FUSED_RESCALE"); return e && *e ? std::atoi(e) : 1; }();
    LtFuseScope fuseScope(fuseMode != 3 && !(k >= cLo && k < cHi));
    ltStep(ctxt, step);
}

void btsCtSStageBatch(const std::vector<Ciphertext*>& cts, const std::vector<const BtsState*>& sts, int k) {
    CudaNvtxRange r(std::string{"bts::cts_stage_batch"});
    if (cts.size() > 4)
        throw std::invalid_argument("[cts_stage_batch] at most 4 ciphertexts");
    const BtsState& st = *sts.at(0);
    ContextData& cc = cts.at(0)->cc;
    auto& pre = cc.GetBootPrecomputation(st.slots);
    bool same = cts.size() > 1;
    for (size_t i = 1; i < cts.size(); ++i)
        same = same && sts[i]->slots == st.slots && cts[i]->getLevel() == cts[0]->getLevel() &&
               cts[i]->NoiseLevel == cts[0]->NoiseLevel;
    if (!same || st.sparseB || st.isLT || (k == 0 && pre.aks0)) {
        for (size_t i = 0; i < cts.size(); ++i)
            btsCtSStage(*cts[i], *sts[i], k);
        return;
    }
    const int cLo = pre.cts_single_lo, cHi = pre.cts_single_lo + pre.cts_single;
    for (auto* c : cts) {
        if (k == 0 && c->NoiseLevel == 2)
            c->rescale();
        if (k > cLo && k <= cHi && c->NoiseLevel == 2) {
            RescaleOneScope one(true);
            c->rescale();
        }
    }
    const int fuseMode = [] { const char* e = std::getenv("FIDESLIB_LT_FUSED_RESCALE"); return e && *e ? std::atoi(e) : 1; }();
    LtFuseScope fuseScope(fuseMode != 3 && !(k >= cLo && k < cHi));
    auto& step = pre.CtS.at(k);
    std::vector<Plaintext*> Aptr(step.slots, nullptr);
    for (int j = 0; j < step.gStep; ++j)
        for (int i = 0; i < step.bStep; ++i)
            if (step.bStep * j + i < step.slots)
                Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
    const int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
    LtPtMaskScope ptMaskScope(step.ptMask);
    LinearTransformBatch(cts, step.slots, step.bStep, Aptr, stride, step.rotOut[0]);
}

void btsEvalMod(Ciphertext& ctxt, const BtsState& st) {
    CudaNvtxRange r(std::string{"bts::eval_mod"});
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    {   // the last CtS stage had one-prime diagonals (FIDESLIB_CTS_SINGLE window reaching the end): drop one prime
        auto& pre = cc.GetBootPrecomputation(st.slots);
        if (pre.cts_single && pre.cts_single_lo + pre.cts_single == st.nCtS && ctxt.NoiseLevel == 2) {
            RescaleOneScope one(true);
            ctxt.rescale();
        }
    }
    Ciphertext aux(cc_);
    if (st.real) {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionReal(ctxt, cc.GetEvalKey(ctxt.keyID), st.scalar);
    } else if (cc.N / 2 == st.slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
            ctxt.rescale();
            ctxtEncI.rescale();
        }
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), st.scalar, st.evalRound);
    } else if (st.sparseB) {
        auto& sb = *cc.GetBootPrecomputation(st.slots).sparseB;
        approxModReductionSparse(ctxt, st.scalar, sb.cheb, sb.daIts);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (auto& rp = cc.GetBootPrecomputation(st.slots); !rp.evalModCheb.empty())
            approxModReductionSparse(ctxt, st.scalar, rp.evalModCheb, rp.evalModDaIts);
        else
            approxModReductionSparse(ctxt, st.scalar);
    }
    if (ctxt.NoiseLevel == 2 && !(st.stcFirst && cc.N / 2 == st.slots))
        ctxt.rescale();
}

void btsStcFirstOutput(Ciphertext& ctxt, const BtsState& st) {
    CudaNvtxRange r(std::string{"bts::stc_first_output"});
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    auto& pre = cc.GetBootPrecomputation(st.slots);
    const uint64_t cfAll = st.corFactor << pre.stc_first_deg;
    if (cfAll != 1)
        multIntScalar(ctxt, cfAll);
    if (cc.N / 2 != st.slots) {
        const int lv = ctxt.getLevel();
        auto it = pre.stc_first_mask_at.find(lv);
        if (it == pre.stc_first_mask_at.end()) {
            const int sh = (lv - pre.stc_first_mask->c0.getLevel()) / cc.compositeDegree();
            it = pre.stc_first_mask_at.emplace(lv, relevelPlaintext(cc_, cc, *pre.stc_first_mask, sh, 1.0)).first;
        }
        ctxt.multPt(it->second, false);
        Ciphertext aux(cc_);
        aux.rotate(ctxt, st.slots);
        ctxt.add(aux);
    }
    if (st.mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    ctxt.slots = st.oldSlots;
}

void btsStCEnter(Ciphertext& ctxt, const BtsState& st) {
    ContextData& cc = ctxt.cc;
    auto& pre = cc.GetBootPrecomputation(st.slots);
    if (!st.isLT && !st.sparseB && !pre.StC.empty()) {
        const int stcL = pre.StC.at(0).A.at(0).c0.getLevel();
        if (ctxt.getLevel() > stcL) {
            if (ctxt.NoiseLevel == 2)
                ctxt.rescale();
            if (ctxt.getLevel() > stcL)
                ctxt.dropToLevel(stcL);
        }
    }
}

void btsStCStage(Ciphertext& ctxt, const BtsState& st, int k) {
    CudaNvtxRange r(std::string{"bts::stc_stage"});
    ContextData& cc = ctxt.cc;
    auto& pre = cc.GetBootPrecomputation(st.slots);
    if (st.sparseB) {
        SparseBSlotsToCoeffs(ctxt, st.slots, pre);
        return;
    }
    if (st.isLT) {
        EvalLinearTransform(ctxt, st.slots, true);
        return;
    }
    if (k == 0 && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if (k >= 1 && k <= pre.stc_single && ctxt.NoiseLevel == 2) {  // stage k-1 had one-prime diagonals
        RescaleOneScope one(true);
        ctxt.rescale();
    }
    const int fuseModeS = [] { const char* e = std::getenv("FIDESLIB_LT_FUSED_RESCALE"); return e && *e ? std::atoi(e) : 1; }();
    LtFuseScope fuseScope(k != (int)pre.StC.size() - 1 && k >= pre.stc_single && fuseModeS != 2);
    ltStep(ctxt, pre.StC.at(k), k == 0 && st.real ? &pre.stcRealA0 : nullptr);
}

// FIDESLIB_FINISH_GRAPH=1: btsFinishStcFirst (the PCMM / CCMM output refresh) replayed from a CUDA graph per shape: the
// first sighting runs eagerly (it warms the tables and scratch), the second is captured, later ones replay (input copied
// into the graph's ciphertext, legacy-stream fences on both sides, output copied back). The key covers the input's
// shape and scale, the route scope (raise drop, real payload, CF override) and every plain field of the state.
static bool finishGraphOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_FINISH_GRAPH"); return e && std::atoi(e) > 0; }();
    return v;
}

void btsFinishStcFirst(Ciphertext& ctxt, BtsState& st, uint64_t mul) {
    auto body = [&st, mul](Ciphertext& c) {
        btsModRaise(c, st);
        btsFold(c, st);
        for (int k = 0; k < st.nCtS; ++k)
            btsCtSStage(c, st, k);
        btsEvalMod(c, st);
        btsStcFirstOutput(c, st);
        if (mul > 1)
            multIntScalar(c, mul);
    };
    ContextData& cc = ctxt.cc;
    if (finishGraphOn() && !FIDESlib::g_cap.active && cc.GPUid.size() == 1 && st.stcFirst && !st.evalRound) {
        static auto* graphs = new std::map<std::string, BtsGraph>;   // never destroyed: graph memory is never released
        std::string k;
        auto put = [&k](const void* p, size_t n) { k.append(static_cast<const char*>(p), n); };
        auto pi = [&put](long long v) { put(&v, sizeof v); };
        auto pd = [&pi](double v) { pi((long long)bitsOf(v)); };
        pi(ctxt.c0.getLevel()); pi(ctxt.c1.getLevel()); pi(ctxt.NoiseLevel); pi(ctxt.slots); pd(ctxt.NoiseFactor);
        pi((int)ctxt.c0.isModUp() + 2 * (int)ctxt.c1.isModUp()); pi(cc.btsRaiseDropEffective());
        pi((int)cc.getBtsRealPayload()); pi(cc.getCorrectionFactorOverride()); pi((int)g_rescaleOnePrime);
        pd(cc.getBtsPreScale()); pi((long long)mul);
        pi(st.slots); pi(st.oldSlots); pi(st.prescaled); pi(st.isLT); pi(st.mixedChain); pi(st.sparseEncaps);
        pi(st.sparseB); pi(st.stcFirst); pi(st.stcFolded); pi(st.exactOnly); pi(st.exactConst); pi(st.aksOn);
        pi(st.shiftedFlow); pi(st.real); pi(st.correction); pi((long long)st.scalar); pi((long long)st.corFactor);
        pd(st.constantEvalMult); pd(st.baked); pi(st.nCtS); pi(st.nStC);
        k += ctxt.keyID;
        BtsGraph& e = (*graphs)[k];
        if (!e.failed) {
            if (!e.exec && e.seen++ >= 1 && !btsGraphCaptureFn(e, ctxt, body))
                e.failed = true;
            if (e.exec) {
                btsGraphReplay(e, ctxt);
                return;
            }
        }
    }
    body(ctxt);
}

void btsFinish(Ciphertext& ctxt, const BtsState& st) {
    CudaNvtxRange r(std::string{"bts::finish"});
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    if (cc.N / 2 != st.slots && !st.sparseB) {
        Ciphertext aux(cc_);
        aux.rotate(ctxt, st.slots);
        ctxt.add(aux);
    }
    if (st.real) {
        Ciphertext aux(cc_);
        aux.conjugate(ctxt);
        ctxt.add(aux);
    }
    if (st.evalRound && st.erRaised) {  // output = gamma * raised - StC(K x - EvalMod(x))
        btsStageProbe("ER-S", ctxt);
        Ciphertext o(cc_);
        o.copy(*st.erRaised);
        // gamma = 2^k * f, f in [1,2): 2^k as an exact integer product, f through the NF. Folding all of gamma into
        // the NF makes the subtraction's scale adjustment encode a ~2^63 constant, which it gets wrong.
        const int k = std::min(62, std::max(0, (int)std::floor(std::log2(st.erGamma))));
        o.NoiseFactor /= st.erGamma / std::ldexp(1.0, k);
        if (k > 0)
            multIntScalar(o, 1ull << k);
        o.sub(ctxt);  // the raised copy is adjusted down to the StC output's level and scale
        ctxt.copy(o);
        btsStageProbe("ER-O", ctxt);
    }
    if (st.corFactor != 1)
        multIntScalar(ctxt, st.corFactor);
    if (st.mixedChain && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    ctxt.slots = st.oldSlots;
}

void BootstrapStaged(Ciphertext& ctxt, int slots, bool prescaled) {
    BtsState st = btsBegin(ctxt, slots, prescaled, true);
    btsStcFirstInput(ctxt, st);
    btsModRaise(ctxt, st);
    btsFold(ctxt, st);
    for (int k = 0; k < st.nCtS; ++k)
        btsCtSStage(ctxt, st, k);
    btsEvalMod(ctxt, st);
    if (st.stcFirst) {
        btsStcFirstOutput(ctxt, st);
        return;
    }
    btsStCEnter(ctxt, st);
    for (int k = 0; k < st.nStC; ++k)
        btsStCStage(ctxt, st, k);
    btsFinish(ctxt, st);
}

}  // namespace FIDESlib::CKKS
