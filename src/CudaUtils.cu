#include <functional>
#include <vector>
#include <map>
#include <unordered_map>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <array>
//
// Created by carlosad on 25/03/24.
//

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <sys/mman.h>
#include "CudaUtils.cuh"

#include <iostream>

//#include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

namespace FIDESlib {

int getNumDevices() {
    int d;
    cudaGetDeviceCount(&d);
    return d;
};

void CudaHostSync() {
    cudaDeviceSynchronize();
}

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
    cudaGraph_t graph;
    if constexpr (capture) {
        cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
        CudaCheckErrorModNoSync;
    }
    run();
    if constexpr (capture) {
        cudaStreamEndCapture(s.ptr(), &graph);
        if (!exec) {
            cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
            //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
            CudaCheckErrorModNoSync;
        } else {
            if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
                CudaCheckErrorModNoSync;
                // only instantiate a new graph if update fails
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
                //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
                CudaCheckErrorModNoSync;
            }
        }
        cudaGraphDestroy(graph);
        cudaGraphLaunch(exec, s.ptr());
    }
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
    void Stream::wait(const Event &ev) const {
        cudaStreamWaitEvent(ptr, ev.ptr());
    }
*/
void Stream::capture_begin() {
    CudaCheckErrorMod;
    std::cout << "Hello capture" << std::endl;
    cudaStreamCaptureStatus cap;
    cudaStreamIsCapturing(ptr(), &cap);

    CudaCheckErrorMod;
    if (cap == cudaStreamCaptureStatusNone) {
        std::cout << "None" << std::endl;
        cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
    } else if (cap == cudaStreamCaptureStatusActive) {
        std::cout << "Fail: activo" << std::endl;
    } else if (cap == cudaStreamCaptureStatusInvalidated) {
        std::cout << "Fail: invalidado" << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    CudaCheckErrorMod;
}

void Stream::capture_end() {
    cudaGraph_t graph;
    cudaStreamEndCapture(ptr(), &graph);
    CudaCheckErrorMod;
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    CudaCheckErrorMod;
    cudaGraphDestroy(graph);
    CudaCheckErrorMod;
    cudaGraphLaunch(graphExec, 0);
    cudaGraphExecDestroy(graphExec);

    cudaStreamSynchronize(0);
}

void Stream::record(bool external) {
    if (ptr_ == 0)
        return;
    CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
    //cudaEventDestroy(ev);
    //cudaEventCreate(&ev, cudaEventDisableTiming);
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    if (g_cap.active && joinGen != g_cap.gen)
        joinCapture();
    cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
    updated = true;
    capGen = g_cap.active ? g_cap.gen : 0;
#endif
}

void Stream::wait_recorded(const Stream& s) {
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
#if !DISABLE_STREAMS
    assert(s.updated);
    if (g_cap.active) {
        if (joinGen != g_cap.gen)
            joinCapture();
        if (s.joinGen != g_cap.gen || s.capGen != g_cap.gen)
            return;   // recorded before the capture: the replay's entry barrier covers it
    } else if (s.capGen != 0) {
        const_cast<Stream&>(s).record();   // last recorded inside a finished capture
    }
    cudaStreamWaitEvent(ptr_, s.ev);
    this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {

#if !DISABLE_STREAMS
    //CudaCheckErrorModNoSync;
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
    assert(ptr_ != nullptr);
    assert(s.ptr_ != nullptr);
    assert(ev != nullptr);
    if (g_cap.active) {
        if (joinGen != g_cap.gen)
            joinCapture();
        if (s.joinGen != g_cap.gen)
            return;   // s did nothing in this capture: its work predates it (entry barrier)
        if (s.updated && s.capGen != g_cap.gen)
            return;
    } else if (s.capGen != 0) {
        s.updated = false;   // ev was last recorded inside a finished capture: re-record eagerly
    }
    if (!s.updated) {
        assert(!external);  // Has to be recorded in the origin graph
        CudaCheckErrorModNoSync;
        cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
        s.updated = true;
        s.capGen = g_cap.active ? g_cap.gen : 0;
        CudaCheckErrorModNoSync;
    }
    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
    this->updated = false;
#endif
    CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {

#if !DISABLE_STREAMS
    //CudaCheckErrorModNoSync;
    if (s == 0 || ptr_ == 0)
        return;
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    if (g_cap.active) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &st);
        if (st != cudaStreamCaptureStatusActive)
            return;   // s is outside the capture: nothing of it to order against
        if (joinGen != g_cap.gen)
            joinCapture();
    }
    CudaCheckErrorModNoSync;
    cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
    updated = false;
    capGen = g_cap.active ? g_cap.gen : 0;
    CudaCheckErrorModNoSync;

    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, ev, cudaEventWaitDefault);
#endif
    CudaCheckErrorModNoSync;
}

int low = -1;
int high = -1;
bool errcheckOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_ERRCHECK"); return e && std::atoi(e) > 0; }();
    return v;
}
bool ompRegionsOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_OMP_REGIONS"); return e && std::atoi(e) > 0; }();
    return v;
}
// ---- FIDESLIB_BTS_GRAPH capture support (see CudaUtils.cuh) ---------------------------------------------------
CaptureState g_cap;
bool captureActiveNow() { return g_cap.active; }

namespace {
std::map<const char*, size_t> g_ownedChunks;   // every chunk handed to graphs: start -> bytes
bool ownedFast(const void* p) {
    if (g_ownedChunks.empty()) return false;
    const char* c = static_cast<const char*>(p);
    auto it = g_ownedChunks.upper_bound(c);
    if (it == g_ownedChunks.begin()) return false;
    --it;
    return c < it->first + it->second;
}
}  // namespace

struct CaptureRegion {
    size_t chunkBytes;
    std::vector<std::pair<char*, size_t>> chunks;
    size_t cur = 0, off = 0;
    std::vector<std::pair<char*, size_t>> pinned;   // H2D snapshots read by the graph's memcpy nodes
    size_t poff = 0;
    explicit CaptureRegion(size_t cb) : chunkBytes(cb) {}
    void* getPinned(size_t bytes) {
        bytes = (bytes + 63) & ~size_t(63);
        if (pinned.empty() || poff + bytes > pinned.back().second) {
            char* h = nullptr;
            const size_t n = std::max<size_t>(256 << 10, bytes);
            if (cudaMallocHost(&h, n) != cudaSuccess || !h) {
                cudaGetLastError();
                captureInvalidate("pinned snapshot: cudaMallocHost failed");
                return nullptr;
            }
            pinned.emplace_back(h, n);
            poff = 0;
        }
        void* p = pinned.back().first + poff;
        poff += bytes;
        return p;
    }
    void reset() { cur = 0; off = 0; }
    bool owns(const void* p) const {
        const char* c = static_cast<const char*>(p);
        for (auto& [b, n] : chunks)
            if (c >= b && c < b + n) return true;
        return false;
    }
    void* get(size_t bytes) {
        bytes = (bytes + 511) & ~size_t(511);
        while (cur < chunks.size() && off + bytes > chunks[cur].second) { ++cur; off = 0; }
        if (cur == chunks.size()) {
            char* b = nullptr;
            const size_t n = std::max(chunkBytes, bytes);
            if ((cudaMalloc)(&b, n) != cudaSuccess || !b) {
                cudaGetLastError();
                captureInvalidate("graph memory: cudaMalloc failed");
                return nullptr;
            }
            chunks.emplace_back(b, n);
            g_ownedChunks[b] = n;
            off = 0;
        }
        void* p = chunks[cur].first + off;
        off += bytes;
        return p;
    }
};

namespace {
// Graph memory. Both arenas are shared by every graph and never released: replays are serialized and fenced, so a
// buffer one graph uses as a temporary is free for any other graph. g_arena: big buffers, reused (a buffer freed in
// the capture is handed out again after the freeing op, through a captured event; across captures with no event).
// g_small: tables, never reused (their snapshot nodes rewrite them on every replay).
CaptureRegion g_arena(256ull << 20);
CaptureRegion g_small(4ull << 20);
std::map<size_t, std::vector<std::pair<void*, cudaEvent_t>>> g_arenaFree;
std::unordered_set<cudaEvent_t> g_capEvSet;   // events recorded inside the current capture
std::vector<cudaEvent_t> g_capEvents;
size_t g_capEventNext = 0;
std::vector<cudaStream_t> g_capJoined;
std::vector<std::pair<cudaStream_t, cudaEvent_t>> g_capDeferred;
uint32_t g_capGenCounter = 0;
std::map<void*, size_t> g_capMallocs;   // cudaMallocAsync'd inside the current capture (served by the arena)
std::string g_capWhy;
cudaEvent_t nextCapEvent() {
    if (g_capEventNext == g_capEvents.size()) {
        cudaEvent_t e = nullptr;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        g_capEvents.push_back(e);
    }
    return g_capEvents[g_capEventNext++];
}
size_t roundAlloc(size_t bytes) {
    if (bytes < 64 * 1024) {
        size_t p = 1024;
        while (p < bytes) p *= 2;
        return p;
    }
    return bytes;
}
void* captureMalloc(size_t bytes, cudaStream_t stream) {
    bytes = roundAlloc(bytes);
    if (bytes < 64 * 1024)
        return g_small.get(bytes);
    if (stream) captureJoin(stream);
    auto& fl = g_arenaFree[bytes];
    if (!fl.empty()) {
        auto [p, e] = fl.back();
        fl.pop_back();
        if (e) {
            if (stream) (cudaStreamWaitEvent)(stream, e, 0);   // the previous user's last captured op
            else captureInvalidate("arena reuse without a stream");
        }
        return p;
    }
    return g_arena.get(bytes);
}
void captureFree(void* ptr, size_t bytes, cudaStream_t stream) {
    bytes = roundAlloc(bytes);
    if (g_small.owns(ptr)) return;   // tables: never reused
    if (g_arena.owns(ptr)) {
        if (!stream) return;   // no ordering information: not reused
        captureJoin(stream);
        cudaEvent_t e = nextCapEvent();
        (cudaEventRecord)(e, stream);
        g_capEvSet.insert(e);
        g_arenaFree[bytes].emplace_back(ptr, e);
        return;
    }
    if (ownedFast(ptr)) return;
    captureInvalidate("memory allocated before the capture freed inside it");   // leaked: the replay would touch it
}
}  // namespace

void captureInvalidate(const char* why) {
    if (!g_cap.invalid) g_cap.why = why;
    g_cap.invalid = true;
}

void captureJoin(cudaStream_t s) {
    if (!g_cap.active || !s || s == g_cap.origin) return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &st);
    if (st == cudaStreamCaptureStatusNone)
        (cudaStreamWaitEvent)(s, g_cap.originEv, 0);   // the real call: the hook would join again
    else if (st == cudaStreamCaptureStatusInvalidated)
        captureInvalidate("a joined stream's capture was invalidated");
    for (auto j : g_capJoined)
        if (j == s) return;
    g_capJoined.push_back(s);
}

CaptureRegion* captureRegionNew() { return new CaptureRegion(1ull << 20); }   // per graph: pinned snapshots only
void captureRegionFree(CaptureRegion* r) {
    if (!r) return;
    for (auto& [h, n] : r->pinned)
        cudaFreeHost(h);
    delete r;
}
bool captureEverStarted() { return g_capGenCounter > 0; }
bool captureOwned(const void* p) { return ownedFast(p); }

bool captureBegin(cudaStream_t origin, CaptureRegion* small) {
    if (g_cap.active || !origin) return false;
    if (!g_cap.originEv) cudaEventCreateWithFlags(&g_cap.originEv, cudaEventDisableTiming);
    g_capJoined.clear();
    g_capMallocs.clear();
    g_capEvSet.clear();
    g_capEventNext = 0;
    cudaGetLastError();
    if (cudaStreamBeginCapture(origin, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    g_cap.active = true;
    g_cap.invalid = false;
    g_cap.why = nullptr;
    g_cap.gen = ++g_capGenCounter;
    g_cap.origin = origin;
    g_cap.small = small;
    (cudaEventRecord)(g_cap.originEv, origin);
    g_capEvSet.insert(g_cap.originEv);
    return true;
}

cudaGraph_t captureEnd() {
    for (auto s : g_capJoined) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &st);
        if (st == cudaStreamCaptureStatusActive) {
            cudaEvent_t e = nextCapEvent();
            (cudaEventRecord)(e, s);
            (cudaStreamWaitEvent)(g_cap.origin, e, 0);
        } else if (st == cudaStreamCaptureStatusInvalidated) {
            captureInvalidate("a joined stream's capture was invalidated");
        }
    }
    cudaGraph_t g = nullptr;
    const cudaError_t err = cudaStreamEndCapture(g_cap.origin, &g);
    g_cap.active = false;
    g_cap.small = nullptr;
    for (auto& [st, ev] : g_capDeferred) {
        if (ev) cudaEventDestroy(ev);
        if (st) cudaStreamDestroy(st);
    }
    g_capDeferred.clear();
    g_capJoined.clear();
    g_capMallocs.clear();
    g_capEvSet.clear();
    for (auto& [b, v] : g_arenaFree)   // later captures need no ordering against this one (replays are fenced)
        for (auto& pe : v) pe.second = nullptr;
    if (err != cudaSuccess || g_cap.invalid || !g) {
        if (!g_cap.why) g_cap.why = err != cudaSuccess ? cudaGetErrorString(err) : "no graph";
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
        return nullptr;
    }
    return g;
}

namespace {
void invalidateAt(const char* f, int l, const char* what) {
    if (!g_cap.invalid) {
        g_capWhy = std::string(what) + " @ " + (f ? f : "?") + ":" + std::to_string(l);
        g_cap.why = g_capWhy.c_str();
    }
    g_cap.invalid = true;
}
bool isHostPtr(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();
        return true;
    }
    return a.type == cudaMemoryTypeUnregistered || a.type == cudaMemoryTypeHost;
}
}  // namespace

cudaError_t captureBlocked(const char* f, int l, const char* what) {
    invalidateAt(f, l, what);
    return cudaSuccess;
}

cudaError_t captureStreamSync(const char* f, int l, cudaStream_t s) {
    invalidateAt(f, l, "cudaStreamSynchronize");
    (void)s;
    return cudaSuccess;
}

cudaError_t captureEventRecord(const char* f, int l, cudaEvent_t e, cudaStream_t s, unsigned flags) {
    if (!s) return captureBlocked(f, l, "event record on the legacy stream");
    captureJoin(s);
    g_capEvSet.insert(e);
    return (cudaEventRecordWithFlags)(e, s, flags);
}

cudaError_t captureStreamWaitEvent(const char* f, int l, cudaStream_t s, cudaEvent_t e, unsigned flags) {
    if (!s) return captureBlocked(f, l, "event wait on the legacy stream");
    if (!g_capEvSet.count(e)) return cudaSuccess;   // recorded before the capture: the entry fence orders it
    captureJoin(s);
    return (cudaStreamWaitEvent)(s, e, flags);
}

cudaError_t captureMemcpyAsync(const char* f, int l, void* dst, const void* src, size_t n, cudaMemcpyKind k,
                               cudaStream_t s) {
    if (n == 0) return cudaSuccess;
    if (!s) return captureBlocked(f, l, "cudaMemcpyAsync on the legacy stream");
    bool srcHost = k == cudaMemcpyHostToDevice || k == cudaMemcpyHostToHost;
    bool dstHost = k == cudaMemcpyDeviceToHost || k == cudaMemcpyHostToHost;
    if (k == cudaMemcpyDefault) {
        srcHost = isHostPtr(src);
        dstHost = isHostPtr(dst);
    }
    if (dstHost) return captureBlocked(f, l, "device-to-host copy");
    captureJoin(s);
    if (!srcHost) return (cudaMemcpyAsync)(dst, src, n, cudaMemcpyDeviceToDevice, s);
    void* snap = g_cap.small ? g_cap.small->getPinned(n) : nullptr;
    if (!snap) return captureBlocked(f, l, "no pinned snapshot");
    std::memcpy(snap, src, n);
    return (cudaMemcpyAsync)(dst, snap, n, cudaMemcpyHostToDevice, s);
}

cudaError_t captureMallocAsyncHook(void** p, size_t n, cudaStream_t s) {
    if (!s) {
        invalidateAt(nullptr, 0, "cudaMallocAsync on the legacy stream");
    }
    void* q = captureMalloc(n, s);
    *p = q;
    if (q) g_capMallocs[q] = roundAlloc(n);
    return q ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t captureFreeAsyncHook(const char* f, int l, void* p, cudaStream_t s) {
    if (!p) return cudaSuccess;
    auto it = g_capMallocs.find(p);
    if (it == g_capMallocs.end()) return captureBlocked(f, l, "cudaFreeAsync of memory older than the capture");
    captureFree(p, it->second, s);
    g_capMallocs.erase(it);
    return cudaSuccess;
}

static void destroyStreamAndEvent(cudaStream_t p, cudaEvent_t e, bool /*shared*/) {
    if (g_cap.active) {   // the stream may be part of the capture: destroyed once it ends
        if (p || e) g_capDeferred.emplace_back(p, e);
        return;
    }
    if (e) cudaEventDestroy(e);
    if (p) cudaStreamDestroy(p);
}

void Stream::init(int priority) {
    if (ptr_) {
        //free[ptr]++;
        destroyStreamAndEvent(ptr_, ev, false);
        ptr_ = nullptr;
        ev = nullptr;
    }
    capGen = joinGen = 0;

#if !DISABLE_STREAMS
    if (high == -1) {
        cudaDeviceGetStreamPriorityRange(&low, &high);
    }

    int prio = low + priority * ((high - low - 1)) / 100;
    cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, prio);
    //cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);

    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);   // (was created twice: one event per stream leaked)
#else
    ptr = nullptr;
    ev = nullptr;
#endif
    //free[ptr] = 0;
}

void Stream::initDefault() {
    ptr_ = 0;
    ev = nullptr;
    updated = true;
}

//std::map<void *, int> free;

Stream::~Stream() {
    destroyStreamAndEvent(ptr_, ev, false);
    ptr_ = nullptr;
    ev = nullptr;
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept
    : ptr_(s.ptr_), ev(s.ev), updated(s.updated), capGen(s.capGen), joinGen(s.joinGen) {
    s.ptr_ = nullptr;
    s.ev = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
    if (GPUprop.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            GPUprop.emplace_back();
            cudaGetDeviceProperties(&GPUprop.back(), i);

            std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
                      << ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l
                      << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
                      << ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor
                      << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
                      << " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

                      << "-bit" << std::endl;
        }
    }
}

/* Deliberately never destroyed: the static context cache (another TU) calls GPUfree from its own
 * destructor, and static destruction order across TUs is unspecified, so the pool must outlive it.
 * Cost is one 8-element pool per process, reclaimed by the OS at exit. */
std::mutex* const mempool_lock = new std::mutex[8];

std::map<int, std::vector<void*>>* const size_to_memory = new std::map<int, std::vector<void*>>[8];

FIDESlib::Stream* const s = new FIDESlib::Stream[8];

/* Slab bookkeeping for the pooled path: a pooled size class is served from slabs (cudaMallocAsync'd chunks carved
 * into equal blocks), and a block freed goes back to its class's free list. Without this a slab was never released,
 * so a class that stopped being used (a phase of the computation, a bootstrap route, a cache length) kept its memory
 * for the life of the process. A slab whose blocks are all on the free list can be handed back to the CUDA
 * stream-ordered pool, which every allocation draws from (GPUpoolTrim; also done before giving up on an allocation).
 * Guarded by mempool_lock[id], like size_to_memory. */
struct PoolSlab {
    int block;
    int nblocks;
    int nfree;
};
std::map<char*, PoolSlab>* const slabs = new std::map<char*, PoolSlab>[8];

static PoolSlab* slab_of(int id, void* ptr) {
    auto& m = slabs[id];
    auto it = m.upper_bound((char*)ptr);
    if (it == m.begin())
        return nullptr;
    --it;
    if ((char*)ptr >= it->first + (size_t)it->second.block * it->second.nblocks)
        return nullptr;
    return &it->second;
}

/* Release the idle slabs of device `id` (mempool_lock[id] held). Returns the bytes released. */
static size_t pool_trim_locked(int id) {
    size_t freed = 0;
    auto& m = slabs[id];
    for (auto it = m.begin(); it != m.end();) {
        PoolSlab& sl = it->second;
        if (sl.nfree != sl.nblocks) {
            ++it;
            continue;
        }
        char* lo = it->first;
        char* hi = lo + (size_t)sl.block * sl.nblocks;
        auto& fl = size_to_memory[id][sl.block];
        fl.erase(std::remove_if(fl.begin(), fl.end(), [&](void* p) { return (char*)p >= lo && (char*)p < hi; }),
                 fl.end());
        // ordered after every free of its blocks: GPUfree makes the pool stream wait on the freeing stream
        cudaFreeAsync(lo, s[id].ptr());
        freed += (size_t)sl.block * sl.nblocks;
        it = m.erase(it);
    }
    return freed;
}

static size_t run_reclaim_hook(int id);

/* cudaMallocAsync on `stream`; on out-of-memory, release the idle slabs of device `id` and retry once
 * (`locked`: mempool_lock[id] is already held). */
static cudaError_t malloc_or_trim(int id, void** ptr, size_t bytes, cudaStream_t stream, bool locked) {
    cudaError_t e = cudaMallocAsync(ptr, bytes, stream);
    if (e != cudaErrorMemoryAllocation)
        return e;
    cudaGetLastError();
    size_t freed;
    if (locked) {
        freed = pool_trim_locked(id);
    } else {
        // sync first: a pool-idle block may still be read by a kernel on another stream
        cudaDeviceSynchronize();
        // idle cached objects (the context's polynomial recycler) give their blocks back first, so their slabs can go
        if (size_t n = run_reclaim_hook(id))
            std::cerr << "[gpu_pool] out of memory for " << bytes << " B: reclaimed " << n << " cached polynomials"
                      << std::endl;
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        freed = pool_trim_locked(id);
    }
    if (freed == 0)
        return cudaMallocAsync(ptr, bytes, stream);
    // the released slabs become reusable by other streams once their frees have run
    cudaStreamSynchronize(s[id].ptr());
    std::cerr << "[gpu_pool] out of memory for " << bytes << " B: released " << (freed >> 20) << " MiB of idle slabs"
              << std::endl;
    return cudaMallocAsync(ptr, bytes, stream);
}

static std::function<size_t()>* const reclaim_hook = new std::function<size_t()>[8];

void GPUpoolSetReclaimHook(int id, std::function<size_t()> hook) {
    std::lock_guard<std::mutex> guard(mempool_lock[id]);
    reclaim_hook[id] = std::move(hook);
}

/* Run device `id`'s reclaim hook (mempool_lock[id] NOT held: the hook frees blocks through GPUfree). */
static size_t run_reclaim_hook(int id) {
    std::function<size_t()> h;
    {
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        h = reclaim_hook[id];
    }
    return h ? h() : 0;
}

size_t GPUpoolTrim(int id) {
    std::lock_guard<std::mutex> guard(mempool_lock[id]);
    if (s[id].ptr() == nullptr)
        return 0;
    size_t freed = pool_trim_locked(id);
    // FIDESLIB_POOL_TRIM_NOSYNC=1: no host waits here -- the slab frees stay ordered on the pool stream (which waits on
    // every freeing stream) and what they release reaches the device at the next trim instead of this one
    static const bool nosync = [] { const char* e = std::getenv("FIDESLIB_POOL_TRIM_NOSYNC"); return e && std::atoi(e) > 0; }();
    if (!nosync) {
        cudaStreamSynchronize(s[id].ptr());
        // the stream-ordered pool never releases (threshold UINT64_MAX): trim it so plain cudaMalloc can use it
        cudaDeviceSynchronize();
    }
    cudaMemPool_t mp;
    if (cudaDeviceGetDefaultMemPool(&mp, id) == cudaSuccess) cudaMemPoolTrimTo(mp, 0);
    return freed;
}

std::vector<std::array<size_t, 4>> GPUpoolBySize(int id) {
    std::lock_guard<std::mutex> guard(mempool_lock[id]);
    std::map<int, std::array<size_t, 4>> by;
    for (auto& [base, sl] : slabs[id]) {
        auto& r = by[sl.block];
        r[0] = sl.block;
        r[1] += (size_t)sl.block * sl.nblocks;
        r[2] += (size_t)sl.block * (sl.nblocks - sl.nfree);
        r[3] += 1;
    }
    std::vector<std::array<size_t, 4>> out;
    for (auto& [b, r] : by)
        out.push_back(r);
    return out;
}

/* FIDESLIB_POOL_TRACE=1: every live pooled block keeps the backtrace of its allocation, so the live bytes can be
 * attributed to call sites (GPUpoolTrace). A diagnostic: a backtrace per allocation is not free. */
static bool pool_trace_on() {
    static const bool on = [] {
        const char* e = std::getenv("FIDESLIB_POOL_TRACE");
        return e && *e && *e != '0';
    }();
    return on;
}
struct PoolTraceRec {
    int bytes;
    int n;
    void* fr[40];
};
std::unordered_map<void*, PoolTraceRec>* const pool_live = new std::unordered_map<void*, PoolTraceRec>[8];

static std::string frame_name(void* a) {
    Dl_info info{};
    if (dladdr(a, &info) && info.dli_sname) {
        int st = 0;
        char* dm = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &st);
        std::string r = (st == 0 && dm) ? dm : info.dli_sname;
        std::free(dm);
        auto paren = r.find('(');
        if (paren != std::string::npos)
            r = r.substr(0, paren);
        return r;
    }
    char buf[512];
    if (info.dli_fname && info.dli_fbase) {
        const char* base = strrchr(info.dli_fname, '/');
        snprintf(buf, sizeof buf, "%s+0x%lx", base ? base + 1 : info.dli_fname,
                 (unsigned long)((char*)a - (char*)info.dli_fbase));
    } else {
        snprintf(buf, sizeof buf, "%p", a);
    }
    return buf;
}

std::vector<std::pair<std::string, size_t>> GPUpoolTrace(int id, int depth) {
    std::map<std::vector<void*>, size_t> by;
    {
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        for (auto& [p, r] : pool_live[id]) {
            std::vector<void*> key(r.fr + 2, r.fr + std::min(r.n, 2 + depth));
            by[key] += r.bytes;
        }
    }
    std::map<std::string, size_t> named;
    for (auto& [fr, b] : by) {
        std::string k;
        for (void* a : fr) {
            if (!k.empty())
                k += " < ";
            k += frame_name(a);
        }
        named[k] += b;
    }
    std::vector<std::pair<std::string, size_t>> out(named.begin(), named.end());
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
    return out;
}

GPUpoolStats GPUpoolStatsOf(int id) {
    std::lock_guard<std::mutex> guard(mempool_lock[id]);
    GPUpoolStats st{};
    for (auto& [base, sl] : slabs[id]) {
        size_t total = (size_t)sl.block * sl.nblocks;
        st.reserved += total;
        st.idle += (size_t)sl.block * sl.nfree;
        if (sl.nfree == sl.nblocks)
            st.trimmable += total;
        st.slabs++;
    }
    cudaMemPool_t mp;
    if (cudaDeviceGetDefaultMemPool(&mp, id) == cudaSuccess) {
        uint64_t v = 0;
        if (cudaMemPoolGetAttribute(mp, cudaMemPoolAttrReservedMemCurrent, &v) == cudaSuccess) st.cuda_reserved = v;
        if (cudaMemPoolGetAttribute(mp, cudaMemPoolAttrUsedMemCurrent, &v) == cudaSuccess) st.cuda_used = v;
    }
    return st;
}

namespace {
std::mutex g_pinnedMutex;
std::unordered_map<void*, size_t> g_pinnedAnon;   // registered anonymous buffers -> mapped length
bool pinnedAnonOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_PINNED_ANON"); return !e || !*e || std::atoi(e) > 0; }();
    return v;
}
}   // namespace

void* pinnedHostAlloc(size_t bytes, bool mapped) {
    bytes = std::max<size_t>(bytes, 1);
    if (pinnedAnonOn()) {
        const size_t len = (bytes + 4095) & ~size_t(4095);
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            madvise(p, len, MADV_NOHUGEPAGE);   // a huge-page fault would directly compact a fragmented node
            if (cudaHostRegister(p, len, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
                std::lock_guard<std::mutex> g(g_pinnedMutex);
                g_pinnedAnon[p] = len;
                return p;
            }
            cudaGetLastError();
            munmap(p, len);
        }
    }
    void* p = nullptr;
    const cudaError_t e = mapped ? cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable)
                                 : cudaMallocHost(&p, bytes);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return p;
}

void pinnedHostFree(void* p) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> g(g_pinnedMutex);
        auto it = g_pinnedAnon.find(p);
        if (it != g_pinnedAnon.end()) {
            const size_t len = it->second;
            g_pinnedAnon.erase(it);
            cudaHostUnregister(p);
            munmap(p, len);
            return;
        }
    }
    cudaFreeHost(p);
}

//void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
    if (g_cap.active)
        return captureMalloc(bytes, stream);
    void* ptr = nullptr;

    // FIDESLIB_POOL_SLAB_MB (default 64): one live block pins its slab, so small slabs keep idle memory trimmable
    static const uint64_t slab_mb = [] {
        const char* e = std::getenv("FIDESLIB_POOL_SLAB_MB");
        return (e && *e) ? std::max<uint64_t>(1, std::strtoull(e, nullptr, 10)) : uint64_t(64);
    }();
    uint64_t MBs = slab_mb;

    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
        MBs = bytes / 1024;
    }

    if (cache && (bytes & (bytes - 1)) == 0) {
        // The whole pooled path holds the lock: the map insert, the refill/pop sequence and the
        // shared per-id event are all racy under concurrent host threads.
        std::unique_lock<std::mutex> guard(mempool_lock[id]);
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        if (s[id].ptr() == nullptr) {
            s[id].init();
        }
        CudaCheckErrorModNoSync;
        if (free_limb.empty()) {
            uint64_t* base = nullptr;
            const size_t slab = std::max<size_t>(MBs * 1024 * 1024, (size_t)bytes);
            cudaError_t e = cudaMallocAsync((void**)&base, slab, s[id].ptr());
            if (e == cudaErrorMemoryAllocation) {
                cudaGetLastError();
                base = nullptr;
                // the device is full: idle cached objects give their blocks back (often of this very size), then
                // idle slabs of other sizes go
                guard.unlock();
                const size_t n = run_reclaim_hook(id);
                guard.lock();
                if (n)
                    std::cerr << "[gpu_pool] out of memory for a " << bytes << " B block: reclaimed " << n
                              << " cached polynomials" << std::endl;
                if (free_limb.empty())
                    malloc_or_trim(id, (void**)&base, slab, s[id].ptr(), true);
            }
            CudaCheckErrorModNoSync;

            if (base != nullptr) {
                for (size_t i = 0; i < slab; i += bytes) {
                    free_limb.emplace_back(((char*)base) + i);
                }
                slabs[id][(char*)base] = PoolSlab{bytes, (int)(slab / bytes), (int)(slab / bytes)};
            }
        }
        CudaCheckErrorModNoSync;
        if (free_limb.empty())     // never pop an empty list
            throw std::runtime_error("[gpu_pool] device " + std::to_string(id) + " out of memory for a " +
                                     std::to_string(bytes) + " B block (after reclaim and trim)");

        if (stream != nullptr) {
            s[id].record();
            CudaCheckErrorModNoSync;
            cudaStreamWaitEvent(stream, s[id].ev);
        }

        CudaCheckErrorModNoSync;
        ptr = free_limb.back();
        free_limb.pop_back();
        if (PoolSlab* sl = slab_of(id, ptr))
            sl->nfree--;
        if (pool_trace_on()) {
            PoolTraceRec r;
            r.bytes = bytes;
            r.n = backtrace(r.fr, 40);
            pool_live[id][ptr] = r;
        }
        //ptr = free_limb.front();
        //free_limb.pop_front();
        //std::cout << "get " << ptr << std::endl;
        return ptr;
    }

    //std::cout << bytes << std::endl;
    if (0) {
        cudaMalloc(&ptr, bytes);
    } else if (1) {
        if (malloc_or_trim(id, &ptr, bytes, 0, false) != cudaSuccess || ptr == nullptr) {
            cudaGetLastError();
            throw std::runtime_error("[gpu_pool] device " + std::to_string(id) + " out of memory for " +
                                     std::to_string(bytes) + " B (after reclaim and trim)");
        }
        if (pool_trace_on()) {     // unpooled sizes are traced too
            PoolTraceRec r;
            r.bytes = bytes;
            r.n = backtrace(r.fr, 40);
            std::lock_guard<std::mutex> guard(mempool_lock[id]);
            pool_live[id][ptr] = r;
        }
    } else {
        if (size_to_memory[id][bytes].empty()) {
            cudaSetDevice(id);
            cudaMallocAsync(&ptr, bytes, stream);
        } else {
            mempool_lock[id].lock();
            if (size_to_memory[id][bytes].empty()) {
                mempool_lock[id].unlock();
                cudaSetDevice(id);
                cudaMallocAsync(&ptr, bytes, stream);
            } else {
                ptr = size_to_memory[id][bytes].back();
                size_to_memory[id][bytes].pop_back();
                mempool_lock[id].unlock();
            }
        }
    }

    return ptr;
}

struct pointerdata {
    void* pointer;
    int id;
    int bytes;
};

void CUDART_CB streamCallback(void* userData) {

    auto* p = reinterpret_cast<pointerdata*>(userData);

    mempool_lock[p->id].lock();
    size_to_memory[p->id][p->bytes].push_back(p->pointer);
    mempool_lock[p->id].unlock();
    delete p;
}

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {
    if (g_cap.active) {
        captureFree(ptr, bytes, stream);
        return;
    }
    if (ownedFast(ptr))
        return;   // graph memory never goes back to the pool
    uint64_t MBs = 1024;

    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
        MBs = bytes / 1024;
    }

    if (cache && (bytes & (bytes - 1)) == 0) {
        // Same full-lock rule as GPUmalloc (map insert + shared event).
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        if (s[id].ptr() == nullptr) {
            s[id].init();
        }
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        CudaCheckErrorModNoSync;
        // cudaDeviceSynchronize();
        if (stream != nullptr) {
            s[id].wait(stream);
        }
        CudaCheckErrorModNoSync;
        free_limb.emplace_back(ptr);
        if (PoolSlab* sl = slab_of(id, ptr))
            sl->nfree++;
        if (pool_trace_on())
            pool_live[id].erase(ptr);
        //std::cout << "free " << ptr << std::endl;
        return;
    }
    if (s[id].ptr() == nullptr) {
        s[id].init();
    }

    if (pool_trace_on()) {
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        pool_live[id].erase(ptr);
    }
    if (0) {
        cudaFree(ptr);
    } else if (1) {
        cudaFreeAsync(ptr, stream);
    } else {
        auto* p = new pointerdata;
        p->id = id;
        p->bytes = bytes;
        p->pointer = ptr;
        cudaLaunchHostFunc(stream, streamCallback, p);
    }
}

int GetTargetThreads(int id) {
    return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}

}  // namespace FIDESlib