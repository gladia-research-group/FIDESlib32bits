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
    cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
    updated = true;
#endif
}

void Stream::wait_recorded(const Stream& s) {
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
#if !DISABLE_STREAMS
    assert(s.updated);
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
    if (!s.updated) {
        assert(!external);  // Has to be recorded in the origin graph
        CudaCheckErrorModNoSync;
        cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
        s.updated = true;
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
    CudaCheckErrorModNoSync;
    cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
    updated = false;
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
void Stream::init(int priority) {
    if (ptr_) {
        //free[ptr]++;
        cudaEventDestroy(ev);
        cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
        ev = nullptr;
    }

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
    if (ptr_) {
        //free[ptr]++;
        cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
    }
    if (ev) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept : ptr_(s.ptr_), ev(s.ev) {
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
        // every queued kernel finishes before anything is released: a block can be idle in the pool's books while a
        // kernel on another stream still reads it, and the trim hands its memory to the next allocation
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
    cudaStreamSynchronize(s[id].ptr());
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
    return st;
}
//void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
    void* ptr = nullptr;

    // slab size of the pooled classes >= 64 KB (FIDESLIB_POOL_SLAB_MB, default 64): a slab goes back to CUDA only when
    // every block in it is free, so one survivor pins the whole slab; small slabs keep idle memory trimmable
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
        if (free_limb.empty())     // still full after the reclaim and the trim: fail loudly, never pop an empty list
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