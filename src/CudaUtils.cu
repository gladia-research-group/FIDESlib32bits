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

    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventCreate(&ev, cudaEventDisableTiming);
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
//void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
    void* ptr = nullptr;

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
        // The whole pooled path holds the lock: the map insert, the refill/pop sequence and the
        // shared per-id event are all racy under concurrent host threads.
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        if (s[id].ptr() == nullptr) {
            s[id].init();
        }
        CudaCheckErrorModNoSync;
        if (free_limb.empty()) {
            uint64_t* base;
            cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());

            for (int i = 0; i < MBs * 1024 * 1024; i += bytes) {
                free_limb.emplace_back(((char*)base) + i);
            }
        }
        CudaCheckErrorModNoSync;

        if (stream != nullptr) {
            s[id].record();
            CudaCheckErrorModNoSync;
            cudaStreamWaitEvent(stream, s[id].ev);
        }

        CudaCheckErrorModNoSync;
        ptr = free_limb.back();
        free_limb.pop_back();
        //ptr = free_limb.front();
        //free_limb.pop_front();
        //std::cout << "get " << ptr << std::endl;
        return ptr;
    }

    //std::cout << bytes << std::endl;
    if (0) {
        cudaMalloc(&ptr, bytes);
    } else if (1) {
        cudaMallocAsync(&ptr, bytes, 0);
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