//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cstdlib>   // _Exit

//#define NCCL

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <execinfo.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace FIDESlib {

extern std::vector<cudaDeviceProp> GPUprop;
void initGPUprop();
int GetTargetThreads(int id);


enum NVTX_CATEGORIES { NONE, LIFETIME, FUNCTION };

// NVTX FUNCTION ranges, header-only nvtx3, OFF unless FIDESLIB_NVTX=1 (profiling only: the per-op
// ranges give nsys a stage breakdown of the bootstrap). LIFETIME ranges stay inert (they raced).
inline bool nvtxFunctionRangesOn() {
    static const bool v = [] { const char* e = std::getenv("FIDESLIB_NVTX"); return e && std::atoi(e) != 0; }();
    return v;
}
inline void CudaNvtxStart(const std::string& s, NVTX_CATEGORIES c = FUNCTION, int = 0) {
    if (c == FUNCTION && nvtxFunctionRangesOn()) nvtxRangePushA(s.c_str());
}
inline void CudaNvtxStop(const std::string& = "", NVTX_CATEGORIES c = FUNCTION) {
    if (c == FUNCTION && nvtxFunctionRangesOn()) nvtxRangePop();
}
struct CudaNvtxRange {
    bool on = false;
    explicit CudaNvtxRange(const std::string& s, NVTX_CATEGORIES c = FUNCTION, int = 0) : on(c == FUNCTION && nvtxFunctionRangesOn()) {
        if (on) nvtxRangePushA(s.c_str());
    }
    explicit CudaNvtxRange(const char* s, NVTX_CATEGORIES c = FUNCTION, int = 0) : on(c == FUNCTION && nvtxFunctionRangesOn()) {
        if (on) nvtxRangePushA(s);
    }
    CudaNvtxRange(CudaNvtxRange&& o) noexcept : on(o.on) { o.on = false; }
    ~CudaNvtxRange() {
        if (on) nvtxRangePop();
    }
};

int getNumDevices();

void CudaHostSync();

inline void breakpoint() {}

/* Fatal CUDA errors exit with _Exit(1): a nonzero status, and no atexit/static-destructor run against a
 * dead CUDA context. cudaErrorCudartUnloading is whitelisted because it is the normal state inside static
 * destructors at process exit, where these macros are still reached. */
#define CudaCheckErrorMod                                                                    \
    do {                                                                                     \
        if (FIDESlib::captureActiveNow()) break;                                             \
        cudaDeviceSynchronize();                                                             \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled &&                    \
            e != cudaErrorCudartUnloading) {                                                 \
                                                                                             \
            fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            _Exit(1);                                                                        \
        }                                                                                    \
    } while (0)

#define CudaCheckErrorModMGPU                                                                \
    do {                                                                                     \
        if (FIDESlib::captureActiveNow()) break;                                             \
        cudaStreamSynchronize(0);                                                            \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled &&                    \
            e != cudaErrorCudartUnloading) {                                                 \
            fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            _Exit(1);                                                                        \
        }                                                                                    \
    } while (0)

bool errcheckOn();     // FIDESLIB_ERRCHECK=1: the per-launch cudaGetLastError polls (off by default: ~1.8M calls/token)
bool ompRegionsOn();   // FIDESLIB_OMP_REGIONS=1: single-GPU `omp parallel` regions (off by default: they run inline)
bool captureActiveNow();   // a FIDESLIB_BTS_GRAPH capture is in progress (error polls and syncs are skipped)
#define CudaCheckErrorModNoSync                                                                                   \
    do {                                                                                                          \
        if (!FIDESlib::errcheckOn() || FIDESlib::captureActiveNow()) break;                                       \
        /*cudaDeviceSynchronize();*/                                                                              \
        cudaError_t e = cudaGetLastError();                                                                       \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && e != cudaErrorGraphExecUpdateFailure \
            && e != cudaErrorCudartUnloading) {                                                                   \
            void* array[10];                                                                                      \
            size_t size;                                                                                          \
            size = backtrace(array, 10);                                                                          \
            backtrace_symbols_fd(array, size, STDERR_FILENO);                                                     \
            fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            FIDESlib::breakpoint();                                                                               \
            _Exit(1);                                                                                                 \
        }                                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                              \
    do {                                                                                            \
        ncclResult_t res = cmd;                                                                     \
        if (res != ncclSuccess) {                                                                   \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

class Event;

extern std::map<void*, int> free;

// ---- Stream capture of whole operations into CUDA graphs (FIDESLIB_BTS_GRAPH). While a capture is active every
// stream the operation touches joins it (Stream::ptr/record/wait), waits on work that predates the capture are dropped
// (the replay is fenced by a legacy-stream barrier instead), GPUmalloc/GPUfree serve the capture from graph memory
// (CaptureRegion: a shared arena for the big buffers, reused across graphs since replays are serialized, and a
// per-graph region for the small tables uploaded once at capture time), and stream destruction is deferred to the end.
struct CaptureRegion;
struct CaptureState {
    bool active = false;
    bool invalid = false;
    const char* why = nullptr;
    uint32_t gen = 0;
    cudaStream_t origin = nullptr;
    cudaEvent_t originEv = nullptr;
    CaptureRegion* small = nullptr;   // the graph's own small allocations
};
extern CaptureState g_cap;
void captureInvalidate(const char* why);
void captureJoin(cudaStream_t s);
bool captureBegin(cudaStream_t origin, CaptureRegion* small);
cudaGraph_t captureEnd();                       // nullptr if the capture was invalidated
bool captureEverStarted();                      // some capture happened: buffers may be referenced by a graph
CaptureRegion* captureRegionNew();
void captureRegionFree(CaptureRegion* r);
bool captureOwned(const void* p);               // memory owned by graphs (never returned to the pool)




class Stream {
   private:
    cudaStream_t ptr_ = nullptr;

   public:
    cudaEvent_t ev = nullptr;
    bool updated = false;
    uint32_t capGen = 0;    // capture generation of ev's last record (0: recorded eagerly)
    uint32_t joinGen = 0;   // capture generation in which ptr_ joined the capture
    //Event ev;

    void init(int priority = 0);

    void joinCapture() {
        captureJoin(ptr_);
        joinGen = g_cap.gen;
    }

    cudaStream_t ptr() {
        updated = false;
        if (g_cap.active && ptr_ && joinGen != g_cap.gen)
            joinCapture();
        return ptr_;
    }

    void initDefault();

    //void wait(const Event &ev) const;
    void wait(Stream& s, bool external = false);
    void wait(cudaStream_t s);

    Stream();

    Stream(Stream& s) = delete;

    Stream(const Stream& s) = delete;

    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& s) noexcept;

    ~Stream();

    void record(bool external = false);

    void wait_recorded(const Stream& s);

    void capture_begin();

    void capture_end();
};

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache = false);
void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache = false);

/* Page-locked host buffers (`mapped`: device-mapped as well). cudaMallocHost/cudaHostAlloc pages are /dev/zero shmem:
 * movable LRU pages that kcompactd isolates, unmaps and fails to migrate, again and again, while their node is
 * fragmented (~400k pages/s, the host thread ~25% slower meanwhile). FIDESLIB_PINNED_ANON (default 1; 0 = the driver
 * allocation): anonymous memory pinned with cudaHostRegister instead, whose pinned pages compaction skips. nullptr on failure. */
void* pinnedHostAlloc(size_t bytes, bool mapped = false);
void pinnedHostFree(void* p);

/* The pooled allocator's slabs on device `id`: reserved bytes, the idle part of them, and the part in slabs that are
 * idle as a whole (what GPUpoolTrim would release). */
struct GPUpoolStats {
    size_t reserved = 0, idle = 0, trimmable = 0, slabs = 0;
    size_t cuda_reserved = 0, cuda_used = 0;   // the CUDA stream-ordered pool (cudaMallocAsync)
};
GPUpoolStats GPUpoolStatsOf(int id);
/* Hand the idle slabs of device `id` back to the CUDA stream-ordered pool (shared by every allocation). Returns the
 * bytes released. Also done automatically before an allocation gives up. */
size_t GPUpoolTrim(int id);
/* A callback that frees idle cached device objects of device `id` (their blocks go back to the pool's free lists);
 * run before an allocation gives up. Returns the objects freed. An empty function unregisters. */
void GPUpoolSetReclaimHook(int id, std::function<size_t()> hook);
/* Per pooled block size: {block bytes, reserved bytes, in-use bytes, slabs}. */
std::vector<std::array<size_t, 4>> GPUpoolBySize(int id);
/* FIDESLIB_POOL_TRACE=1 only: live pooled bytes per allocation call chain (`depth` frames above GPUmalloc), largest
 * first. */
std::vector<std::pair<std::string, size_t>> GPUpoolTrace(int id, int depth);



// ---- Capture hooks (FIDESlib's own translation units only, FIDESLIB_CAPTURE_HOOKS): outside a capture they are the
// plain CUDA calls; inside one, pageable H2D copies become memcpy nodes from a pinned per-graph snapshot, and anything
// that cannot be captured (syncs, D2H reads, freeing memory older than the capture) invalidates it and is skipped.
cudaError_t captureMemcpyAsync(const char* f, int l, void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s);
cudaError_t captureBlocked(const char* f, int l, const char* what);
cudaError_t captureMallocAsyncHook(void** p, size_t n, cudaStream_t s);
cudaError_t captureFreeAsyncHook(const char* f, int l, void* p, cudaStream_t s);
cudaError_t captureStreamSync(const char* f, int l, cudaStream_t s);
cudaError_t captureEventRecord(const char* f, int l, cudaEvent_t e, cudaStream_t s, unsigned flags);
cudaError_t captureStreamWaitEvent(const char* f, int l, cudaStream_t s, cudaEvent_t e, unsigned flags);

inline cudaError_t hookMemcpyAsync(const char* f, int l, void* dst, const void* src, size_t n, cudaMemcpyKind k,
                                   cudaStream_t s = 0) {
    if (!g_cap.active) return (cudaMemcpyAsync)(dst, src, n, k, s);
    return captureMemcpyAsync(f, l, dst, src, n, k, s);
}
inline cudaError_t hookMemcpy(const char* f, int l, void* dst, const void* src, size_t n, cudaMemcpyKind k) {
    if (!g_cap.active) return (cudaMemcpy)(dst, src, n, k);
    return captureBlocked(f, l, "cudaMemcpy");
}
inline cudaError_t hookMemset(const char* f, int l, void* dst, int v, size_t n) {
    if (!g_cap.active) return (cudaMemset)(dst, v, n);
    return captureBlocked(f, l, "cudaMemset");
}
inline cudaError_t hookStreamSynchronize(const char* f, int l, cudaStream_t s) {
    if (!g_cap.active) return (cudaStreamSynchronize)(s);
    return captureStreamSync(f, l, s);
}
inline cudaError_t hookDeviceSynchronize(const char* f, int l) {
    if (!g_cap.active) return (cudaDeviceSynchronize)();
    return captureBlocked(f, l, "cudaDeviceSynchronize");
}
template <class T>
inline cudaError_t hookMallocAsync(const char*, int, T** p, size_t n, cudaStream_t s) {
    if (!g_cap.active) return (cudaMallocAsync)((void**)p, n, s);
    return captureMallocAsyncHook((void**)p, n, s);
}
inline cudaError_t hookFreeAsync(const char* f, int l, void* p, cudaStream_t s) {
    if (!g_cap.active) return (cudaFreeAsync)(p, s);
    return captureFreeAsyncHook(f, l, p, s);
}
inline cudaError_t hookEventRecord(const char* f, int l, cudaEvent_t e, cudaStream_t s = 0) {
    if (!g_cap.active) return (cudaEventRecord)(e, s);
    return captureEventRecord(f, l, e, s, 0);
}
inline cudaError_t hookEventRecordWithFlags(const char* f, int l, cudaEvent_t e, cudaStream_t s = 0, unsigned flags = 0) {
    if (!g_cap.active) return (cudaEventRecordWithFlags)(e, s, flags);
    return captureEventRecord(f, l, e, s, flags);
}
inline cudaError_t hookStreamWaitEvent(const char* f, int l, cudaStream_t s, cudaEvent_t e, unsigned flags = 0) {
    if (!g_cap.active) return (cudaStreamWaitEvent)(s, e, flags);
    return captureStreamWaitEvent(f, l, s, e, flags);
}
inline cudaError_t hookFree(const char* f, int l, void* p) {
    if (!g_cap.active) return (cudaFree)(p);
    return captureBlocked(f, l, "cudaFree");
}

}  // namespace FIDESlib

#if defined(FIDESLIB_CAPTURE_HOOKS) && !defined(cudaMemcpyAsync) && !defined(cudaStreamSynchronize)
#define cudaMemcpyAsync(...) FIDESlib::hookMemcpyAsync(__FILE__, __LINE__, __VA_ARGS__)
#define cudaMemcpy(...) FIDESlib::hookMemcpy(__FILE__, __LINE__, __VA_ARGS__)
#define cudaMemset(...) FIDESlib::hookMemset(__FILE__, __LINE__, __VA_ARGS__)
#define cudaStreamSynchronize(...) FIDESlib::hookStreamSynchronize(__FILE__, __LINE__, __VA_ARGS__)
#define cudaDeviceSynchronize() FIDESlib::hookDeviceSynchronize(__FILE__, __LINE__)
#define cudaMallocAsync(...) FIDESlib::hookMallocAsync(__FILE__, __LINE__, __VA_ARGS__)
#define cudaFreeAsync(...) FIDESlib::hookFreeAsync(__FILE__, __LINE__, __VA_ARGS__)
#define cudaFree(...) FIDESlib::hookFree(__FILE__, __LINE__, __VA_ARGS__)
#define cudaEventRecord(...) FIDESlib::hookEventRecord(__FILE__, __LINE__, __VA_ARGS__)
#define cudaEventRecordWithFlags(...) FIDESlib::hookEventRecordWithFlags(__FILE__, __LINE__, __VA_ARGS__)
#define cudaStreamWaitEvent(...) FIDESlib::hookStreamWaitEvent(__FILE__, __LINE__, __VA_ARGS__)
#endif
#endif  //FIDESLIB_CUDAUTILS_CUH

