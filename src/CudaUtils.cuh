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
        cudaStreamSynchronize(0);                                                            \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled &&                    \
            e != cudaErrorCudartUnloading) {                                                 \
            fprintf(stderr, "Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            _Exit(1);                                                                        \
        }                                                                                    \
    } while (0)

#define CudaCheckErrorModNoSync                                                                                   \
    do {                                                                                                          \
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

class Stream {
   private:
    cudaStream_t ptr_ = nullptr;

   public:
    cudaEvent_t ev = nullptr;
    bool updated = false;
    //Event ev;

    void init(int priority = 0);

    cudaStream_t ptr() {
        updated = false;
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


}  // namespace FIDESlib
#endif  //FIDESLIB_CUDAUTILS_CUH
