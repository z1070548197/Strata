// STRATA_GPU_BUSY: see include/strata/core/gpu_busy.hpp for what this is and what it costs.
#include "strata/core/gpu_busy.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace strata {
namespace core {
namespace {

// Bounds on the spin.  Each outer round is kRounds multiplies per thread (register-only); at 256 threads
// one round is ~kRounds cycles, so kCapRound defaults well past a 70 ms window (the longest observed).
// The cap exists only so a lost end() cannot leave a kernel spinning forever.
constexpr int kThreads = 256;
constexpr int kRounds = 256;      // multiplies between flag reads
constexpr int kFlagEvery = 8;     // rounds between flag reads -> one PCIe read per 2048 multiplies
constexpr unsigned long long kCapRound = 400000;   // ~0.4 s of spinning at worst

__global__ void gpu_busy_spin_kernel(unsigned int* flag) {
    __shared__ unsigned int go;
    unsigned int acc = 1664525u * (threadIdx.x + 1u) + 1013904223u;
    if (threadIdx.x == 0) go = *(volatile unsigned int*) flag;
    __syncthreads();
    for (unsigned long long it = 0; it < kCapRound; ++it) {
#pragma unroll 4
        for (int j = 0; j < kRounds; ++j) acc = acc * 1664525u + 1013904223u;
        if ((it & (unsigned long long) (kFlagEvery - 1)) != 0) continue;
        if (threadIdx.x == 0) {
            __threadfence_system();
            go = *(volatile unsigned int*) flag;
        }
        __syncthreads();
        if (go != 0u) break;
    }
    // Keeps the recurrence alive: the compiler cannot prove acc never takes this value, so it may not
    // fold the chain away.
    if (acc == 0xDEADBEEFu) __threadfence_system();
}

struct State {
    cudaStream_t s = nullptr;
    unsigned int* h_flag = nullptr;   // mapped host memory (what the host writes)
    unsigned int* d_flag = nullptr;   // its device alias (what the kernel reads)
    bool launched = false;            // a kernel may still be running on s
};

std::mutex g_mu;
std::map<int, State> g_states;

bool env_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GPU_BUSY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

// Drops a still-running kernel from the previous window before relaunching.  Bounded: the kernel leaves
// within microseconds of the flag write, so this normally costs one query.
void settle(State& st) {
    if (!st.launched) return;
    for (int i = 0; i < 2000; ++i) {
        const cudaError_t q = cudaStreamQuery(st.s);
        if (q == cudaSuccess) { st.launched = false; return; }
        if (q != cudaErrorNotReady) { (void) cudaGetLastError(); st.launched = false; return; }
    }
    // Still spinning after ~4 ms: leave it.  begin() rewrites the flag, so the old kernel keeps running to
    // its cap while the new one runs beside it - wasteful, never wrong.
}

}  // namespace

bool gpu_busy_enabled() { return env_on(); }

bool gpu_busy_init(std::string& err) {
    if (!env_on()) return true;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { err = std::string("gpu_busy: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_states.count(dev)) return true;
    State st;
    if (cudaHostAlloc((void**) &st.h_flag, sizeof(unsigned int), cudaHostAllocMapped) != cudaSuccess) {
        err = std::string("gpu_busy: cudaHostAlloc: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    *st.h_flag = 1u;   // released: a stale kernel from a previous process would leave at once
    if (cudaHostGetDevicePointer((void**) &st.d_flag, st.h_flag, 0) != cudaSuccess) {
        err = std::string("gpu_busy: cudaHostGetDevicePointer: ") + cudaGetErrorString(cudaGetLastError());
        cudaFreeHost(st.h_flag);
        return false;
    }
    int lo = 0, hi = 0;
    cudaError_t se = cudaErrorInvalidValue;
    if (cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess)   // lo = the least priority there is
        se = cudaStreamCreateWithPriority(&st.s, cudaStreamNonBlocking, lo);
    if (se != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFreeHost(st.h_flag);
        err = std::string("gpu_busy: cudaStreamCreateWithPriority: ") + cudaGetErrorString(se);
        return false;
    }
    g_states.emplace(dev, st);
    // Self-evidence: a null A/B with this feature on is ambiguous unless the log says it armed.  One line per
    // device, at the first window.
    std::fprintf(stderr, "strata gpu_busy: device %d, spin kernel armed on a lowest-priority stream (STRATA_GPU_BUSY, "
                         "%d threads, one block)\n", dev, kThreads);
    return true;
}

void gpu_busy_begin() {
    if (!env_on()) return;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return; }
    bool have = false;
    { std::lock_guard<std::mutex> lk(g_mu); have = g_states.count(dev) != 0; }
    if (!have) {   // first window on this device: set the stream and the flag up now
        std::string e;
        if (!gpu_busy_init(e)) {
            std::fprintf(stderr, "strata gpu_busy: %s (the window runs with the feature OFF)\n", e.c_str());
            return;
        }
    }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_states.find(dev);
    if (it == g_states.end()) return;
    State& st = it->second;
    settle(st);
    *st.h_flag = 0u;   // hold
    cudaLaunchKernel((const void*) gpu_busy_spin_kernel, dim3(1), dim3(kThreads), (void**) &st.d_flag, 0, st.s);
    if (cudaGetLastError() != cudaSuccess) { *st.h_flag = 1u; return; }
    st.launched = true;
}

void gpu_busy_end() {
    if (!env_on()) return;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return; }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_states.find(dev);
    if (it == g_states.end()) return;
    *it->second.h_flag = 1u;   // release; the kernel leaves within microseconds
}

void gpu_busy_shutdown() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& kv : g_states) {
        State& st = kv.second;
        if (st.h_flag != nullptr) *st.h_flag = 1u;
        if (st.s != nullptr) { (void) cudaStreamSynchronize(st.s); (void) cudaStreamDestroy(st.s); st.s = nullptr; }
        if (st.h_flag != nullptr) { (void) cudaFreeHost(st.h_flag); st.h_flag = nullptr; st.d_flag = nullptr; }
    }
    g_states.clear();
}

GpuBusyScope::GpuBusyScope() { gpu_busy_begin(); }
GpuBusyScope::~GpuBusyScope() { gpu_busy_end(); }

}  // namespace core
}  // namespace strata
