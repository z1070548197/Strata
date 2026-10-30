// src/core/tp_allreduce.cu - see include/strata/core/tp_allreduce.hpp.
//
// Two-GPU allreduce over P2P, single-writer form.  One device (dev_a) runs a single
// kernel that reads both buffers (buf_a local, buf_b via P2P), computes the element-wise
// sum, and writes it back to both (buf_a local, buf_b via P2P).  After the kernel,
// buf_a == buf_b == original_a + original_b.
//
// Why single-writer instead of a two-phase ring: a symmetric two-phase design (each
// device reads the other, then each adds) has a cross-device race - dev_a's phase 2
// writes buf_a while dev_b's phase 1 is still reading buf_a, and the two devices have
// no ordering between them without an explicit barrier.  The single-writer form removes
// the race entirely: only dev_a's kernel runs, each element is touched by exactly one
// thread (read both, write both), so there is no read/write overlap.  The cost is that
// dev_a does all the work - but the allreduce message is tiny (H=2560 floats = 10 KiB
// per token) and the real tensor-parallel work (the layer compute) is what is split
// across the two cards, so an unbalanced 10 KiB glue operation is the right trade.
//
// The P2P detection/enable mirrors peer_experts.cpp (cudaDeviceCanAccessPeer both
// ways, then cudaDeviceEnablePeerAccess both ways, tolerating
// cudaErrorPeerAccessAlreadyEnabled).  STRATA_TP_NO_P2P=1 forces P2P off (test knob).
#include "strata/core/tp_allreduce.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>

namespace strata::core {
namespace {

constexpr int THREADS = 256;

inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("tp allreduce: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

/// Switch the calling thread to `dev` and back (the engine's thread lives on device 0).
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// Single-writer allreduce: `dst_a[i] = dst_b[i] = dst_a[i] + dst_b[i]`.
/// `dst_a` is on the launching device (local); `dst_b` is the peer's buffer (P2P).
/// One thread per element; the read of both and the write of both happen in the same
/// thread, so there is no read/write race on either buffer.
__global__ void allreduce_kernel(float* __restrict__ dst_a, float* __restrict__ dst_b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float s = dst_a[i] + dst_b[i];
        dst_a[i] = s;
        dst_b[i] = s;
    }
}

/// One-way P2P copy: `dst_b[i] = src_a[i]` (src_a local, dst_b the peer's buffer).
__global__ void broadcast_kernel(const float* __restrict__ src_a, float* __restrict__ dst_b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst_b[i] = src_a[i];
}
__global__ void copy_rows_kernel(const float* __restrict__ src, int64_t src_ld,
                                 float* __restrict__ dst, int64_t dst_ld,
                                 int64_t rows, int64_t cols) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int64_t row = i / cols, col = i % cols;
    dst[row * dst_ld + col] = src[row * src_ld + col];
}

// ---- fused handshake forms (P1): one launch per layer instead of three, the way vLLM's custom all-reduce
// folds its barrier into the reduction kernel.  `want` is the layer's sequence number: the flags carry the
// global layer index + 1, so a window's L layers raise 1..L and nothing else matches.
__device__ __forceinline__ void spin_ge(const volatile uint32_t* flag, uint32_t want) {
#if defined(__CUDA_ARCH__)
    while (*flag < want) {
#if __CUDA_ARCH__ < 700
        // sm_6x: spin (as strata_spin_pause does); every call site is a single-thread wait
#else
        __nanosleep(100);
#endif
    }
#endif
}

/// rank 0: wait for the peer's partial, sum both buffers, tell the peer the sum is there.
/// ONE block: the handshake and the reduction then share a block, so __syncthreads() is a full barrier
/// between "every element written" and "the peer may read" - no grid-wide counter, no extra flag word.
/// The window's residual is 7*2560 floats = 70 KiB; one block of 256 threads moves that in a few us.
__global__ void allreduce_handshake_kernel(float* __restrict__ dst_a, float* __restrict__ dst_b, int64_t n,
                                           const volatile uint32_t* wait_flag, uint32_t wait_value,
                                           uint32_t* raise_flag, uint32_t raise_value) {
    if (threadIdx.x == 0) {
        spin_ge(wait_flag, wait_value);
        __threadfence_system();
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float s = dst_a[i] + dst_b[i];
        dst_a[i] = s;
        dst_b[i] = s;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *raise_flag = raise_value;
}

/// rank 1: publish this card's partial, then wait for the sum to land in `buf_b` (written over P2P by rank 0).
__global__ void publish_wait_kernel(uint32_t* raise_flag, uint32_t raise_value,
                                    const volatile uint32_t* wait_flag, uint32_t wait_value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *raise_flag = raise_value;
        spin_ge(wait_flag, wait_value);
        __threadfence_system();
    }
}

/// `dst[i] = src[i] + off`: the experts' FFN split addresses each rank's half of a packed blob by shifting the
/// group pointers (see tp_moe_split.hpp), and the plan's pointer array is device memory.
__global__ void shift_ptrs_kernel(const unsigned long long* __restrict__ src, unsigned long long* __restrict__ dst,
                                  int64_t n, unsigned long long off) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] + off;
}

}  // namespace

TpAllreduce::~TpAllreduce() { close(); }

void TpAllreduce::close() {
    if (dev_a_ < 0) return;
    p2p_ = false;
    dev_a_ = dev_b_ = -1;
}

bool TpAllreduce::open(int dev_a, int dev_b, std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (dev_a < 0 || dev_a >= count || dev_b < 0 || dev_b >= count || dev_a == dev_b) {
        err = "tp allreduce: need two distinct visible devices (got " + std::to_string(dev_a) + ", " +
              std::to_string(dev_b) + "; " + std::to_string(count) + " visible)";
        return false;
    }
    dev_a_ = dev_a;
    dev_b_ = dev_b;
    int a = 0, b = 0;
    cudaDeviceCanAccessPeer(&a, dev_a, dev_b);
    cudaDeviceCanAccessPeer(&b, dev_b, dev_a);
    if (const char* np = std::getenv("STRATA_TP_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = b = 0;
    p2p_ = a && b;
    if (p2p_) {
        On on_a(dev_a);
        cudaError_t e_a = cudaDeviceEnablePeerAccess(dev_b, 0);
        if (e_a == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e_a = cudaSuccess; }
        On on_b(dev_b);
        cudaError_t e_b = cudaDeviceEnablePeerAccess(dev_a, 0);
        if (e_b == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e_b = cudaSuccess; }
        p2p_ = e_a == cudaSuccess && e_b == cudaSuccess;
        if (!p2p_) err = "tp allreduce: cudaDeviceEnablePeerAccess failed";
    }
    return p2p_;
}

bool TpAllreduce::allreduce(float* buf_a, float* buf_b, int64_t n,
                            cudaStream_t stream_a, cudaStream_t stream_b, std::string& err) {
    if (dev_a_ < 0 || !p2p_) { err = "tp allreduce: not open with P2P"; return false; }
    if (n <= 0) return true;
    // Single writer: dev_a's kernel reads and writes both buffers.  `stream_a` must be a
    // stream on dev_a.  `stream_b` is accepted for interface symmetry but unused - the
    // single-writer form has no work on dev_b.  The caller must order dev_b's subsequent
    // work after this kernel completes (e.g. sync stream_a, or record an event).
    (void) stream_b;
    On on_a(dev_a_);
    allreduce_kernel<<<grid_for(n), THREADS, 0, stream_a>>>(buf_a, buf_b, n);
    return ck(cudaGetLastError(), "allreduce kernel", err);
}

bool TpAllreduce::allreduce_wait_raise(float* buf_a, float* buf_b, int64_t n, cudaStream_t stream_a,
                                       const uint32_t* wait_flag, uint32_t wait_value,
                                       uint32_t* raise_flag, uint32_t raise_value, std::string& err) {
    if (dev_a_ < 0 || !p2p_) { err = "tp allreduce: not open with P2P"; return false; }
    if (n <= 0) return true;
    On on_a(dev_a_);
    // one block (see the kernel's comment); 1024 threads keeps 70 KiB at ~18 floats per lane
    allreduce_handshake_kernel<<<1, 1024, 0, stream_a>>>(
        buf_a, buf_b, n, (const volatile uint32_t*) wait_flag, wait_value, raise_flag, raise_value);
    return ck(cudaGetLastError(), "allreduce handshake kernel", err);
}

bool TpAllreduce::publish_wait(uint32_t* raise_flag, uint32_t raise_value,
                               const uint32_t* wait_flag, uint32_t wait_value, cudaStream_t stream,
                               std::string& err) {
    // No p2p_ check: rank 1 only raises its own flag and spins on rank 0's - it never touches the peer's
    // memory, and its TpAllreduce object was never opened (rank 0 owns the P2P pair).
    if (dev_b_ < 0) { err = "tp allreduce: publish_wait on a closed object"; return false; }
    On on_b(dev_b_);
    publish_wait_kernel<<<1, 1, 0, stream>>>(raise_flag, raise_value, (const volatile uint32_t*) wait_flag,
                                             wait_value);
    return ck(cudaGetLastError(), "publish_wait kernel", err);
}

bool TpAllreduce::broadcast(float* src_a, float* dst_b, int64_t n, cudaStream_t stream_a, std::string& err) {
    if (dev_a_ < 0 || !p2p_) { err = "tp allreduce: broadcast needs P2P"; return false; }
    if (n <= 0) return true;
    On on_a(dev_a_);
    broadcast_kernel<<<grid_for(n), THREADS, 0, stream_a>>>(src_a, dst_b, n);
    return ck(cudaGetLastError(), "broadcast kernel", err);
}

bool TpAllreduce::broadcast_to_a(float* src_b, float* dst_a, int64_t n, cudaStream_t stream_b,
                                 std::string& err) {
    if (dev_a_ < 0 || !p2p_) { err = "tp allreduce: broadcast_to_a needs P2P"; return false; }
    if (n <= 0) return true;
    // rank 1's card runs it: reads ITS buffer, writes the peer's over P2P (the reverse of broadcast)
    On on_b(dev_b_);
    broadcast_kernel<<<grid_for(n), THREADS, 0, stream_b>>>(src_b, dst_a, n);
    return ck(cudaGetLastError(), "broadcast_to_a kernel", err);
}

bool TpAllreduce::copy_rows(float* src, int64_t src_ld, float* dst, int64_t dst_ld,
                            int64_t rows, int64_t cols, bool from_a, cudaStream_t stream,
                            std::string& err) {
    if (dev_a_ < 0 || !p2p_) { err = "tp allreduce: copy_rows needs P2P"; return false; }
    if (rows <= 0 || cols <= 0) return true;
    if (src_ld < cols || dst_ld < cols) { err = "tp allreduce: copy_rows pitch is smaller than width"; return false; }
    On on(from_a ? dev_a_ : dev_b_);
    copy_rows_kernel<<<grid_for(rows * cols), THREADS, 0, stream>>>(src, src_ld, dst, dst_ld, rows, cols);
    return ck(cudaGetLastError(), "copy_rows kernel", err);
}

bool tp_shift_group_ptrs(const unsigned long long* src, unsigned long long* dst, int64_t n,
                         unsigned long long off, cudaStream_t stream, std::string& err) {
    if (n <= 0) return true;
    shift_ptrs_kernel<<<grid_for(n), THREADS, 0, stream>>>(src, dst, n, off);
    return ck(cudaGetLastError(), "shift pointers kernel", err);
}

}  // namespace strata::core
