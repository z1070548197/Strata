// tests/core/tp_p2p_bench.cu - P2P (NVLink) bandwidth and latency benchmark.
//
// Phase 1 of the tensor-parallel work measured the allreduce at ONE size (71 KB,
// 11.3 us).  This benchmark measures the underlying P2P link across a range of sizes
// so Phase 2 can decide: per-layer allreduce vs batched allreduce, and whether weights
// should stream over NVLink.  It is a pure measurement - no model, no engine, no
// service.  It allocates small buffers (<= 8 MiB per device) so it does not disturb
// the running model's VRAM headroom.
//
// Measures, for each size:
//   - P2P read:  a kernel on dev0 reads `n` floats from dev1's buffer (P2P), writes local
//   - P2P write: a kernel on dev0 writes `n` floats to dev1's buffer (P2P), reads local
//   - allreduce: the single-writer allreduce (read both + write both) via TpAllreduce
// Bandwidth is reported in GB/s (bytes moved / time).  For read, bytes = n*4 (the P2P
// read).  For write, bytes = n*4 (the P2P write).  For allreduce, bytes = 2*n*4 (one
// P2P read + one P2P write of n floats).
//
// Requires two visible CUDA devices with P2P.  SKIPs cleanly otherwise.
#include "strata/core/tp_allreduce.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

namespace {
constexpr int THREADS = 256;
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// `dst[i] = src[i]`.  Used for both P2P read (src on peer) and P2P write (dst on peer).
__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

/// Time `launch` on `stream` with cudaEvent (GPU-side, excludes host launch overhead).
/// Returns the median of `iters` runs in microseconds.  Templated so `launch` can be a
/// capturing lambda (the allreduce case needs the TpAllreduce object).
template <typename Launch>
double bench_us(Launch&& launch, cudaStream_t stream, int iters) {
    std::vector<double> us;
    us.reserve(iters);
    cudaEvent_t ev0, ev1;
    gpu(cudaEventCreate(&ev0), "event0");
    gpu(cudaEventCreate(&ev1), "event1");
    // Warmup: first launch pays module-load / P2P-map cost.
    launch();
    gpu(cudaStreamSynchronize(stream), "warmup sync");
    for (int i = 0; i < iters; ++i) {
        gpu(cudaEventRecord(ev0, stream), "ev0");
        launch();
        gpu(cudaEventRecord(ev1, stream), "ev1");
        gpu(cudaEventSynchronize(ev1), "ev1 sync");
        float ms = 0.0f;
        gpu(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
        us.push_back((double) ms * 1000.0);
    }
    gpu(cudaEventDestroy(ev0), "destroy ev0");
    gpu(cudaEventDestroy(ev1), "destroy ev1");
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}
}  // namespace

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) {
        std::fprintf(stderr, "SKIP: only %d CUDA device(s) visible; need two for P2P\n", count);
        return 0;
    }

    // Enable P2P both ways (same sequence as TpAllreduce::open).
    int a = 0, b = 0;
    cudaDeviceCanAccessPeer(&a, 0, 1);
    cudaDeviceCanAccessPeer(&b, 1, 0);
    if (!a || !b) {
        std::fprintf(stderr, "SKIP: no P2P between device 0 and 1\n");
        return 0;
    }
    gpu(cudaSetDevice(0), "set 0");
    cudaError_t e0 = cudaDeviceEnablePeerAccess(1, 0);
    if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
    gpu(e0, "enable peer 0->1");
    gpu(cudaSetDevice(1), "set 1");
    cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
    if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
    gpu(e1, "enable peer 1->0");

    // Sizes: from the allreduce message (10 KiB) up to 8 MiB (well within the running
    // model's VRAM headroom).  n is in floats; bytes = n*4.
    const int64_t sizes[] = {
        2560,            // 10 KiB  - one token's hidden vector
        2560 * 7,        // 70 KiB  - a spec-6 window
        2560 * 64,       // 640 KiB
        2560 * 512,      // 5 MiB
        2560 * 1024,     // 10 MiB -> capped below
    };
    const int64_t MAX_FLOATS = (int64_t) (8 * 1024 * 1024) / 4;  // 8 MiB in floats

    // Buffers: local (dev0) and peer (dev1), each MAX_FLOATS floats.
    float *local = nullptr, *peer = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc((void**) &local, (size_t) MAX_FLOATS * sizeof(float)), "malloc local");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMalloc((void**) &peer, (size_t) MAX_FLOATS * sizeof(float)), "malloc peer");
    // Seed both so the read has real data and the write has a real source.
    gpu(cudaMemset(local, 0, (size_t) MAX_FLOATS * sizeof(float)), "memset local");
    gpu(cudaMemset(peer, 0, (size_t) MAX_FLOATS * sizeof(float)), "memset peer");

    cudaStream_t s0;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream0");

    const int iters = 50;
    std::fprintf(stderr, "P2P bandwidth (dev0 <-> dev1, NVLink), %d iters each, median:\n", iters);
    std::fprintf(stderr, "  %-12s %-14s %-14s %-14s\n", "size", "read (GB/s)", "write (GB/s)", "allreduce (GB/s)");

    for (int64_t n : sizes) {
        if (n > MAX_FLOATS) n = MAX_FLOATS;
        const double bytes = (double) n * sizeof(float);
        // P2P read: dev0 reads `peer` (on dev1), writes `local`.
        const double read_us = bench_us([&] {
            copy_kernel<<<grid_for(n), THREADS, 0, s0>>>(peer, local, n);
        }, s0, iters);
        // P2P write: dev0 reads `local`, writes `peer` (on dev1).
        const double write_us = bench_us([&] {
            copy_kernel<<<grid_for(n), THREADS, 0, s0>>>(local, peer, n);
        }, s0, iters);
        // Allreduce: single-writer (one P2P read + one P2P write of n floats).
        strata::core::TpAllreduce ar;
        std::string err;
        check(ar.open(0, 1, err), "allreduce open");
        const double ar_us = bench_us([&] {
            std::string e;
            check(ar.allreduce(local, peer, n, s0, s0, e), "allreduce");
        }, s0, iters);
        ar.close();

        const double read_gbs = bytes / (read_us * 1e-6) / 1e9;
        const double write_gbs = bytes / (write_us * 1e-6) / 1e9;
        const double ar_gbs = (2.0 * bytes) / (ar_us * 1e-6) / 1e9;
        std::fprintf(stderr, "  %-12.0f %-14.2f %-14.2f %-14.2f  (read %.1f us, write %.1f us, ar %.1f us)\n",
                     bytes / 1024.0, read_gbs, write_gbs, ar_gbs, read_us, write_us, ar_us);
    }

    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaFree(local), "free local");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaFree(peer), "free peer");
    gpu(cudaStreamDestroy(s0), "destroy s0");
    std::fprintf(stderr, "PASS: tp_p2p_bench\n");
    return 0;
}
