// tests/core/tp_weight_split_test.cu - Phase 2a: weight splitting prototype.
//
// Proves the core TP mechanism with synthetic weights (real model dimensions):
//   Test 1: Column-parallel (qkv pattern: 2560 x 10240, split N in half)
//   Test 2: Row-parallel + allreduce (ssm_out pattern: 6144 x 2560, split K in half)
//   Test 3: GDN v-head split (6144 x 2560, split N in half, 24+24 heads)
//   Test 4: Timing - allreduce latency for 2560 floats (per-layer cost)

#include "strata/core/tp_allreduce.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok, const char* what, const std::string& err = "") {
    if (!ok) { std::fprintf(stderr, "FAIL: %s%s%s\n", what, err.empty() ? "" : " - ", err.c_str()); std::exit(1); }
}
static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void gemv_kernel(const float* __restrict__ x, const float* __restrict__ W,
                            float* __restrict__ y, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += x[k] * W[k * N + n];
    y[n] = acc;
}

static float prand(int64_t i) {
    uint64_t x = (uint64_t)i * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return (float)(x & 0xFFFFFF) / 8388608.0f - 1.0f;
}

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) { std::fprintf(stderr, "SKIP: only %d device(s)\n", count); return 0; }

    strata::core::TpAllreduce ar;
    std::string err;
    if (!ar.open(0, 1, err)) { std::fprintf(stderr, "SKIP: no P2P (%s)\n", err.c_str()); return 0; }

    cudaStream_t s0, s1;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "s0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "s1");

    // Test 1: Column-parallel (qkv: K=2560, N=10240)
    {
        const int64_t K = 2560, N = 10240, Nh = N / 2;
        std::fprintf(stderr, "=== Test 1: Column-parallel (qkv %lld x %lld) ===\n", (long long)K, (long long)N);
        std::vector<float> W(K * N), x(K);
        for (int64_t i = 0; i < K * N; ++i) W[i] = prand(i);
        for (int64_t i = 0; i < K; ++i) x[i] = prand(i + 999999);
        std::vector<float> y_ref(N);
        for (int64_t n = 0; n < N; ++n) { float a = 0; for (int64_t k = 0; k < K; ++k) a += x[k] * W[k * N + n]; y_ref[n] = a; }
        std::vector<float> W0(K * Nh), W1(K * Nh);
        for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < Nh; ++n) { W0[k*Nh+n] = W[k*N+n]; W1[k*Nh+n] = W[k*N+Nh+n]; }
        float *dx, *dW0, *dy0, *dW1, *dy1;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMalloc((void**)&dx, K*sizeof(float)), "dx");
        gpu(cudaMalloc((void**)&dW0, K*Nh*sizeof(float)), "dW0");
        gpu(cudaMalloc((void**)&dy0, Nh*sizeof(float)), "dy0");
        // H2D on explicit streams (legacy default stream does not order with
        // non-blocking streams - a real TP integration hazard).
        gpu(cudaMemcpyAsync(dx, x.data(), K*sizeof(float), cudaMemcpyHostToDevice, s0), "cx");
        gpu(cudaMemcpyAsync(dW0, W0.data(), K*Nh*sizeof(float), cudaMemcpyHostToDevice, s0), "cW0");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMalloc((void**)&dW1, K*Nh*sizeof(float)), "dW1");
        gpu(cudaMalloc((void**)&dy1, Nh*sizeof(float)), "dy1");
        gpu(cudaMemcpyAsync(dW1, W1.data(), K*Nh*sizeof(float), cudaMemcpyHostToDevice, s1), "cW1");
        // x is identical on both GPUs: upload to dev1 directly from host on s1
        // (avoids a cross-stream P2P read of dev0's dx, which s0's H2D does not order).
        float *dx1b;
        gpu(cudaMalloc((void**)&dx1b, K*sizeof(float)), "dx1b");
        gpu(cudaMemcpyAsync(dx1b, x.data(), K*sizeof(float), cudaMemcpyHostToDevice, s1), "cx1b");
        int th = 256, bl = (int)((Nh + th - 1) / th);
        gpu(cudaSetDevice(0), "set 0");
        gemv_kernel<<<bl, th, 0, s0>>>(dx, dW0, dy0, (int)K, (int)Nh);
        gpu(cudaGetLastError(), "g0");
        gpu(cudaSetDevice(1), "set 1");
        gemv_kernel<<<bl, th, 0, s1>>>(dx1b, dW1, dy1, (int)K, (int)Nh);
        gpu(cudaGetLastError(), "g1");
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaStreamSynchronize(s0), "s0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaStreamSynchronize(s1), "s1");
        std::vector<float> y0(Nh), y1(Nh);
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaMemcpy(y0.data(), dy0, Nh*sizeof(float), cudaMemcpyDeviceToHost), "y0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaMemcpy(y1.data(), dy1, Nh*sizeof(float), cudaMemcpyDeviceToHost), "y1");
        float e0 = 0, e1 = 0;
        for (int64_t n = 0; n < Nh; ++n) { e0 = std::max(e0, std::fabs(y0[n]-y_ref[n])); e1 = std::max(e1, std::fabs(y1[n]-y_ref[Nh+n])); }
        std::fprintf(stderr, "  GPU0: max_err=%.6e  GPU1: max_err=%.6e\n", e0, e1);
        check(e0 < 1e-2f && e1 < 1e-2f, "col-parallel");
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaFree(dx), "f"); gpu(cudaFree(dW0), "f"); gpu(cudaFree(dy0), "f");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaFree(dW1), "f"); gpu(cudaFree(dy1), "f");
    }

    // Test 2: Row-parallel + allreduce (ssm_out: K=6144, N=2560)
    {
        const int64_t K = 6144, N = 2560, Kh = K / 2;
        std::fprintf(stderr, "\n=== Test 2: Row-parallel + allreduce (ssm_out %lld x %lld) ===\n", (long long)K, (long long)N);
        std::vector<float> W(K * N), x(K);
        for (int64_t i = 0; i < K * N; ++i) W[i] = prand(i + 500000);
        for (int64_t i = 0; i < K; ++i) x[i] = prand(i + 888888);
        std::vector<float> y_ref(N);
        for (int64_t n = 0; n < N; ++n) { float a = 0; for (int64_t k = 0; k < K; ++k) a += x[k] * W[k * N + n]; y_ref[n] = a; }
        std::vector<float> W0(Kh*N), W1(Kh*N), x0(Kh), x1(Kh);
        for (int64_t k = 0; k < Kh; ++k) { x0[k]=x[k]; x1[k]=x[Kh+k]; for (int64_t n=0;n<N;++n){W0[k*N+n]=W[k*N+n]; W1[k*N+n]=W[(Kh+k)*N+n];} }
        float *dx0,*dW0,*dy0,*dx1,*dW1,*dy1;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMalloc((void**)&dx0, Kh*sizeof(float)), "dx0");
        gpu(cudaMalloc((void**)&dW0, Kh*N*sizeof(float)), "dW0");
        gpu(cudaMalloc((void**)&dy0, N*sizeof(float)), "dy0");
        gpu(cudaMemcpyAsync(dx0, x0.data(), Kh*sizeof(float), cudaMemcpyHostToDevice, s0), "cx0");
        gpu(cudaMemcpyAsync(dW0, W0.data(), Kh*N*sizeof(float), cudaMemcpyHostToDevice, s0), "cW0");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMalloc((void**)&dx1, Kh*sizeof(float)), "dx1");
        gpu(cudaMalloc((void**)&dW1, Kh*N*sizeof(float)), "dW1");
        gpu(cudaMalloc((void**)&dy1, N*sizeof(float)), "dy1");
        // NOTE: H2D copies must use explicit streams (not the legacy default) so
        // they order correctly with the non-blocking compute streams.
        gpu(cudaMemcpyAsync(dx1, x1.data(), Kh*sizeof(float), cudaMemcpyHostToDevice, s1), "cx1");
        gpu(cudaMemcpyAsync(dW1, W1.data(), Kh*N*sizeof(float), cudaMemcpyHostToDevice, s1), "cW1");
        int th = 256, bl = (int)((N + th - 1) / th);
        // Cross-device sync via events (the production TP pattern):
        //   GEMV0 on s0 (dev0) and GEMV1 on s1 (dev1) run independently.
        //   allreduce (single-writer on dev0) must wait for BOTH to finish:
        //     - s0 is already ordered after GEMV0 (same stream)
        //     - record event on s1 after GEMV1, make s0 wait on it
        //   After allreduce, dev1 must see the updated dy1:
        //     - record event on s0 after allreduce, make s1 wait on it
        cudaEvent_t ev1, ev2;
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming), "ev1");
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming), "ev2");
        gpu(cudaSetDevice(0), "set 0");
        gemv_kernel<<<bl, th, 0, s0>>>(dx0, dW0, dy0, (int)Kh, (int)N);
        gpu(cudaGetLastError(), "g0");
        gpu(cudaSetDevice(1), "set 1");
        gemv_kernel<<<bl, th, 0, s1>>>(dx1, dW1, dy1, (int)Kh, (int)N);
        gpu(cudaGetLastError(), "g1");
        gpu(cudaEventRecord(ev1, s1), "rec ev1");
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaStreamWaitEvent(s0, ev1, 0), "wait ev1");
        ar.allreduce(dy0, dy1, N, s0, s1, err);
        check(err.empty(), "allreduce", err);
        gpu(cudaEventRecord(ev2, s0), "rec ev2");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaStreamWaitEvent(s1, ev2, 0), "wait ev2");
        gpu(cudaEventDestroy(ev1), "del ev1");
        gpu(cudaEventDestroy(ev2), "del ev2");
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaStreamSynchronize(s0), "s0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaStreamSynchronize(s1), "s1");
        std::vector<float> y0(N), y1(N);
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaMemcpy(y0.data(), dy0, N*sizeof(float), cudaMemcpyDeviceToHost), "y0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaMemcpy(y1.data(), dy1, N*sizeof(float), cudaMemcpyDeviceToHost), "y1");
        float e0 = 0, e1 = 0;
        for (int64_t n = 0; n < N; ++n) { e0 = std::max(e0, std::fabs(y0[n]-y_ref[n])); e1 = std::max(e1, std::fabs(y1[n]-y_ref[n])); }
        std::fprintf(stderr, "  GPU0: max_err=%.6e  GPU1: max_err=%.6e\n", e0, e1);
        check(e0 < 1e-2f && e1 < 1e-2f, "row-parallel+allreduce");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 200; ++i) ar.allreduce(dy0, dy1, N, s0, s1, err);
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaStreamSynchronize(s0), "st");
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200;
        std::fprintf(stderr, "\n=== Timing ===\n  Allreduce %lld floats: %.1f us/layer\n", (long long)N, us);
        std::fprintf(stderr, "  48 layers: %.2f ms/window (%.1f%% of 41.57ms)\n", 48.0*us/1000.0, 100.0*48.0*us/1000.0/41.57);
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaFree(dx0), "f"); gpu(cudaFree(dW0), "f"); gpu(cudaFree(dy0), "f");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaFree(dx1), "f"); gpu(cudaFree(dW1), "f"); gpu(cudaFree(dy1), "f");
    }

    // Test 3: GDN v-head split (out-proj 6144 x 2560, column-parallel)
    {
        const int64_t K = 6144, N = 2560, Nh = N / 2;
        std::fprintf(stderr, "\n=== Test 3: GDN v-head split (out-proj %lld x %lld) ===\n", (long long)K, (long long)N);
        std::vector<float> W(K * N), x(K);
        for (int64_t i = 0; i < K * N; ++i) W[i] = prand(i + 700000);
        for (int64_t i = 0; i < K; ++i) x[i] = prand(i + 666666);
        std::vector<float> y_ref(N);
        for (int64_t n = 0; n < N; ++n) { float a = 0; for (int64_t k = 0; k < K; ++k) a += x[k] * W[k * N + n]; y_ref[n] = a; }
        std::vector<float> W0(K*Nh), W1(K*Nh);
        for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < Nh; ++n) { W0[k*Nh+n]=W[k*N+n]; W1[k*Nh+n]=W[k*N+Nh+n]; }
        float *dx,*dW0,*dy0,*dW1,*dy1;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMalloc((void**)&dx, K*sizeof(float)), "dx");
        gpu(cudaMalloc((void**)&dW0, K*Nh*sizeof(float)), "dW0");
        gpu(cudaMalloc((void**)&dy0, Nh*sizeof(float)), "dy0");
        gpu(cudaMemcpyAsync(dx, x.data(), K*sizeof(float), cudaMemcpyHostToDevice, s0), "cx");
        gpu(cudaMemcpyAsync(dW0, W0.data(), K*Nh*sizeof(float), cudaMemcpyHostToDevice, s0), "cW0");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMalloc((void**)&dW1, K*Nh*sizeof(float)), "dW1");
        gpu(cudaMalloc((void**)&dy1, Nh*sizeof(float)), "dy1");
        gpu(cudaMemcpyAsync(dW1, W1.data(), K*Nh*sizeof(float), cudaMemcpyHostToDevice, s1), "cW1");
        float *dx1b;
        gpu(cudaMalloc((void**)&dx1b, K*sizeof(float)), "dx1b");
        gpu(cudaMemcpyAsync(dx1b, x.data(), K*sizeof(float), cudaMemcpyHostToDevice, s1), "cx1b");
        int th = 256, bl = (int)((Nh + th - 1) / th);
        gpu(cudaSetDevice(0), "set 0");
        gemv_kernel<<<bl, th, 0, s0>>>(dx, dW0, dy0, (int)K, (int)Nh);
        gpu(cudaGetLastError(), "g0");
        gpu(cudaSetDevice(1), "set 1");
        gemv_kernel<<<bl, th, 0, s1>>>(dx1b, dW1, dy1, (int)K, (int)Nh);
        gpu(cudaGetLastError(), "g1");
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaStreamSynchronize(s0), "s0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaStreamSynchronize(s1), "s1");
        std::vector<float> y0(Nh), y1(Nh);
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaMemcpy(y0.data(), dy0, Nh*sizeof(float), cudaMemcpyDeviceToHost), "y0");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaMemcpy(y1.data(), dy1, Nh*sizeof(float), cudaMemcpyDeviceToHost), "y1");
        float e0 = 0, e1 = 0;
        for (int64_t n = 0; n < Nh; ++n) { e0 = std::max(e0, std::fabs(y0[n]-y_ref[n])); e1 = std::max(e1, std::fabs(y1[n]-y_ref[Nh+n])); }
        std::fprintf(stderr, "  GPU0 (heads 0-23): max_err=%.6e\n  GPU1 (heads 24-47): max_err=%.6e\n", e0, e1);
        check(e0 < 1e-2f && e1 < 1e-2f, "gdn v-head split");
        gpu(cudaSetDevice(0), "set 0"); gpu(cudaFree(dx), "f"); gpu(cudaFree(dW0), "f"); gpu(cudaFree(dy0), "f");
        gpu(cudaSetDevice(1), "set 1"); gpu(cudaFree(dW1), "f"); gpu(cudaFree(dy1), "f");
    }

    gpu(cudaSetDevice(0), "set 0"); gpu(cudaStreamDestroy(s0), "ds0");
    gpu(cudaSetDevice(1), "set 1"); gpu(cudaStreamDestroy(s1), "ds1");
    ar.close();
    std::fprintf(stderr, "\nALL TESTS PASSED\n");
    return 0;
}
