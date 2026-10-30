// Two independent CUDA ranks repeatedly run the large-T QSA output projection concurrently.
// This isolates per-device Gemm/cuBLAS state from Prefill's phase-aliased arena.
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int64_t T = 1536;
constexpr int64_t N = 1280;
constexpr int64_t K = 6144;
constexpr int kRounds = 8;
constexpr uint32_t kCanary = 0x7f123456u;

struct Rank {
    int dev = -1;
    cudaStream_t stream = nullptr;
    uint16_t* x = nullptr;
    uint8_t* w = nullptr;
    float* y = nullptr;
    float* ref = nullptr;
    strata::prefill::Gemm gemm;
    std::string err;
};

std::atomic<int> failures{0};

void fail(int rank, const char* what, const char* detail = nullptr) {
    std::fprintf(stderr, "FAIL rank %d: %s%s%s\n", rank, what, detail ? ": " : "", detail ? detail : "");
    ++failures;
}

bool cuda_ok(int rank, cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    fail(rank, what, cudaGetErrorString(e));
    return false;
}

void init_rank(Rank& r, int rank) {
    r.dev = rank;
    if (!cuda_ok(rank, cudaSetDevice(rank), "set device") ||
        !cuda_ok(rank, cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking), "create stream")) return;

    const size_t x_bytes = (size_t) T * K * sizeof(uint16_t);
    const size_t row_bytes = (size_t) (K / 32) * 34;
    const size_t w_bytes = (size_t) N * row_bytes;
    const size_t y_elems = (size_t) T * N;
    if (!cuda_ok(rank, cudaMalloc(&r.x, x_bytes), "malloc X") ||
        !cuda_ok(rank, cudaMalloc(&r.w, w_bytes), "malloc W") ||
        !cuda_ok(rank, cudaMalloc(&r.y, (y_elems + 2) * sizeof(float)), "malloc Y") ||
        !cuda_ok(rank, cudaMalloc(&r.ref, y_elems * sizeof(float)), "malloc reference")) return;

    std::vector<uint16_t> x((size_t) T * K);
    for (size_t i = 0; i < x.size(); ++i) {
        const __half h = __float2half_rn((float) ((int) (i % 31) - 15) / 64.0f);
        std::memcpy(&x[i], &h, sizeof(h));
    }
    std::vector<uint8_t> w(w_bytes, 0);
    for (int64_t row = 0; row < N; ++row) {
        for (int64_t block = 0; block < K / 32; ++block) {
            uint8_t* p = w.data() + ((size_t) row * (K / 32) + block) * 34;
            const __half hd = __float2half_rn(1.0f / 128.0f);
            uint16_t d = 0;
            std::memcpy(&d, &hd, sizeof(d));
            std::memcpy(p, &d, sizeof(d));
            for (int j = 0; j < 32; ++j) p[2 + j] = (uint8_t) ((row + block + j) % 15 - 7);
        }
    }
    if (!cuda_ok(rank, cudaMemcpyAsync(r.x, x.data(), x_bytes, cudaMemcpyHostToDevice, r.stream), "upload X") ||
        !cuda_ok(rank, cudaMemcpyAsync(r.w, w.data(), w_bytes, cudaMemcpyHostToDevice, r.stream), "upload W")) return;
    if (!r.gemm.init(r.stream, N * K, r.err)) {
        fail(rank, "Gemm::init", r.err.c_str());
        return;
    }
    r.gemm.native(r.x, 8, r.w, r.ref, T, N, K);
    cuda_ok(rank, cudaStreamSynchronize(r.stream), "reference sync");
}

void run_rank(Rank& r, int rank, std::atomic<int>& ready, std::atomic<bool>& go) {
    if (!cuda_ok(rank, cudaSetDevice(r.dev), "worker set device")) return;
    ++ready;
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();

    const size_t y_elems = (size_t) T * N;
    for (int round = 0; round < kRounds; ++round) {
        cuda_ok(rank, cudaMemsetAsync(r.y, 0, (y_elems + 2) * sizeof(float), r.stream), "clear output");
        cuda_ok(rank, cudaMemcpyAsync(r.y, &kCanary, sizeof(kCanary), cudaMemcpyHostToDevice, r.stream), "left canary");
        cuda_ok(rank, cudaMemcpyAsync(r.y + y_elems + 1, &kCanary, sizeof(kCanary), cudaMemcpyHostToDevice, r.stream),
                "right canary");
        r.gemm.native(r.x, 8, r.w, r.y + 1, T, N, K);
        if (!cuda_ok(rank, cudaStreamSynchronize(r.stream), "concurrent GEMM sync")) return;

        uint32_t guard[2] = {};
        cuda_ok(rank, cudaMemcpy(&guard[0], r.y, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read left canary");
        cuda_ok(rank, cudaMemcpy(&guard[1], r.y + y_elems + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                "read right canary");
        if (guard[0] != kCanary || guard[1] != kCanary) fail(rank, "output canary changed");

        std::vector<float> got(y_elems), ref(y_elems);
        cuda_ok(rank, cudaMemcpy(got.data(), r.y + 1, y_elems * sizeof(float), cudaMemcpyDeviceToHost), "read output");
        cuda_ok(rank, cudaMemcpy(ref.data(), r.ref, y_elems * sizeof(float), cudaMemcpyDeviceToHost), "read reference");
        for (size_t i = 0; i < y_elems; ++i) {
            if (!std::isfinite(got[i]) || std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
                char detail[160];
                std::snprintf(detail, sizeof(detail), "round %d token %zu col %zu got %.9g ref %.9g", round,
                              i / (size_t) N, i % (size_t) N, got[i], ref[i]);
                fail(rank, "concurrent result mismatch", detail);
                return;
            }
        }
    }
}

void free_rank(Rank& r) {
    cudaSetDevice(r.dev);
    cudaFree(r.x);
    cudaFree(r.w);
    cudaFree(r.y);
    cudaFree(r.ref);
    cudaStreamDestroy(r.stream);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::fprintf(stderr, "SKIP: two CUDA devices required\n");
        return 77;
    }

    Rank ranks[2];
    init_rank(ranks[0], 0);
    init_rank(ranks[1], 1);
    if (failures.load() == 0) {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::thread workers[2] = {
            std::thread(run_rank, std::ref(ranks[0]), 0, std::ref(ready), std::ref(go)),
            std::thread(run_rank, std::ref(ranks[1]), 1, std::ref(ready), std::ref(go)),
        };
        while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        go.store(true, std::memory_order_release);
        workers[0].join();
        workers[1].join();
    }
    free_rank(ranks[0]);
    free_rank(ranks[1]);

    if (failures.load() != 0) return 1;
    std::printf("tp_dense_concurrent_test OK: %d rounds/rank, T=%lld N=%lld K=%lld\n", kRounds,
                (long long) T, (long long) N, (long long) K);
    return 0;
}
