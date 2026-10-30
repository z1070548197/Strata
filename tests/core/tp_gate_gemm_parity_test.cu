#include "strata/core/tp_split_plan.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr int64_t K = 2560;
constexpr int64_t N = 6144;
constexpr int64_t NH = N / 2;
constexpr int64_t GEMM_SCRATCH = 32ll << 20;
constexpr size_t ROW_BYTES = (size_t) (K / 32) * 34;
constexpr size_t FULL_BYTES = (size_t) N * ROW_BYTES;

void fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::vector<uint8_t> read_weight(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) fail("open gate weight");
    const std::streamsize size = in.tellg();
    if (size != (std::streamsize) FULL_BYTES) fail("unexpected gate weight size");
    std::vector<uint8_t> data((size_t) size);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(data.data()), size)) fail("read gate weight");
    return data;
}

void run_case(const std::vector<uint8_t>& full, int64_t T) {
    std::vector<uint8_t> half[2] = {
        std::vector<uint8_t>((size_t) NH * ROW_BYTES),
        std::vector<uint8_t>((size_t) NH * ROW_BYTES),
    };
    for (int rank = 0; rank < 2; ++rank) {
        const size_t written = strata::core::tp_split_q8_0_host(
            full.data(), strata::core::TpMode::Col, K, N, rank, half[rank].data());
        if (written != half[rank].size()) fail("gate host split size");
    }

    std::vector<uint16_t> x((size_t) T * K);
    for (size_t i = 0; i < x.size(); ++i) {
        const uint32_t v = (uint32_t) i * 1664525u + 1013904223u;
        const float f = ((float) ((v >> 8) & 0xffff) / 32768.0f - 1.0f) * 0.125f;
        const __half h = __float2half_rn(f);
        std::memcpy(&x[i], &h, sizeof(h));
    }

    cudaStream_t stream = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    uint16_t* dx = nullptr;
    uint8_t *dw_full = nullptr, *dw_half[2] = {};
    float *dy_full = nullptr, *dy_canonical = nullptr, *dy_half[2] = {};
    ck(cudaMalloc(&dx, x.size() * sizeof(uint16_t)), "malloc X");
    ck(cudaMalloc(&dw_full, full.size()), "malloc full weight");
    ck(cudaMalloc(&dy_full, (size_t) T * N * sizeof(float)), "malloc full output");
    ck(cudaMalloc(&dy_canonical, (size_t) T * N * sizeof(float)), "malloc canonical output");
    ck(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream), "upload X");
    ck(cudaMemcpyAsync(dw_full, full.data(), full.size(), cudaMemcpyHostToDevice, stream), "upload full weight");
    for (int rank = 0; rank < 2; ++rank) {
        ck(cudaMalloc(&dw_half[rank], half[rank].size()), "malloc half weight");
        ck(cudaMalloc(&dy_half[rank], (size_t) T * NH * sizeof(float)), "malloc half output");
        ck(cudaMemcpyAsync(dw_half[rank], half[rank].data(), half[rank].size(), cudaMemcpyHostToDevice, stream),
           "upload half weight");
    }

    std::string err;
    strata::prefill::Gemm gemm;
    if (!gemm.init(stream, GEMM_SCRATCH, err)) {
        std::fprintf(stderr, "FAIL: Gemm::init: %s\n", err.c_str());
        std::exit(1);
    }
    gemm.native(dx, 8, dw_full, dy_full, T, N, K);
    gemm.native(dx, 8, dw_full, dy_canonical, T, NH, K, N);
    gemm.native(dx, 8, dw_full + NH * ROW_BYTES, dy_canonical + NH, T, NH, K, N);
    for (int rank = 0; rank < 2; ++rank)
        gemm.native(dx, 8, dw_half[rank], dy_half[rank], T, NH, K);
    ck(cudaStreamSynchronize(stream), "GEMM sync");

    cudaEvent_t begin = nullptr, end = nullptr;
    ck(cudaEventCreate(&begin), "create timing begin");
    ck(cudaEventCreate(&end), "create timing end");
    constexpr int rounds = 10;
    ck(cudaEventRecord(begin, stream), "record full begin");
    for (int i = 0; i < rounds; ++i) gemm.native(dx, 8, dw_full, dy_full, T, N, K);
    ck(cudaEventRecord(end, stream), "record full end");
    ck(cudaEventSynchronize(end), "sync full timing");
    float full_ms = 0.0f;
    ck(cudaEventElapsedTime(&full_ms, begin, end), "full elapsed");
    ck(cudaEventRecord(begin, stream), "record canonical begin");
    for (int i = 0; i < rounds; ++i) {
        gemm.native(dx, 8, dw_full, dy_canonical, T, NH, K, N);
        gemm.native(dx, 8, dw_full + NH * ROW_BYTES, dy_canonical + NH, T, NH, K, N);
    }
    ck(cudaEventRecord(end, stream), "record canonical end");
    ck(cudaEventSynchronize(end), "sync canonical timing");
    float canonical_ms = 0.0f;
    ck(cudaEventElapsedTime(&canonical_ms, begin, end), "canonical elapsed");

    std::vector<float> got((size_t) T * NH);
    std::vector<float> want((size_t) T * N);
    ck(cudaMemcpy(want.data(), dy_canonical, want.size() * sizeof(float), cudaMemcpyDeviceToHost),
       "canonical output d2h");
    for (int rank = 0; rank < 2; ++rank) {
        ck(cudaMemcpy(got.data(), dy_half[rank], got.size() * sizeof(float), cudaMemcpyDeviceToHost), "half output d2h");
        size_t mismatches = 0;
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t row = 0; row < NH; ++row) {
                const size_t gi = (size_t) t * NH + row;
                const size_t wi = (size_t) t * N + rank * NH + row;
                if (std::memcmp(&got[gi], &want[wi], sizeof(float)) != 0) {
                    if (mismatches++ == 0)
                        std::fprintf(stderr,
                                     "FAIL: T=%lld rank=%d token=%lld row=%lld got=%.9g want=%.9g\n",
                                     (long long) T, rank, (long long) t, (long long) row,
                                     got[gi], want[wi]);
                }
            }
        }
        if (mismatches != 0) {
            std::fprintf(stderr, "FAIL: T=%lld rank=%d mismatches=%zu/%zu\n",
                         (long long) T, rank, mismatches, got.size());
            std::exit(1);
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(dx);
    cudaFree(dw_full);
    cudaFree(dy_full);
    cudaFree(dy_canonical);
    for (int rank = 0; rank < 2; ++rank) {
        cudaFree(dw_half[rank]);
        cudaFree(dy_half[rank]);
    }
    cudaStreamDestroy(stream);
    std::printf("PASS: real gate canonical split-write vs TP halves exact T=%lld, full=%.3f ms canonical=%.3f ms ratio=%.3f\n",
                (long long) T, full_ms / rounds, canonical_ms / rounds, canonical_ms / full_ms);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/blk0-attn-gate.q8_0\n", argv[0]);
        return 2;
    }
    int devices = 0;
    ck(cudaGetDeviceCount(&devices), "device count");
    if (devices < 1) return 77;
    const std::vector<uint8_t> weight = read_weight(argv[1]);
    run_case(weight, 64);
    run_case(weight, 1536);
    return 0;
}
