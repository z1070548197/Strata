#include "strata/prefill/tp_coordinator.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(bool ok, const char* what, const std::string& err = {}) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s%s%s\n", what, err.empty() ? "" : ": ", err.c_str());
        std::exit(1);
    }
}
void gpu(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}
struct Buffers {
    float* local = nullptr;
    float* full = nullptr;
    cudaStream_t stream = nullptr;
};
}  // namespace

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "device count");
    if (count < 2) {
        std::fprintf(stderr, "SKIP: need two CUDA devices\n");
        return 0;
    }

    strata::prefill::TpCoordinator coordinator;
    std::string err;
    if (!coordinator.open(0, 1, err)) {
        std::fprintf(stderr, "SKIP: %s\n", err.c_str());
        return 0;
    }

    constexpr int64_t rows = 7, cols = 13, local_ld = cols, full_ld = 31;
    Buffers b[2];
    std::vector<float> host[2];
    for (int rank = 0; rank < 2; ++rank) {
        host[rank].resize((size_t) rows * local_ld);
        for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < local_ld; ++c)
            host[rank][(size_t) r * local_ld + c] = 1000.0f * rank + 10.0f * r + (float) c;
        gpu(cudaSetDevice(rank), "set device");
        gpu(cudaStreamCreateWithFlags(&b[rank].stream, cudaStreamNonBlocking), "create stream");
        gpu(cudaMalloc(&b[rank].local, host[rank].size() * sizeof(float)), "allocate local");
        gpu(cudaMalloc(&b[rank].full, (size_t) rows * full_ld * sizeof(float)), "allocate full");
        gpu(cudaMemcpyAsync(b[rank].local, host[rank].data(), host[rank].size() * sizeof(float),
                            cudaMemcpyHostToDevice, b[rank].stream), "upload local");
        gpu(cudaMemsetAsync(b[rank].full, 0, (size_t) rows * full_ld * sizeof(float), b[rank].stream), "clear full");
    }

    const auto run_round = [&](bool output, int64_t chunk, int64_t layer) {
        bool ok[2] = {false, false};
        std::string errors[2];
        std::thread threads[2];
        for (int rank = 0; rank < 2; ++rank) {
            threads[rank] = std::thread([&, rank] {
                gpu(cudaSetDevice(rank), "thread set device");
                if (chunk == 30 && rank == 1) std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (output) {
                    ok[rank] = coordinator.stitch_output(rank, chunk, layer, rows,
                                                         b[rank].local, local_ld,
                                                         b[rank].full, full_ld,
                                                         b[rank].stream, errors[rank]);
                } else {
                    ok[rank] = coordinator.stitch_y(rank, chunk, layer, rows,
                                                    b[rank].local, local_ld,
                                                    b[rank].full, full_ld,
                                                    b[rank].stream, errors[rank]);
                }
            });
        }
        threads[0].join();
        threads[1].join();
        check(ok[0], "rank 0 exchange", errors[0]);
        check(ok[1], "rank 1 exchange", errors[1]);
        for (int rank = 0; rank < 2; ++rank) {
            std::vector<float> got((size_t) rows * full_ld);
            gpu(cudaSetDevice(rank), "read set device");
            gpu(cudaMemcpy(got.data(), b[rank].full, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "read full");
            for (int64_t r = 0; r < rows; ++r) for (int peer = 0; peer < 2; ++peer)
                for (int64_t c = 0; c < cols; ++c) {
                    const float want = host[peer][(size_t) r * local_ld + c];
                    check(got[(size_t) r * full_ld + peer * cols + c] == want, "stitched row value");
                }
        }
    };

    run_round(false, 23, 4);
    run_round(true, 23, 4);
    run_round(false, 30, 5);

    constexpr int64_t hq_bytes = 257;
    uint8_t* hq_local[2] = {};
    uint8_t* hq_peer[2] = {};
    std::vector<uint8_t> hq_host[2];
    for (int rank = 0; rank < 2; ++rank) {
        hq_host[rank].resize((size_t) hq_bytes);
        for (int64_t i = 0; i < hq_bytes; ++i)
            hq_host[rank][(size_t) i] = (uint8_t) (rank * 97 + i);
        gpu(cudaSetDevice(rank), "HQ set device");
        gpu(cudaMalloc(&hq_local[rank], (size_t) hq_bytes), "allocate local HQ");
        gpu(cudaMalloc(&hq_peer[rank], (size_t) hq_bytes), "allocate peer HQ");
        gpu(cudaMemcpyAsync(hq_local[rank], hq_host[rank].data(), (size_t) hq_bytes,
                            cudaMemcpyHostToDevice, b[rank].stream), "upload local HQ");
        gpu(cudaMemsetAsync(hq_peer[rank], 0, (size_t) hq_bytes, b[rank].stream), "clear peer HQ");
    }
    bool hq_ok[2] = {};
    std::string hq_err[2];
    std::thread hq_thread[2];
    for (int rank = 0; rank < 2; ++rank) {
        hq_thread[rank] = std::thread([&, rank] {
            gpu(cudaSetDevice(rank), "HQ thread set device");
            hq_ok[rank] = coordinator.exchange_hq(
                rank, 31, 6, hq_local[rank], hq_bytes,
                hq_peer[rank], hq_bytes, b[rank].stream, hq_err[rank]);
        });
    }
    for (auto& thread : hq_thread) thread.join();
    for (int rank = 0; rank < 2; ++rank) {
        check(hq_ok[rank], "HQ exchange", hq_err[rank]);
        std::vector<uint8_t> got((size_t) hq_bytes);
        gpu(cudaSetDevice(rank), "read HQ set device");
        gpu(cudaMemcpy(got.data(), hq_peer[rank], got.size(), cudaMemcpyDeviceToHost), "read peer HQ");
        check(got == hq_host[1 - rank], "peer HQ half");
    }

    for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < cols; ++c)
        host[0][(size_t) r * local_ld + c] = 5000.0f + 10.0f * r + (float) c;
    gpu(cudaSetDevice(0), "broadcast upload set device");
    gpu(cudaMemcpyAsync(b[0].local, host[0].data(), host[0].size() * sizeof(float),
                        cudaMemcpyHostToDevice, b[0].stream), "upload broadcast residual");
    bool broadcast_ok[2] = {};
    std::string broadcast_err[2];
    std::thread broadcast_thread[2];
    for (int rank = 0; rank < 2; ++rank) {
        broadcast_thread[rank] = std::thread([&, rank] {
            gpu(cudaSetDevice(rank), "broadcast thread set device");
            broadcast_ok[rank] = coordinator.broadcast_residual(
                rank, 32, 6, b[rank].local, rows * cols,
                b[rank].stream, broadcast_err[rank]);
        });
    }
    for (auto& thread : broadcast_thread) thread.join();
    check(broadcast_ok[0], "rank 0 residual broadcast", broadcast_err[0]);
    check(broadcast_ok[1], "rank 1 residual broadcast", broadcast_err[1]);
    std::vector<float> residual(host[0].size());
    gpu(cudaSetDevice(1), "read broadcast set device");
    gpu(cudaMemcpy(residual.data(), b[1].local, residual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "read broadcast residual");
    check(residual == host[0], "broadcast residual value");

    coordinator.reset(9);
    bool mismatch_ok[2] = {true, true};
    std::string mismatch_err[2];
    std::thread t0([&] {
        gpu(cudaSetDevice(0), "mismatch set 0");
        mismatch_ok[0] = coordinator.stitch_y(0, 40, 7, rows, b[0].local, local_ld,
                                               b[0].full, full_ld, b[0].stream, mismatch_err[0]);
    });
    std::thread t1([&] {
        gpu(cudaSetDevice(1), "mismatch set 1");
        mismatch_ok[1] = coordinator.stitch_output(1, 40, 7, rows, b[1].local, local_ld,
                                                    b[1].full, full_ld, b[1].stream, mismatch_err[1]);
    });
    t0.join();
    t1.join();
    check(!mismatch_ok[0] && !mismatch_ok[1], "phase mismatch wakes both ranks");

    for (int rank = 0; rank < 2; ++rank) {
        gpu(cudaSetDevice(rank), "cleanup set device");
        gpu(cudaFree(hq_local[rank]), "free local HQ");
        gpu(cudaFree(hq_peer[rank]), "free peer HQ");
        gpu(cudaFree(b[rank].local), "free local");
        gpu(cudaFree(b[rank].full), "free full");
        gpu(cudaStreamDestroy(b[rank].stream), "destroy stream");
    }
    std::fprintf(stderr, "PASS: tp_prefill_coordinator_test\n");
    return 0;
}
