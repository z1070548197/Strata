#include "strata/prefill/tp_coordinator.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>

namespace strata::prefill {
namespace {

struct OnDevice {
    int previous = 0;
    explicit OnDevice(int device) {
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
    }
    ~OnDevice() {
        int current = 0;
        cudaGetDevice(&current);
        if (current != previous) cudaSetDevice(previous);
    }
};

std::string cuda_error(const char* what, cudaError_t status) {
    return std::string("TP prefill coordinator: ") + what + ": " + cudaGetErrorString(status);
}

// STRATA_TP_XCH_STATS=1: how much wall time the collective boundary costs.  Every exchange()/exchange_bytes()
// is a host rendezvous that ends with cudaStreamSynchronize on both ranks' compute streams, and the MoE path
// calls it twice per expert group (the HQ exchange and the output stitch).  Printing the call count and the
// time inside them prices a restructuring from data; the counters are off (and cost nothing) by default.
struct XchStats {
    bool on = false;
    std::atomic<long long> calls{0};
    std::atomic<long long> ns{0};
    // By phase: 0 = GDN y stitch, 1 = output stitch (GDN ssm_out / QSA attn_output / MoE columns), 2 = MoE HQ,
    // 3 = residual broadcast.  One line per phase says which collective to collapse next.
    std::atomic<long long> phase_calls[4] = {};
    std::atomic<long long> phase_ns[4] = {};
    XchStats() {
        const char* v = std::getenv("STRATA_TP_XCH_STATS");
        on = v != nullptr && v[0] != '0';
    }
    ~XchStats() {
        const long long c = calls.load();
        if (!on || c == 0) return;
        const double ms = (double) ns.load() / 1e6;
        std::fprintf(stderr, "strata tp: %lld collective exchanges left unreported, %.1f ms inside them\n",
                     c, ms);
        std::fflush(stderr);
    }
};

XchStats g_xch_stats;

// Per chunk, not per process: the first prompt of a load streams experts off the disk and its collectives absorb
// that wait, so a process total cannot say what a warm prompt pays.  run() covers exactly one chunk, so printing
// and resetting here separates the cold and warm regimes.
void report_chunk(int64_t rows, double wall_ms) {
    if (!g_xch_stats.on) return;
    const long long calls = g_xch_stats.calls.exchange(0);
    const long long ns = g_xch_stats.ns.exchange(0);
    if (calls == 0) return;
    std::fprintf(stderr, "strata tp: chunk %lld rows: %lld exchanges, %.1f ms inside them (%.3f ms each), "
                         "chunk wall %.1f ms (%.1f%% inside)\n",
                 (long long) rows, calls, (double) ns / 1e6, (double) ns / 1e6 / (double) calls, wall_ms,
                 100.0 * (double) ns / 1e6 / wall_ms);
    static const char* kName[4] = {"gdn y", "output", "moe hq", "broadcast"};
    for (int p = 0; p < 4; ++p) {
        const long long pc = g_xch_stats.phase_calls[p].exchange(0);
        const long long pns = g_xch_stats.phase_ns[p].exchange(0);
        if (pc == 0) continue;
        std::fprintf(stderr, "strata tp:   %-9s %5lld calls, %8.1f ms inside (%.3f ms each)\n", kName[p], pc,
                     (double) pns / 1e6, (double) pns / 1e6 / (double) pc);
    }
    std::fflush(stderr);
}

struct XchTimer {
    const bool on = g_xch_stats.on;
    const int phase;
    std::chrono::steady_clock::time_point t0;
    explicit XchTimer(int p) : phase(p) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~XchTimer() {
        if (!on) return;
        ++g_xch_stats.calls;
        const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
        g_xch_stats.ns += ns;
        if (phase >= 0 && phase < 4) {
            ++g_xch_stats.phase_calls[phase];
            g_xch_stats.phase_ns[phase] += ns;
        }
    }
};

}  // namespace

TpCoordinator::~TpCoordinator() {
    for (int rank = 0; rank < 2; ++rank) {
        if (arrival_[rank].produced != nullptr && devices_[rank] >= 0) {
            const OnDevice on(devices_[rank]);
            cudaEventDestroy(arrival_[rank].produced);
        }
    }
}

bool TpCoordinator::open(int device0, int device1, std::string& err) {
    if (!transport_.open(device0, device1, err)) return false;
    devices_[0] = device0;
    devices_[1] = device1;
    for (int rank = 0; rank < 2; ++rank) {
        const OnDevice on(devices_[rank]);
        const cudaError_t status = cudaEventCreateWithFlags(&arrival_[rank].produced, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            err = cuda_error("creating producer event", status);
            abort(err);
            return false;
        }
    }
    reset(1);
    return true;
}

void TpCoordinator::clear_round() {
    for (auto& arrival : arrival_) {
        arrival.ready = false;
        arrival.src = nullptr;
        arrival.dst = nullptr;
        arrival.src_ld = 0;
        arrival.dst_ld = 0;
        arrival.stream = nullptr;
    }
    chunk_pos_ = -1;
    layer_ = -1;
    rows_ = 0;
    phase_ = -1;
    departed_ = 0;
    transfer_done_ = false;
}

void TpCoordinator::reset(int64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++epoch_;
    if (epoch > epoch_) epoch_ = epoch;
    aborted_ = false;
    failure_.clear();
    clear_round();
    cv_.notify_all();
}

void TpCoordinator::abort(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    failure_ = reason.empty() ? "TP prefill coordinator aborted" : reason;
    cv_.notify_all();
}

bool TpCoordinator::run(Prefill& rank0, Prefill& rank1, const int64_t* tokens, int64_t n, int64_t pos0,
                        std::string& err) {
    static std::atomic<int64_t> next_epoch{1};
    const auto chunk_start = std::chrono::steady_clock::now();
    reset(next_epoch.fetch_add(1, std::memory_order_relaxed));
    std::string rank_error[2];
    bool ok[2] = {false, false};
    std::thread peer([&] {
        try {
            const OnDevice on(devices_[1]);
            ok[1] = rank1.run(tokens, n, pos0, rank_error[1]);
        } catch (const std::exception& exception) {
            rank_error[1] = std::string("rank 1 prefill exception: ") + exception.what();
        } catch (...) {
            rank_error[1] = "rank 1 prefill exception";
        }
        if (!ok[1]) abort(rank_error[1].empty() ? "rank 1 prefill failed" : rank_error[1]);
    });
    try {
        const OnDevice on(devices_[0]);
        ok[0] = rank0.run(tokens, n, pos0, rank_error[0]);
    } catch (const std::exception& exception) {
        rank_error[0] = std::string("rank 0 prefill exception: ") + exception.what();
    } catch (...) {
        rank_error[0] = "rank 0 prefill exception";
    }
    if (!ok[0]) abort(rank_error[0].empty() ? "rank 0 prefill failed" : rank_error[0]);
    peer.join();
    if (ok[0] && ok[1]) {
        if (rank0.stopped_at() != rank1.stopped_at()) {
            err = "TP prefill coordinator: ranks stopped at different chunk boundaries";
            abort(err);
            return false;
        }
        report_chunk(n, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - chunk_start).count());
        return true;
    }
    err = !rank_error[0].empty() ? rank_error[0]
                                 : !rank_error[1].empty() ? rank_error[1] : failure_;
    return false;
}

bool TpCoordinator::stitch_y(int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                             const float* local_y, int64_t local_ld, float* full_y, int64_t full_ld,
                             void* stream, std::string& err) {
    return exchange(0, rank, chunk_pos, layer, T, local_y, local_ld, full_y, full_ld,
                    local_ld, (cudaStream_t) stream, err);
}

bool TpCoordinator::stitch_output(int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                                  const float* local_output, int64_t local_ld,
                                  float* full_output, int64_t full_ld,
                                  void* stream, std::string& err) {
    return exchange(1, rank, chunk_pos, layer, T, local_output, local_ld, full_output, full_ld,
                    local_ld, (cudaStream_t) stream, err);
}

bool TpCoordinator::exchange_hq(int rank, int64_t chunk_pos, int64_t layer,
                                const void* local_hq, int64_t local_bytes,
                                void* full_hq, int64_t full_bytes,
                                void* stream, std::string& err) {
    return exchange_bytes(2, rank, chunk_pos, layer, local_hq, local_bytes,
                          full_hq, full_bytes, (cudaStream_t) stream, err);
}

bool TpCoordinator::broadcast_residual(int rank, int64_t chunk_pos, int64_t layer,
                                       float* residual, int64_t count,
                                       void* stream, std::string& err) {
    return exchange_bytes(3, rank, chunk_pos, layer,
                          rank == 0 ? residual : nullptr,
                          count * (int64_t) sizeof(float), residual,
                          count * (int64_t) sizeof(float),
                          (cudaStream_t) stream, err);
}

bool TpCoordinator::exchange(int phase, int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                             const float* src, int64_t src_ld, float* dst, int64_t dst_ld,
                             int64_t cols, cudaStream_t stream, std::string& err) {
    const XchTimer xch_timer(phase);
    if (rank < 0 || rank > 1 || T <= 0 || cols <= 0 || src == nullptr || dst == nullptr ||
        src_ld < cols || dst_ld < 2 * cols || stream == nullptr) {
        err = "TP prefill coordinator: invalid exchange arguments";
        abort(err);
        return false;
    }
    {
        const OnDevice on(devices_[rank]);
        const cudaError_t status = cudaEventRecord(arrival_[rank].produced, stream);
        if (status != cudaSuccess) {
            err = cuda_error("recording producer event", status);
            abort(err);
            return false;
        }
    }

    bool leader = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return aborted_ || departed_ == 0; });
        if (aborted_) {
            err = failure_;
            return false;
        }
        if (arrival_[rank].ready) {
            err = "TP prefill coordinator: rank arrived twice";
            aborted_ = true;
            failure_ = err;
            cv_.notify_all();
            return false;
        }
        if (!arrival_[1 - rank].ready) {
            chunk_pos_ = chunk_pos;
            layer_ = layer;
            rows_ = T;
            phase_ = phase;
        } else if (chunk_pos_ != chunk_pos || layer_ != layer || rows_ != T || phase_ != phase ||
                   arrival_[1 - rank].src_ld != src_ld || arrival_[1 - rank].dst_ld != dst_ld) {
            err = "TP prefill coordinator: rank phase or shape mismatch";
            aborted_ = true;
            failure_ = err;
            cv_.notify_all();
            return false;
        }
        arrival_[rank].ready = true;
        arrival_[rank].src = src;
        arrival_[rank].dst = dst;
        arrival_[rank].src_ld = src_ld;
        arrival_[rank].dst_ld = dst_ld;
        arrival_[rank].stream = stream;
        leader = arrival_[1 - rank].ready;
        cv_.notify_all();
        if (!leader) {
            cv_.wait(lock, [&] { return aborted_ || transfer_done_; });
            if (aborted_) {
                err = failure_;
                return false;
            }
        }
    }

    if (leader) {
        std::string transfer_error;
        for (int r = 0; r < 2 && transfer_error.empty(); ++r) {
            const OnDevice on(devices_[r]);
            const cudaError_t status = cudaEventSynchronize(arrival_[r].produced);
            if (status != cudaSuccess) transfer_error = cuda_error("waiting for producer", status);
        }
        for (int r = 0; r < 2 && transfer_error.empty(); ++r) {
            const OnDevice on(devices_[r]);
            const cudaError_t local = cudaMemcpy2DAsync(
                static_cast<float*>(arrival_[r].dst) + (size_t) r * cols,
                (size_t) arrival_[r].dst_ld * sizeof(float),
                arrival_[r].src, (size_t) arrival_[r].src_ld * sizeof(float),
                (size_t) cols * sizeof(float), (size_t) T, cudaMemcpyDeviceToDevice, arrival_[r].stream);
            if (local != cudaSuccess) transfer_error = cuda_error("copying local rows", local);
        }
        if (transfer_error.empty() &&
            !transport_.copy_rows(const_cast<float*>(static_cast<const float*>(arrival_[0].src)),
                                  arrival_[0].src_ld, static_cast<float*>(arrival_[1].dst),
                                  arrival_[1].dst_ld, T, cols, true,
                                  arrival_[0].stream, transfer_error)) {
        }
        if (transfer_error.empty() &&
            !transport_.copy_rows(const_cast<float*>(static_cast<const float*>(arrival_[1].src)),
                                  arrival_[1].src_ld,
                                  static_cast<float*>(arrival_[0].dst) + cols,
                                  arrival_[0].dst_ld, T, cols, false,
                                  arrival_[1].stream, transfer_error)) {
        }
        for (int r = 0; r < 2 && transfer_error.empty(); ++r) {
            const OnDevice on(devices_[r]);
            const cudaError_t status = cudaStreamSynchronize(arrival_[r].stream);
            if (status != cudaSuccess) transfer_error = cuda_error("waiting for stitched rows", status);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!transfer_error.empty()) {
            aborted_ = true;
            failure_ = transfer_error;
        } else {
            transfer_done_ = true;
        }
        cv_.notify_all();
        if (aborted_) {
            err = failure_;
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++departed_;
        if (departed_ == 2) clear_round();
        cv_.notify_all();
    }
    return true;
}

bool TpCoordinator::exchange_bytes(int phase, int rank, int64_t chunk_pos, int64_t layer,
                                   const void* src, int64_t src_bytes, void* dst, int64_t dst_bytes,
                                   cudaStream_t stream, std::string& err) {
    const XchTimer xch_timer(phase);
    const bool broadcast = phase == 3;
    if (rank < 0 || rank > 1 || src_bytes <= 0 || dst_bytes <= 0 || dst == nullptr ||
        stream == nullptr || (!broadcast && src == nullptr) ||
        (!broadcast && dst_bytes != src_bytes) ||
        (broadcast && dst_bytes != src_bytes) || (broadcast && rank == 0 && src == nullptr)) {
        err = "TP prefill coordinator: invalid byte exchange arguments";
        abort(err);
        return false;
    }
    {
        const OnDevice on(devices_[rank]);
        const cudaError_t status = cudaEventRecord(arrival_[rank].produced, stream);
        if (status != cudaSuccess) {
            err = cuda_error("recording byte producer event", status);
            abort(err);
            return false;
        }
    }

    bool leader = false;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return aborted_ || departed_ == 0; });
        if (aborted_) { err = failure_; return false; }
        if (arrival_[rank].ready) {
            err = "TP prefill coordinator: rank arrived twice";
            aborted_ = true;
            failure_ = err;
            cv_.notify_all();
            return false;
        }
        if (!arrival_[1 - rank].ready) {
            chunk_pos_ = chunk_pos;
            layer_ = layer;
            rows_ = 1;
            phase_ = phase;
        } else if (chunk_pos_ != chunk_pos || layer_ != layer || phase_ != phase ||
                   arrival_[1 - rank].src_ld != src_bytes || arrival_[1 - rank].dst_ld != dst_bytes) {
            err = "TP prefill coordinator: rank byte phase or shape mismatch";
            aborted_ = true;
            failure_ = err;
            cv_.notify_all();
            return false;
        }
        arrival_[rank].ready = true;
        arrival_[rank].src = src;
        arrival_[rank].dst = dst;
        arrival_[rank].src_ld = src_bytes;
        arrival_[rank].dst_ld = dst_bytes;
        arrival_[rank].stream = stream;
        leader = arrival_[1 - rank].ready;
        cv_.notify_all();
        if (!leader) {
            cv_.wait(lock, [&] { return aborted_ || transfer_done_; });
            if (aborted_) { err = failure_; return false; }
        }
    }

    if (leader) {
        std::string transfer_error;
        for (int r = 0; r < 2 && transfer_error.empty(); ++r) {
            const OnDevice on(devices_[r]);
            const cudaError_t status = cudaEventSynchronize(arrival_[r].produced);
            if (status != cudaSuccess) transfer_error = cuda_error("waiting for byte producer", status);
        }
        if (transfer_error.empty() && broadcast) {
            const OnDevice on(devices_[0]);
            const cudaError_t status = cudaMemcpyPeerAsync(
                arrival_[1].dst, devices_[1], arrival_[0].src, devices_[0],
                (size_t) src_bytes, arrival_[0].stream);
            if (status != cudaSuccess) transfer_error = cuda_error("broadcasting residual", status);
        } else if (transfer_error.empty()) {
            const OnDevice on0(devices_[0]);
            const cudaError_t to1 = cudaMemcpyPeerAsync(
                arrival_[1].dst, devices_[1], arrival_[0].src, devices_[0],
                (size_t) src_bytes, arrival_[0].stream);
            if (to1 != cudaSuccess) transfer_error = cuda_error("copying rank 0 HQ", to1);
            if (transfer_error.empty()) {
                const OnDevice on1(devices_[1]);
                const cudaError_t to0 = cudaMemcpyPeerAsync(
                    arrival_[0].dst, devices_[0], arrival_[1].src, devices_[1],
                    (size_t) src_bytes, arrival_[1].stream);
                if (to0 != cudaSuccess) transfer_error = cuda_error("copying rank 1 HQ", to0);
            }
        }
        for (int r = 0; r < 2 && transfer_error.empty(); ++r) {
            const OnDevice on(devices_[r]);
            const cudaError_t status = cudaStreamSynchronize(arrival_[r].stream);
            if (status != cudaSuccess) transfer_error = cuda_error("waiting for byte exchange", status);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!transfer_error.empty()) {
            aborted_ = true;
            failure_ = transfer_error;
        } else {
            transfer_done_ = true;
        }
        cv_.notify_all();
        if (aborted_) { err = failure_; return false; }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++departed_;
        if (departed_ == 2) clear_round();
        cv_.notify_all();
    }
    return true;
}

}  // namespace strata::prefill
