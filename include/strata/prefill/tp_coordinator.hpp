#pragma once

#include "strata/core/tp_allreduce.hpp"
#include "strata/prefill/prefill.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

namespace strata::prefill {

class TpCoordinator final : public GdnYExchange {
public:
    TpCoordinator() = default;
    ~TpCoordinator() override;
    TpCoordinator(const TpCoordinator&) = delete;
    TpCoordinator& operator=(const TpCoordinator&) = delete;

    bool open(int device0, int device1, std::string& err);
    void reset(int64_t epoch);
    void abort(const std::string& reason);
    bool run(Prefill& rank0, Prefill& rank1, const int64_t* tokens, int64_t n, int64_t pos0,
             std::string& err);

    bool stitch_y(int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                  const float* local_y, int64_t local_ld, float* full_y, int64_t full_ld,
                  void* stream, std::string& err) override;
    bool stitch_output(int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                       const float* local_output, int64_t local_ld,
                       float* full_output, int64_t full_ld,
                       void* stream, std::string& err) override;
    bool exchange_hq(int rank, int64_t chunk_pos, int64_t layer,
                     const void* local_hq, int64_t local_bytes,
                     void* full_hq, int64_t full_bytes,
                     void* stream, std::string& err) override;
    bool broadcast_residual(int rank, int64_t chunk_pos, int64_t layer,
                            float* residual, int64_t count,
                            void* stream, std::string& err) override;

private:
    struct Arrival {
        bool ready = false;
        const void* src = nullptr;
        void* dst = nullptr;
        int64_t src_ld = 0;
        int64_t dst_ld = 0;
        cudaStream_t stream = nullptr;
        cudaEvent_t produced = nullptr;
    };

    bool exchange(int phase, int rank, int64_t chunk_pos, int64_t layer, int64_t T,
                  const float* src, int64_t src_ld, float* dst, int64_t dst_ld,
                  int64_t cols, cudaStream_t stream, std::string& err);
    bool exchange_bytes(int phase, int rank, int64_t chunk_pos, int64_t layer,
                        const void* src, int64_t src_bytes, void* dst, int64_t dst_bytes,
                        cudaStream_t stream, std::string& err);
    bool rendezvous(int phase, int rank, int64_t chunk_pos, int64_t layer, int64_t rows,
                    const void* src, int64_t src_ld, void* dst, int64_t dst_ld,
                    cudaStream_t stream, std::string& err);
    void clear_round();

    core::TpAllreduce transport_;
    int devices_[2] = {-1, -1};
    std::mutex mutex_;
    std::condition_variable cv_;
    Arrival arrival_[2];
    int64_t epoch_ = 0;
    int64_t chunk_pos_ = -1;
    int64_t layer_ = -1;
    int64_t rows_ = 0;
    int phase_ = -1;
    int departed_ = 0;
    bool transfer_done_ = false;
    bool aborted_ = false;
    std::string failure_;
};

}  // namespace strata::prefill
