// include/strata/core/tp_allreduce.hpp - two-GPU allreduce over P2P (NVLink or PCIe).
//
// Phase 1 of the tensor-parallel work: the communication primitive.  A tensor-parallel
// layer split needs, at the end of every layer, the two halves' partial sums added
// together so that BOTH cards hold the full result.  With P2P (NVLink on the 2080 Ti
// pair, or a PCIe bridge) a card can read AND write the other's device memory directly
// - no host staging, no pinned RAM bounce.
//
// The allreduce is single-writer: one device (dev_a) runs a single kernel that reads
// both buffers (buf_a local, buf_b via P2P), computes the element-wise sum, and writes
// it back to both.  After the kernel, buf_a == buf_b == original_a + original_b.
//
// Why single-writer instead of a symmetric two-phase ring: a two-phase design (each
// device reads the other, then each adds) has a cross-device race - dev_a's add writes
// buf_a while dev_b's read is still reading buf_a, and the two devices have no ordering
// between them without an explicit barrier.  The single-writer form removes the race
// entirely: only dev_a's kernel runs, each element is touched by exactly one thread
// (read both, write both), so there is no read/write overlap.  The cost is that dev_a
// does all the work - but the allreduce message is tiny (H=2560 floats = 10 KiB per
// token) and the real tensor-parallel work (the layer compute) is what is split across
// the two cards, so an unbalanced 10 KiB glue operation is the right trade.
//
// This is OPT-IN and does not touch the default path: the engine's layer-split decode
// window (verify.cpp) is unchanged.  The class is standalone - construct, open,
// allreduce, close - and is exercised by tests/core/tp_allreduce_test.cu.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace strata::core {

/// Two-GPU allreduce over P2P.  After `allreduce`, both buffers hold the element-wise
/// sum of their original contents.  The P2P path is detected and enabled in `open`
/// (the same `cudaDeviceCanAccessPeer` + `cudaDeviceEnablePeerAccess` sequence that
/// `peer_experts.cpp` uses for the `--peer-device` expert tier).
///
/// Usage:
///   TpAllreduce ar;
///   ar.open(0, 1, err);          // enable P2P both ways
///   ar.allreduce(buf0, buf1, n, stream0, stream1, err);
///   // buf0[i] == buf1[i] == orig0[i] + orig1[i]
class TpAllreduce {
public:
    TpAllreduce() = default;
    ~TpAllreduce();
    TpAllreduce(const TpAllreduce&) = delete;
    TpAllreduce& operator=(const TpAllreduce&) = delete;

    /// Detect and enable P2P between `dev_a` and `dev_b`.  Returns false (with `err`)
    /// if either device is missing or P2P is unavailable.  The `STRATA_TP_NO_P2P`
    /// environment variable (value != 0) forces P2P off - a test knob that mirrors
    /// `STRATA_PEER_NO_P2P` in peer_experts.cpp.
    bool open(int dev_a, int dev_b, std::string& err);

    /// Release P2P.  Safe to call multiple times.
    void close();

    /// Whether P2P was successfully enabled.
    bool p2p() const { return p2p_; }
    int dev_a() const { return dev_a_; }
    int dev_b() const { return dev_b_; }

    /// Allreduce: `buf_a` (on dev_a) and `buf_b` (on dev_b) both become
    /// `buf_a + buf_b` (element-wise, `n` floats).  A single kernel on dev_a reads and
    /// writes both buffers, so there is no cross-device race.  `stream_a` must be a
    /// stream on dev_a; `stream_b` is accepted for interface symmetry but unused.
    /// The caller must order dev_b's subsequent work after this kernel completes
    /// (e.g. sync `stream_a`, or record an event on dev_b).  Returns false (with `err`)
    /// on a CUDA error.
    bool allreduce(float* buf_a, float* buf_b, int64_t n,
                   cudaStream_t stream_a, cudaStream_t stream_b,
                   std::string& err);

    /// Fused forms (P1): the per-layer handshake and the reduction in ONE launch, the way vLLM's custom
    /// all-reduce folds its barrier into the reduction kernel.  Without them a layer pays three launches
    /// (wait_b, allreduce, raise_a) on rank 0 and two (raise_b, wait_a) on rank 1.
    ///
    /// `allreduce_wait_raise` (rank 0): spin until `*wait_flag >= wait_value`, sum `buf_a`/`buf_b` into both,
    /// then set `*raise_flag = raise_value`.
    bool allreduce_wait_raise(float* buf_a, float* buf_b, int64_t n, cudaStream_t stream_a,
                              const uint32_t* wait_flag, uint32_t wait_value,
                              uint32_t* raise_flag, uint32_t raise_value, std::string& err);

    /// `publish_wait` (rank 1): set `*raise_flag = raise_value` (this card's partial is complete), then spin
    /// until `*wait_flag >= wait_value` (rank 0's sum has landed in this card's buffer over P2P).
    /// `stream` must be a stream on dev_b.
    bool publish_wait(uint32_t* raise_flag, uint32_t raise_value,
                      const uint32_t* wait_flag, uint32_t wait_value, cudaStream_t stream, std::string& err);

    /// Rank 1's side: record the card publish_wait runs on.  No P2P is enabled - rank 1 never touches the
    /// peer's memory; it only raises its own flag and spins on rank 0's.  The P2P pair belongs to rank 0's
    /// object (open() there).
    void set_peer_device(int dev_b) { dev_b_ = dev_b; }

    /// broadcast: copy `src_a` (on dev_a) to `dst_b` (on dev_b) over P2P.  One kernel on dev_a, so the write to
    /// dev_b is ordered after dev_a's stream work.  Used where one card computes a value BOTH cards need (the
    /// MoE residual, which a TP window computes once on rank 0 instead of twice).  The caller must order dev_b's
    /// readers after this kernel (a handshake flag, as for allreduce).
    bool broadcast(float* src_a, float* dst_b, int64_t n, cudaStream_t stream_a, std::string& err);

    /// The reverse: rank 1's card copies ITS buffer to the peer's over P2P.  Used by the full-width GDN
    /// out-projection, where rank 0 needs rank 1's value-head half.  `stream_b` must be on dev_b.
    bool broadcast_to_a(float* src_b, float* dst_a, int64_t n, cudaStream_t stream_b, std::string& err);

    /// Strided peer copy for chunked prefill.  Copies `rows` rows of `cols`
    /// floats while preserving independent source/destination pitches.  The
    /// launch device is selected by `from_a`; no mapped-host spin flags are
    /// involved, so the host coordinator must order producer and consumer.
    bool copy_rows(float* src, int64_t src_ld, float* dst, int64_t dst_ld,
                   int64_t rows, int64_t cols, bool from_a, cudaStream_t stream,
                   std::string& err);

private:
    int dev_a_ = -1, dev_b_ = -1;
    bool p2p_ = false;
};

/// `dst[i] = src[i] + off` over `n` group pointers, on the CURRENT device's stream.  The experts' FFN split
/// addresses each rank's half of a packed blob this way (tp_moe_split.hpp); a plan's pointer array is device
/// memory, so a rank's copy is shifted right after it is staged.  Not a P2P operation.
bool tp_shift_group_ptrs(const unsigned long long* src, unsigned long long* dst, int64_t n,
                         unsigned long long off, cudaStream_t stream, std::string& err);

}  // namespace strata::core
