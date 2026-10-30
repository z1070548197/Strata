// include/strata/core/tp_split.hpp - Phase 3a: tensor-parallel weight splitting for the dense Q8_0 path.
//
// The dense weights (attention / GDN / QSA / HC) are Q8_0 (GGML type 8): 32 elements per block, 34 bytes per
// block (a 2-byte half-precision scale `d` + 32 int8 quantized values `qs`).  A weight matrix of shape
// (n_out rows x n_in cols) is stored as a flat array of n_out * (n_in / 32) Q8_0 blocks, row-major, with n_in
// the contiguous reduction axis: block index = row * (n_in / 32) + kbx.
//
// Tensor parallelism splits a dense GEMV y = W x two ways (both validated in Phase 2a/2f with real weights and
// the real native_mmvq kernel):
//   * COLUMN-parallel (split n_out, the row axis): each card holds n_out / 2 rows and computes n_out / 2 output
//     floats; the two halves are concatenated (no allreduce).  Because a row's blocks are contiguous, this is a
//     plain contiguous split of the flat block array.
//   * ROW-parallel (split n_in, the reduction axis): each card holds n_in / 2 columns (half of every row's
//     blocks) and computes a partial sum over all n_out outputs; the two partial sums are allreduced.  This is an
//     interleaved split: for each row, card 0 takes the first n_in / 64 blocks and card 1 the next n_in / 64.
//
// Both splits are exact (no re-quantization): a Q8_0 block is self-contained (its scale covers exactly its 32
// elements), so taking whole blocks preserves the values bit-for-bit.  n_in must be a multiple of 64 (so n_in / 2
// is a whole number of 32-element blocks) and n_out must be even for column-parallel.
//
// These are one-time weight-load operations (not per-token), so they run on the device with a small kernel and a
// stream; the source and destination may be on the same device or, with P2P enabled, on different devices.
#pragma once

#include <cuda_runtime.h>

namespace strata::core {

/// True when `n_in` can be row-parallel split into two whole-block halves (a multiple of 64).
bool tp_row_split_ok(int n_in);
/// True when `n_out` can be column-parallel split into two whole-row halves (even).
bool tp_col_split_ok(int n_out);

/// ROW-parallel split of a Q8_0 weight (n_out x n_in): copy half `half` (0 or 1) of the n_in reduction blocks to
/// `dst`.  `src` is the full weight (n_out * (n_in / 32) blocks); `dst` must hold n_out * (n_in / 64) blocks.
/// For each row, half 0 takes blocks [0, n_in / 64) and half 1 takes blocks [n_in / 64, n_in / 32).  The result is
/// a (n_out x n_in / 2) Q8_0 weight.  Runs on the current device on `stream`.
void tp_split_q8_0_row(const void* src, int n_in, int n_out, int half, void* dst, cudaStream_t stream);

/// COLUMN-parallel split of a Q8_0 weight (n_out x n_in): copy half `half` (0 or 1) of the n_out rows to `dst`.
/// `src` is the full weight (n_out * (n_in / 32) blocks); `dst` must hold (n_out / 2) * (n_in / 32) blocks.
/// Half 0 takes rows [0, n_out / 2) and half 1 takes rows [n_out / 2, n_out) - a contiguous split of the flat
/// block array.  The result is a (n_out / 2 x n_in) Q8_0 weight.  Runs on the current device on `stream`.
void tp_split_q8_0_col(const void* src, int n_in, int n_out, int half, void* dst, cudaStream_t stream);

/// GDN conv weight (ssm_conv1d, 4 taps x 10240 channels, channel-major) split for this rank: the local 7168
/// channels (q 2048 + k 2048 + v_half 3072).  `full` is the full conv weight (10240 * 4 floats); `local` receives
/// the local conv weight (7168 * 4 floats).  rank 0 = channels [0, 7168) (contiguous); rank 1 = channels
/// [0, 4096) + [7168, 10240) (q/k + the second v half).  One-time weight-load op; runs on the current device on
/// `stream`.
void tp_local_conv_weights(const float* full, int rank, float* local, cudaStream_t stream);

} // namespace strata::core
