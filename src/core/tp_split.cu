// src/core/tp_split.cu - Phase 3a: tensor-parallel weight splitting for the dense Q8_0 path.
//
// See include/strata/core/tp_split.hpp for the layout and the two split modes.  A Q8_0 block is 34 bytes
// (2-byte half scale + 32 int8 values) and is only 2-byte aligned (34 is not a multiple of 4), so the block
// copies below move 17 uint16 words per block.  Both splits take whole blocks, so the values are preserved
// bit-for-bit (no re-quantization).
#include "strata/core/tp_split.hpp"
#include "strata/core/tp_split_plan.hpp"

#include <cstdint>
#include <cstdio>

namespace strata::core {

namespace {

constexpr int kQ80Bytes = 34;
constexpr int kQ80U16 = kQ80Bytes / 2;  // 17 uint16 words per block

/// Copy one Q8_0 block (17 uint16 words) from src block `src_block` to dst block `dst_block`.
__device__ __forceinline__ void copy_q80_block(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst,
                                               int src_block, int dst_block) {
    const uint16_t* s = src + (size_t)src_block * kQ80U16;
    uint16_t* d = dst + (size_t)dst_block * kQ80U16;
#pragma unroll
    for (int k = 0; k < kQ80U16; ++k) d[k] = s[k];
}

/// ROW-parallel: for each of n_out rows, copy half `half` of the row's n_in / 32 blocks (the first or second
/// n_in / 64) to the corresponding row of dst.  dst has n_out * (n_in / 64) blocks.
__global__ void tp_row_split_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int n_out,
                                    int nb, int half, int nblocks_dst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nblocks_dst) return;
    const int half_nb = nb / 2;
    const int r = j / half_nb;
    const int i = j % half_nb;
    copy_q80_block(src, dst, r * nb + half * half_nb + i, j);
}

/// COLUMN-parallel: copy half `half` of the n_out rows (the first or second n_out / 2) to dst.  dst has
/// (n_out / 2) * (n_in / 32) blocks.
__global__ void tp_col_split_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int n_out,
                                    int nb, int half, int nblocks_dst) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nblocks_dst) return;
    const int half_rows = n_out / 2;
    const int r = j / nb;
    const int i = j % nb;
    copy_q80_block(src, dst, (half * half_rows + r) * nb + i, j);
}

constexpr int kThreads = 256;
inline int grid_for(int nblocks) { return (nblocks + kThreads - 1) / kThreads; }

}  // namespace

bool tp_row_split_ok(int n_in) { return n_in > 0 && (n_in % 64) == 0; }
bool tp_col_split_ok(int n_out) { return n_out > 0 && (n_out % 2) == 0; }

void tp_split_q8_0_row(const void* src, int n_in, int n_out, int half, void* dst, cudaStream_t stream) {
    if (!tp_row_split_ok(n_in) || half < 0 || half > 1) {
        std::fprintf(stderr, "tp_split_q8_0_row: n_in=%d must be a multiple of 64 and half in {0,1}\n", n_in);
        return;
    }
    const int nb = n_in / 32;          // blocks per row
    const int nblocks_dst = n_out * (nb / 2);
    tp_row_split_kernel<<<grid_for(nblocks_dst), kThreads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst), n_out, nb, half, nblocks_dst);
}

void tp_split_q8_0_col(const void* src, int n_in, int n_out, int half, void* dst, cudaStream_t stream) {
    if (!tp_col_split_ok(n_out) || half < 0 || half > 1) {
        std::fprintf(stderr, "tp_split_q8_0_col: n_out=%d must be even and half in {0,1}\n", n_out);
        return;
    }
    const int nb = n_in / 32;          // blocks per row
    const int nblocks_dst = (n_out / 2) * nb;
    tp_col_split_kernel<<<grid_for(nblocks_dst), kThreads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst), n_out, nb, half, nblocks_dst);
}

// ---- GDN conv weight split (TP) -------------------------------------------------
// The GDN conv weight (ssm_conv1d) is 4 taps x 10240 channels, channel-major:
// full[c*4 + tap].  The local conv (7168 channels = q 2048 + k 2048 + v_half 3072)
// needs the local channels' 4 taps.  rank 0 = channels [0, 7168) (contiguous);
// rank 1 = channels [0, 4096) + [7168, 10240) (q/k + the second v half).
namespace {
constexpr int kLocalConvChannels = 7168;
constexpr int kQkChannels = 4096;   // q 2048 + k 2048

__global__ void tp_local_conv_kernel(const float* __restrict__ full, float* __restrict__ local, int rank) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kLocalConvChannels) return;
    // local channel c -> full channel:
    //   c < 4096: full c (q/k, replicated)
    //   c >= 4096: full 4096 + (c - 4096) + rank * 3072 (v half)
    const int full_c = (c < kQkChannels) ? c : (kQkChannels + (c - kQkChannels) + rank * (kLocalConvChannels - kQkChannels));
    const float4 w = *reinterpret_cast<const float4*>(full + (size_t) full_c * 4);
    *reinterpret_cast<float4*>(local + (size_t) c * 4) = w;
}
}  // namespace

void tp_local_conv_weights(const float* full, int rank, float* local, cudaStream_t stream) {
    if (rank < 0 || rank > 1) {
        std::fprintf(stderr, "tp_local_conv_weights: rank must be 0 or 1\n");
        return;
    }
    tp_local_conv_kernel<<<grid_for(kLocalConvChannels), kThreads, 0, stream>>>(full, local, rank);
}

}  // namespace strata::core
