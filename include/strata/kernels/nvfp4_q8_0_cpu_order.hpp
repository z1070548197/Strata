// include/strata/kernels/nvfp4_q8_0_cpu_order.hpp - NVFP4 expert rows on the GPU, bit for bit what the CPU pool
// computes (nvfp4_256_rows / nvfp4_256_gu_rows, i.e. ggml-cpu's AVX2 dot product).
//
// The engine computes a routed expert on the CPU pool when it is a miss and on the GPU when it is resident. On the
// native NVFP4 pack those two paths used to disagree, so changing which experts were resident changed the output
// tokens. These produce the CPU pool's arithmetic exactly: same Q8_0 blocks, same fp32 per-block scales, same FMA
// order, same horizontal-sum tree, and a bit-exact transcription of glibc's expf for the SiLU.
//
// Parity is asserted by nvfp4_q8_0_cpu_parity (tests/cuda/), which runs both against a live CPU build.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// out[t][row] = scale * dot(weight row, activation of token t); rows [r0, r1), tokens [0, n_tok).
/// `act` holds n_tok contiguous Q8_0 rows of act_stride bytes; `w` holds rows of row_bytes.
/// Both must be device pointers. `stream` may be null (the default stream). Asynchronous.
void nvfp4_q8_0_rows(const uint8_t* w, size_t row_bytes, int n, const uint8_t* act, int act_stride, int n_tok,
                     float* out, int r0, int r1, float scale, void* stream = nullptr);

/// ff[t][row] = silu(s_gate * g) * (s_up * u) for rows [r0, r1), the expression nvfp4_256_gu_rows ends with.
/// `blob` holds the gate rows at its start and the up rows at up_off, each gu_row bytes.
void nvfp4_q8_0_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const uint8_t* act, int act_stride,
                        int n_tok, float* ff, int r0, int r1, float s_gate, float s_up, void* stream = nullptr);

/// ---------------------------------------------------------------------------------------------------------
/// GROUPED: the native expert path's shape, with the CPU pool's arithmetic.
///
/// `native_expert_grouped` (iq_kernels.cu) takes a `block_q8_1` activation and computes each row with a 16- or
/// 8-lane sub-warp reduction.  The CPU pool takes `block_q8_0` and keeps eight fp32 accumulators in a fixed FMA
/// order.  The two do not produce the same number, which is why a resident expert and a missed one disagreed.
/// These take the POOL's activation contract and reproduce its arithmetic, so a resident expert's row is the
/// number the pool would have produced.
///
/// Layout: `NativeExpertLayout` (iq_kernels.hpp), which `TpMoeSplit` also expresses its halves with - so the
/// full and the per-rank calls are the same kernel with a different geometry, `grp_ptr` and activation width.
///
/// **ACTIVATIONS ARE PER ENTRY, INDEXED BY `ent_tok[]`.**  `act` holds `n_tok` contiguous Q8_0 rows of
/// `act_stride` bytes and entry `e` reads row `ent_tok[e]`.  The pool quantizes every token of a job
/// separately, so a windowed call passes T rows and the same `ent_tok` the existing grouped kernels take.
///
/// `n_groups` and `grp_start` are DEVICE pointers, as in `native_expert_grouped`; `cap_entries` bounds the entry
/// indices the kernels will write.

/// gu: for group g, entry e in [grp_start[g], grp_start[g+1]), row r in [0, n_ff): the blob at `grp_ptr[g]`
///     holds gate row r at r*gu_row and up row r at up_off + r*gu_row.  Writes the pool's composition,
///     ff[e*n_ff + r] = silu(s_gate*g) * (s_up*u), the same expression `nvfp4_256_gu_rows` ends with, so a
///     grouped gu is directly comparable to the pool's per-expert call.
///     `tail_off` must be non-zero (an NVFP4 blob always has it); `n_embd` must be a multiple of 64.
///
/// `sc_lo` is an OPTIONAL fp32 per-block scale array (n_tok rows of `sc_stride` floats; block 2*il and 2*il+1).
/// **On a native NVFP4 pack you do NOT need it.** The pool's own activation path is `traits()->from_float` into
/// ggml's `block_q8_0` (src/kernels/cpu/native_expert.cpp:113), and the pool's NVFP4 row kernels then read that
/// block's fp16 `d` (`h2f(y[ib].d)`, src/kernels/cpu/nvfp4_avx2.cpp:88) - the same value this kernel reads when
/// `sc_lo` is null. Passing null is therefore already bit-identical to the pool, which is what the parity test
/// asserts. The parameter exists for the OTHER activation contract (`quantize_q8_0_scaled`, used by the non-native
/// S2 path), where a hit must multiply by an fp32 `ActQ::scale` that a block's fp16 `d` cannot represent; there the
/// two differ by 4.761e-04 relative on 80 of 80 chunks. Null (the default) keeps the previous behaviour exactly.
void nvfp4_q8_0_grouped_gu(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                           const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, int64_t n_embd,
                           int64_t n_ff, size_t gu_row, size_t up_off, size_t tail_off, const uint8_t* act,
                           int act_stride, float* ff, void* stream = nullptr, const float* sc_lo = nullptr,
                           int sc_stride = 0);

/// down: for group g, entry e, row r in [0, n_embd): the blob at `grp_ptr[g]` holds down row r at
///       down_off + r*d_row, reducing the FULL n_ff terms against entry e's activation row and multiplying by
///       the tail's s_down.  Writes out[ent_dst[e]*n_embd + r].  `act_stride` is one row in bytes.
///       `sc_lo`/`sc_stride`: see the gu comment above.
void nvfp4_q8_0_grouped_down(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                             const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                             int64_t cap_entries, int64_t n_embd, int64_t n_ff, size_t d_row, size_t down_off,
                             size_t tail_off, const uint8_t* act, int act_stride, float* out,
                             void* stream = nullptr, const float* sc_lo = nullptr, int sc_stride = 0);

}  // namespace strata::kernels
