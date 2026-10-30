// include/strata/core/tp_moe_split.hpp - tensor parallelism for the MoE experts' FFN.
//
// The experts are the heaviest per-layer work (512 experts, 10 routed per token), and with `--tp` the
// single-card design left ALL of it on rank 0 while rank 1 waited behind a broadcast.  This splits each
// expert's own FFN across the two cards, and BOTH halves are bit-identical with the single card:
//
// The arithmetic, per expert e and token x:
//
//     h = swiglu(gate_e[0..n_ff) . x, up_e[0..n_ff) . x)        (n_ff wide)
//     y = down_e[0..n_embd) . h                                 (n_embd wide)
//     out = sum_k w_k * down_e_k(h_k) + shared
//
// gate/up: rank r keeps gate rows [r*320, r*320+320) and up rows [r*320, r*320+320) - half the intermediate
// width (320 of 640), each row's full n_embd reduction on this card only.  swiglu is element-wise, so rank r's
// gate/up pair produces exactly h[r*320 + i] for its own i, bit-identically with the single card (same rows,
// same kernel, same block layout).
//
// down: rank r keeps down ROWS [r*1280, r*1280+1280) - half the OUTPUT width, each row reducing the FULL 640
// intermediate terms.  This is the split that keeps the window bit-exact: the old Megatron split cut the
// reduction axis (down columns [r*320, r*320+320), each card summing 320 of the 640 terms and the allreduce
// adding the partials), and (a1+a2)+b is not a1+(a2+b) in float32 - the down's row_dot is a lane-strided
// warp-tree reduction, so halving the term count changes every lane's accumulation groups and the tree shape.
// The n_out split runs the same kernel, the same block layout and the same warp assignment the single card
// uses, on disjoint rows, so every element is bit-identical and the down needs no allreduce of its own (the
// per-layer allreduce then adds the two disjoint output halves, `x + 0.0f == x` bitwise).
//
// The cost of the down's full-width reduction: each card needs the FULL intermediate h (both ranks' gate/up
// halves) before it runs its down.  The window therefore runs the experts in two phases per card -
// gate/up (320 rows) for BOTH group sets, a P2P exchange of the two quantized halves (moe_hq_exchange),
// then the down (1280 rows, full 640 reduction) for both group sets - and stitches the compact [cap][1280]
// down output into the full [cap][2560] buffer (moe_hit_pitch).  See the header of the post's expert block
// in verify.cpp for the handshake.
//
// NO WEIGHT COPY: the blob is left exactly as packed.  A per-rank NativeExpertLayout plus a pointer offset
// addresses the rank's rows:
//
//   * gate/up: both start `r * 320 * gu_row` further into the blob, so ONE offset on the group pointers
//     covers both (up's absolute position is `up_off` and its rows shift with gate's).
//   * down (Ldown): its rows are offset by `r * (n_embd/2) * d_row` - a whole number of rows, no column
//     arithmetic - and its n_embd is halved (1280 of 2560 rows).  `d_row` stays the original packed stride.
//   * the NVFP4 tail {s_gate, s_up, s_down, 0} keeps its own offset, shifted back by the group offset (the
//     down reads s_down from there).
//
// `tp_moe_split_ok` refuses any geometry where the halves are not whole blocks (320 must be a multiple of the
// gate/up block and of 32 for the launch, 1280 a whole number of down rows per block).
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// This rank's view of one expert blob's FFN.
struct TpMoeSplit {
    /// The gate/up layout: n_ff is the LOCAL width (320), gu_row/d_row are the PACKED row widths (unchanged),
    /// and up_off/tail_off already carry the rank's offsets - so the caller pairs it with `grp_ptr[g] + ptr_off`.
    strata::kernels::NativeExpertLayout layout;
    /// The down layout: n_embd is the LOCAL output width (1280), n_ff the FULL width (640), down_off the row
    /// offset minus `down_ptr_off`.  Pair it with `grp_ptr[g] + down_ptr_off`.
    strata::kernels::NativeExpertLayout down;
    /// Bytes to add to every group pointer before the GATE/UP layout applies.
    size_t ptr_off = 0;
    /// Bytes to add to every group pointer before the DOWN layout applies.
    size_t down_ptr_off = 0;
    /// The local intermediate width (n_ff / 2).
    int64_t n_ff_local = 0;
    /// The local output width (n_embd / 2).
    int64_t n_out_local = 0;
};

/// Whether `n_ff` can be split in two for these quantized types: both halves must be a whole number of the
/// down projection's blocks (otherwise a rank's down row would start mid-block) and of the gate/up blocks.
bool tp_moe_split_ok(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, std::string& why);

/// Build rank `rank`'s (0 or 1) view of the layout `native_expert_layout(gu_type, d_type, n_embd, n_ff)`.
/// Returns false (and fills `why`) when the geometry cannot be split; see tp_moe_split_ok.
bool tp_moe_split(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int rank, TpMoeSplit& out,
                  std::string& why);

/// Bytes of `native_expert_grouped` scratch at the LOCAL width (what each rank allocates).
size_t tp_moe_split_scratch_bytes(int64_t cap_entries, const TpMoeSplit& s);

}  // namespace strata::core
