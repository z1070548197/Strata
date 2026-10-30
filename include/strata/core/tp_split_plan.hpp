#pragma once

// ============================================================================
// TP split plan: the single source of truth for how each dense Q8_0 weight is
// split across the two tensor-parallel cards.  Used by BOTH
//   - NativeDense::load (which half of the GGUF tensor to upload per rank)
//   - Verifier::record_window_tp (which per-rank n_in/n_out to feed the GEMV)
// so the two can never disagree about a tensor's shape.
//
// The dense Q8_0 weights are the GDN/QSA projections (attn_qkv / attn_gate /
// ssm_out for GDN layers, attn_q / attn_k / attn_v / attn_output for QSA
// layers).  The MoE experts (NVFP4) are NOT here - they are expert-parallel
// (Milestone B) and, in Milestone A, replicated.  The BF16/F32 pack-arena
// weights (hc_*, ssm_alpha/beta, ple_*, output_hc_*) are replicated.
//
// Split modes (a weight matrix is n_out rows x n_in cols, n_in the contiguous
// reduction axis, stored row-major as Q8_0 blocks of 32 elements):
//   Replicate - both cards hold the full matrix (small, or needed whole).
//   Col       - column-parallel: split the n_out (row) axis; each card holds a
//               contiguous half of the rows and computes half the output.
//               No allreduce (the outputs are disjoint).
//   Row       - row-parallel: split the n_in (reduction) axis; each card holds
//               half the reduction and computes a PARTIAL sum of the full
//               output.  The two partials are allreduced (single-writer).
//   Qkv       - the GDN qkv projection: q/k are REPLICATED (both cards hold all
//               q and k heads, because the v-head pairing h % h_k needs every
//               q/k head on every card) and v is split by head (each card holds
//               half the v heads).  Per-card output = q + k + v_half.
//
// rank is 0 or 1.  For Col/Row the per-rank half is the first half (rank 0) or
// the second half (rank 1) of the split axis.
// ============================================================================

#include <cstdint>
#include <string>

namespace strata::core {

enum class TpMode : int {
    Replicate = 0,   ///< both cards hold the full matrix
    Col = 1,         ///< column-parallel: split n_out (rows), no allreduce
    Row = 2,         ///< row-parallel: split n_in (reduction), allreduce
    Qkv = 3,         ///< GDN qkv: q/k replicated + v split by head
};

/// The per-rank view of one dense weight: its split mode and the per-rank
/// n_in / n_out to feed the GEMV.  `full_n_in` / `full_n_out` are the model's
/// (unsplit) dimensions, kept so a caller can reason about the whole matrix.
struct TpSplit {
    TpMode mode = TpMode::Replicate;
    int full_n_in = 0;    ///< the model's n_in (reduction axis)
    int full_n_out = 0;   ///< the model's n_out (row axis)
    int n_in = 0;         ///< this rank's n_in (== full_n_in for Col/Replicate/Qkv)
    int n_out = 0;        ///< this rank's n_out (== full_n_out for Row/Replicate)
    /// True when this tensor's per-rank GEMV output must be allreduced with the
    /// peer's (Row mode, and the Qkv v-part is folded into the out-proj Row).
    bool allreduce() const { return mode == TpMode::Row; }
};

/// Rank-local execution view used by both decode and large-T prefill.  Col and
/// Qkv split output rows while preserving the complete reduction axis; Row is
/// intentionally rejected by TP prefill because adding partial reductions is
/// not bit-identical to the single-kernel accumulation order.
struct TpProjection {
    TpSplit split;
    int rank = 0;
    int output_row0 = 0;
    bool replicated = true;
    bool split_output_rows = false;
    bool special_qkv = false;
};

TpProjection tp_projection_for(const std::string& name, int rank);
bool tp_prefill_projection_ok(const TpProjection& projection, std::string* why = nullptr);

/// The peer rank's rows of a split dense projection, for the card that runs the PROMPT path.
///
/// A prompt chunk is computed on ONE card, but under `--tp` that card holds only its half of the matrix.  The full
/// matrix's rows come from two places, and they are contiguous in the SAME order the single card reads them:
///   rows [0, n_own)             this card's half, from the start of its upload (a Col half is the leading half of
///                               the rows, and a Qkv half is [q; k; v_half] = the full [q; k; v]'s leading rows);
///   rows [n_own, n_own + rows)  the peer's half, starting at row `src_row0` of ITS upload (Col: 0; Qkv: the peer's
///                               v_half, which sits after the replicated q and k it does not need to supply).
/// Dequantizing both into one scratch in row order and running ONE GEMM over the assembled matrix is what keeps the
/// prompt path bit-identical to the single card's.
///
/// Row mode is NOT representable this way (both cards hold all n_out rows and half the reduction), so this struct
/// never describes one: the prompt path refuses a Row tensor loudly rather than read half a reduction as if it were
/// the whole.
struct TpPeerRows {
    const void* blocks = nullptr;   ///< the peer's native (Q8_0) upload of its half
    int64_t src_row0 = 0;           ///< the first row of `blocks` that belongs to the full matrix
    int64_t n_own = 0;              ///< rows this card's own half supplies (the full matrix's leading rows)
    int64_t rows = 0;               ///< rows the peer supplies (the full matrix ends at n_own + rows)
};

/// The row mapping above, from the split plan alone - `blocks` comes back null (the caller fills it from the peer's
/// weight table).  False, with `why` set, for a name that is not split, for Row mode, and for an `n_out` that is not
/// the full matrix's.  On success the two ranges cover the full matrix exactly: n_own + rows == n_out.
bool tp_peer_rows_for(const std::string& name, int64_t n_out, TpPeerRows& out, std::string* why = nullptr);

/// Resolve the split for a dense weight by its canonical tensor name (e.g.
/// "blk.11.attn_q.weight") and the model geometry.  Returns Replicate for any
/// name that is not a dense GDN/QSA projection (the caller then keeps the full
/// weight).  `rank` is 0 or 1.
///
/// Geometry (from the pack index, authoritative):
///   N = 2560 (n_embd)
///   GDN:  attn_qkv 2560 x 10240 (q 2048 + k 2048 + v 6144),
///         attn_gate 2560 x 6144, ssm_out 6144 x 2560
///   QSA:  attn_q 2560 x 12288 (NH*2*HD, NH=48, HD=128),
///         attn_k 2560 x 512 (NKV*HD, NKV=4), attn_v 2560 x 512,
///         attn_output 6144 x 2560 (NH*HD x N)
TpSplit tp_split_for(const std::string& name, int rank);

/// The per-rank byte count of a Q8_0 tensor under the split plan (for
/// NativeDense::load to size its cudaMalloc).  Q8_0: 34 bytes per 32-element
/// block, n_in the contiguous axis, so bytes = (n_in/32) * 34 * n_out.
/// Returns 0 when the name is not a dense Q8_0 projection.
uint64_t tp_q8_0_bytes(const std::string& name, int rank);

/// Host-side Q8_0 split: produce this rank's tensor from the full tensor.
/// `full` is the full Q8_0 tensor (n_out rows x n_in cols, row-major, 34-byte
/// blocks). `half` receives this rank's tensor. Returns bytes written.
/// For Replicate this is a full copy (caller should skip calling it).
size_t tp_split_q8_0_host(const uint8_t* full, TpMode mode, int n_in, int n_out,
                          int rank, uint8_t* half);

/// The per-rank GDN conv channel count (7168 = q 2048 + k 2048 + v_half 3072).
constexpr int kTpGdnConvChannels = 7168;

} // namespace strata::core
