// include/strata/core/kv_redistribute.hpp - TP_RESULTS §8.3 step 2: the sequence state a LAYER-SPLIT prefill
// leaves behind, moved into the shape a TENSOR-PARALLEL decode reads.
//
// THE PHASE PLAN'S ONE STATE TRANSITION.  Layer-split prefill wins above ~26K tokens and TP decode wins there
// too (§5.1.1, §8), so the two modes run one after the other on the same sequence - and they hold that
// sequence in different shapes:
//
//   * QSA - REPLICATED in both modes, but a layer-split stage holds only the layers it runs (`qsa_ord0 ..
//     +qsa_alloc`, global ordinals), so each TP rank has to receive all 12 states' bytes from whichever stage
//     owns the layer.  The bytes are identical on both ranks: TP decode runs the full attention on each card
//     and splits only `attn_output`'s rows (TP_RESULTS §5.2, `qsa-is-replicated`).
//   * GDN - FULL WIDTH on a layer-split stage (48 v-heads, 10240 conv channels) and SPLIT in
//     `Verifier::gdn_state_tp_` (24 local v-heads, 7168 local conv channels = the replicated q/k channels plus
//     this card's v band).  That is a PERMUTE, not a copy - `gdn_split_plan` is the one definition of it, and
//     it is the same mapping `tp_local_conv_weights` applies to the conv weights and `gdn_recurrence_local`
//     applies through its `head_offset`.
//   * PLE - a conv history of normalized rows, which is sequence state and so moves with the rest.
//
// NOT COPIED, DELIBERATELY: the RoPE tables (built per state at init and identical), the per-token counts
// (`step`, `attention_status`, `pos_dev` - `qsa_step_fill` recomputes every one of them from the position
// before the first decode token), and the MTP drafter's ring (a separate state with its own window).
//
// WHAT IT ASSUMES (all checked, each with its own message):
//   * every session - source and destination - is fully resident (`kv_mode == 0`, the default) and sized for
//     the same `max_cells`, in the same K/V format.  A streamed state's VRAM pools are slots rather than
//     identity pages, so a prefix copy would be wrong.  An ELASTIC pool (`--kv-grow`, the serve default) is
//     fine: it keeps the identity order and only its mapping grows, and the addresses never move - the caller
//     must have grown it to this request's token count (`kvg_ensure` does, before the prompt is read), which
//     the call checks.
//   * the sources' layer ranges cover every layer of the model (a layer-split pipeline is contiguous).
//   * a destination is a WHOLE-MODEL session: TP decode runs all 48 layers on both cards.  A destination with
//     an explicit `layer_lo/layer_hi` instead holds only that slice, which is what a cut-to-cut move between
//     two layer splits needs; it then copies only those layers, and its GDN rows are read from its own
//     `gdn_ord0`.  With one destination the GDN layout is full width on both sides, so that move is a plain
//     copy rather than the TP permute.
//   * a source and a destination MAY be the same session (the phased run of §8.4 shares them), in which case the
//     spans that are already in place are skipped - the call copies only what the other side is missing.
//   * both devices are idle.  The function is synchronous and returns only when every copy has completed.
#pragma once

#include "strata/core/geometry.hpp"
#include "strata/core/session.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

/// The GDN state a TP rank reads: `heads` local v-heads starting at global head `head_off`, and `channels`
/// local conv channels, of which `qk_channels` (the q/k part) are replicated and the rest is this rank's v band.
struct GdnSplitPlan {
    int64_t S = 0;            ///< ssm_state_size: rows and columns of the recurrence
    int64_t HV = 0;           ///< global v-heads
    int64_t heads = 0;        ///< local v-heads (0: `rank`/`n_ranks` do not split evenly - the caller errors)
    int64_t head_off = 0;     ///< this rank's first v-head in the global list
    int64_t qk_channels = 0;  ///< 2 * ssm_k_heads * S: local channels [0, qk_channels) are the replicated q/k
    int64_t channels = 0;     ///< local conv channels: qk_channels + heads * S
    int64_t d_conv = 0;
    /// Floats one GDN layer's split state takes, in `gdn_state_tp_floats_`'s layout: the recurrence
    /// [S][heads][S] then the conv history [channels][d_conv-1].
    int64_t floats() const { return S * heads * S + channels * (d_conv - 1); }
};

/// rank `rank` of `n_ranks`: v-heads are cut into contiguous bands (`head_off = rank * HV / n_ranks`), which
/// is what `tp_local_conv_weights` does to the conv channels and what `gdn_recurrence_local` expects.
inline GdnSplitPlan gdn_split_plan(const ModelGeometry& g, int rank, int n_ranks) {
    GdnSplitPlan p;
    p.S = g.ssm_state_size;
    p.HV = g.ssm_v_heads;
    p.d_conv = g.ssm_d_conv;
    p.qk_channels = 2 * g.ssm_k_heads * g.ssm_state_size;
    if (n_ranks <= 0 || rank < 0 || rank >= n_ranks || p.HV % n_ranks != 0) return p;   // heads stays 0
    p.heads = p.HV / n_ranks;
    p.head_off = (int64_t) rank * p.heads;
    p.channels = p.qk_channels + p.heads * p.S;
    return p;
}

/// One layer-split pipeline stage: the session that OWNS `[layer_lo, layer_hi)` running on `device`.
struct KvSrcStage {
    const SessionState* session = nullptr;
    int device = 0;
    /// **The layers this stage OWNS.  -1: the session's own `layer_lo/layer_hi`.**
    ///
    /// A plain layer-split pipeline's stage holds exactly its slice, so the session answers this.  A PHASED run
    /// (§8.4) is the case that needs it explicit: the source session is the WHOLE-MODEL one the TP decode phase
    /// also uses - two stages then share one session object, and `layer_lo/layer_hi` (which must stay 0/48 for
    /// the destination checks) can no longer say who owns what.  With explicit ranges the same call also
    /// becomes a "fill in the half this card did not compute" pass over the shared sessions, and the
    /// `dst ptr == src ptr` spans it finds there are skipped rather than copied onto themselves.
    int layer_lo = -1;
    int layer_hi = -1;
};

/// One TP rank: a whole-model session (QSA + PLE) on `device`, and the verifier's split GDN arena
/// (`Verifier::gdn_state_tp_`: `n_gdn_layers() * gdn_floats` floats, one row per global GDN ordinal).
/// `gdn_state == nullptr` skips the GDN half for every rank - the QSA/PLE half still runs.
struct KvDstRank {
    SessionState* session = nullptr;
    int device = 0;
    int rank = 0;
    float* gdn_state = nullptr;
    int64_t gdn_floats = 0;
    /// **The layers this destination RECEIVES.  -1: every layer (a whole-model destination).**
    ///
    /// A TP rank receives the model, so it needs no range.  The DUAL-CUT LAYER SPLIT is the case that does:
    /// prefill runs the cuts one way and decode the other, so the layers that changed owner move from one
    /// layer-split stage INTO the other.  Such a destination holds a slice, and `gdn_state` is then the
    /// session's own arena (row 0 = global GDN ordinal `session->gdn_ord0`).
    int layer_lo = -1;
    int layer_hi = -1;
};

struct KvRedistStats {
    int qsa_layers = 0;         ///< QSA layers copied (a layer, not a layer x rank)
    int gdn_layers = 0;
    /// **TOTALS ACROSS DESTINATION RANKS**, not per rank: the figure answers "what did this handover cost",
    /// and a phased call's ranks move different amounts (each brings only the half the other card computed).
    uint64_t qsa_bytes = 0;
    uint64_t gdn_bytes = 0;
    uint64_t ple_bytes = 0;
};

/// Copies `tokens` cells of sequence state from the layer-split stages into every TP rank.
///
/// `tokens` bounds the copy: the pools are page-major and a page's cells are contiguous, so cells
/// `[0, ceil(tokens / page_size))` are a byte prefix of the identity layout.  Unused cells are left alone.
///
/// Returns false with `err` set on a shape mismatch, a missing owner, an unsupported K/V mode or a CUDA error.
/// On success `stats` (optional) says how much moved.
bool kv_redistribute(const ModelGeometry& g, int64_t tokens, const KvSrcStage* src, int n_src, KvDstRank* dst,
                     int n_dst, KvRedistStats* stats, std::string& err);

}  // namespace strata::core
