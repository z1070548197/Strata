// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <cstdio>

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/tp_allreduce.hpp"
#include "strata/core/tp_gdn_state.hpp"
#include "strata/core/tp_moe_split.hpp"
#include "strata/kernels/verify_kernels.hpp"   // FetchSpans (the TP half-fetch's per-layer byte spans)
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// 0.1.43 (STRATA_ADAPT_OVERLAP): the adaptive tier's copies still landing while a decode window runs.  The window's
/// residency tables are the baseline's (the swapped-in experts are resident from the window's first layer), but their
/// bytes arrive layer by layer: `run` holds a layer's plan (the first thing that lets the GPU read the layer's expert
/// slots) back until `landed >= layer` for a layer with a swap (`has[layer]`).  Set `active` only around one window.
struct AdaptGate {
    struct Tick { AdaptGate* gate; int32_t layer; };
    std::atomic<int32_t> landed{-1};      ///< the highest layer whose swap copies have landed (n_layers: all of them)
    std::atomic<bool> active{false};
    std::vector<uint8_t> has;             ///< per layer: a swap copies into this layer's slots
    std::vector<Tick> ticks;              ///< host-function contexts, one per layer and the final one (stable storage)
    const int32_t* dvals = nullptr;       ///< device [0..n_layers]: the source of those copies
    volatile int32_t* hland = nullptr;   ///< pinned host int the copy stream writes (a D2H copy of the layer number), instead of a host function; null: host functions
    int32_t reached() const {
        const int32_t a = landed.load(std::memory_order_acquire);
        if (hland == nullptr) return a;
        const int32_t b = *hland;
        return a > b ? a : b;
    }
};

class NativeHead;
class RemoteExpertOpt;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer);

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const int32_t* h_res = nullptr;      ///< host [n_layers * n_expert] slot or -1 (when all >= 0 in stage: zero-doorbell)
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;
    void set_remote_expert_opt(RemoteExpertOpt* opt) { remote_opt_ = opt; } // before init/capture

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;
    /// #649 (STRATA_VERIFY_TRACE=1, opt-in): the last windows' host-side handshake events and the GPU's breadcrumbs
    /// (in mapped memory, so they read while the GPU hangs).  Nothing when the trace is off.
    void trace_dump(std::FILE* f) const;

    /// #267: raise every flag the window's spin kernels wait on past any ring (UINT32_MAX), so a window the GPU
    /// cannot finish drains instead of staying resident, then wait up to `timeout_ms` for its streams.  For the
    /// paths that give up on the engine (a timed-out window, the serve watchdog): the window then ran on whatever
    /// the flags guarded, so this verifier refuses every later window.  True when the streams finished.
    bool release_gpu_waits(int timeout_ms);

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Diagnostics: row `t` of the last window's head logits (n_vocab floats) to the host. Valid after run().
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return next_ ? next_->vocab() : n_vocab_; }
    /// #879 GUARD: the first row of the last finished window (run, run_slot_rows, batch_poll, pl_finish) whose head
    /// logits held a NaN or an infinity, else -1; nonfinite_pos() is that row's position.  Every window graph
    /// checks its T rows on the device beside the greedy pick (logits_nonfinite_rows into mapped flags, read after
    /// the sync the window already does: no extra sync).  On the stage with the head (a layer split asks the last
    /// stage).  STRATA_NAN_GUARD=0: off (always -1).  STRATA_NAN_INJECT=<n>[,<n>...] (debug): the n-th window the
    /// guard checks in this process (counted from 1 over every stage-with-head verifier: prompt-tail windows, decode
    /// windows and batch windows alike) reports row 0 as non-finite, so the whole response can be tested.
    int nonfinite_row() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_row() : nf_row_; }
    int64_t nonfinite_pos() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_pos() : nf_pos_; }
    /// Row t of the last window (a batch window: each slot's own rows) held a non-finite logit.
    bool nonfinite_in(int t) const {
        if (next_ != nullptr && !last_stage()) return next_->nonfinite_in(t);
        return t >= 0 && t < 32 && ((nf_mask_ >> t) & 1u) != 0;
    }
    static bool nan_guard_on();
    /// The most tokens one window holds (`init`'s `max_t`; 0 before it).
    int max_t() const { return max_t_; }
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        sampling_.logit_bias = logit_bias_host_.empty() ? nullptr : logit_bias_device_;
        if (next_) next_->set_sampling(sp);
    }
    // One request's fixed vocabulary bias. Empty clears it; copied only when changed, on the final stage's device.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// PROBABILISTIC DRAFT ACCEPTANCE (core/spec_prob.hpp, STRATA_SPEC_PROB=1): for the NEXT run() only, judge the
    /// window's first `n_q` drafts against the drafter's distributions `q` (host memory, n_q rows of kSpecQStride
    /// int32: ids, -1 terminated, then probabilities as float bits) with rejection sampling instead of exact match; the
    /// drafts past `n_q` (a lookup chain's tail) are point masses.  Has no effect on a greedy request, a window of one
    /// token, or where the split sampler cannot run - those take the exact-match path.  Cleared by run().
    void set_spec_q(const int32_t* q, int n_q) {
        spec_q_ = q;
        spec_nq_ = n_q;
        if (next_) next_->set_spec_q(q, n_q);
    }
    /// Counters of the rejection path (windows judged, drafts kept / offered over its rows with a q list).
    int64_t spec_windows = 0;

    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers must be device-visible (mapped pinned memory, portable when the stages are on two devices).
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The rows the hand-off buffers of `set_stage` hold (0: kVerifyMaxT, the default).  More than --batch's 8 slots on a
    /// layer split: the pipelined slot groups write their rows at hbase = group * rows-per-group, one row per slot.
    void set_handoff_rows(int rows) { handoff_rows_ = rows; }
    /// Is the zero-doorbell graph the one a batch window would run now (every expert of the stage in VRAM by the host
    /// table)?  Looks at the table, as each window does.  (STRATA_ADMIT_MERGE_FIRST's gate.)
    bool ar_live() { refresh_ar(); return ar_on(); }
    /// STRATA_SPLIT_OVERLAP=1 (opt-in; stages on DIFFERENT devices only): the hand-off carries a ready flag (one
    /// mapped word per hand-off buffer: host and device addresses).  The writing stage raises it from the GPU when
    /// its hand-off is out (handoff_publish); the reading stage's window graph waits on it (wait_flag_ge) - so the
    /// next stage's graph is launched while this stage is still running, and the host no longer syncs between the
    /// stages.  Null (the default): the old order - sync, then launch the next stage.  Set before `init`.
    void set_handoff_flags(uint32_t* in_host, uint32_t* in_dev, uint32_t* out_host, uint32_t* out_dev) {
        hflag_in_h_ = in_host; hflag_in_d_ = in_dev; hflag_out_h_ = out_host; hflag_out_d_ = out_dev;
    }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.
    void set_next(Verifier* next, void* next_user) { next_ = next; next_user_ = next_user; }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }

    // ================================ TENSOR PARALLELISM (two GPUs) ================================
    /// --tp: this verifier is one half of a two-GPU tensor-parallel pair.  `rank` is 0 (the single-writer
    /// allreduce device, CUDA0) or 1 (CUDA1); `peer_dev` is the other card's device index.  Both verifiers run
    /// the SAME layers (the whole model), each with half the dense weights and half the GDN/QSA heads; the
    /// per-layer dense partial sums are allreduced so both cards hold the full residual.  Set before `init`.
    /// The rank-0 verifier is the coordinator: it owns both graphs and launches them concurrently in `run_tp`.
    void set_tp(int rank, int peer_dev) { tp_rank_ = rank; tp_peer_dev_ = peer_dev; tp_on_ = true; }
    /// The peer verifier of this TP pair (the other rank).  Set before `init` on the coordinator (rank 0).
    void set_tp_peer(Verifier* peer) { tp_peer_ = peer; }
    bool tp_on() const { return tp_on_; }
    int tp_rank() const { return tp_rank_; }
    int tp_peer_dev() const { return tp_peer_dev_; }
    /// One TP window: the coordinator (rank 0) launches BOTH cards' graphs concurrently and waits for both.
    /// `tokens`/`pos0`/`out` are the same as `run`; the pool is the coordinator's (rank 0's) pool.
    bool run_tp(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                std::string& err);
    /// The coordinator (rank 0) allocates the pair's shared handshake flags and installs the device pointers
    /// on both ranks; opens P2P.  Called after both verifiers' `set_tp`/`set_tp_peer`.
    bool tp_setup_flags(std::string& err);
    /// Sizes and allocates THIS rank's half-width pipeline scratch and extracts its local 7168-channel GDN
    /// conv weights.  Call on BOTH ranks after their `init` and `tp_setup_flags`.
    bool tp_setup_scratch(std::string& err);
    /// The rank-local persistent GDN state used by TP decode.  Prefill binds this exact view and advances it
    /// in place; no second state allocation or post-prefill copy is allowed.
    TpGdnStateView tp_gdn_state_view() const {
        return {gdn_state_tp_, gdn_state_tp_floats_, g_ ? g_->n_gdn_layers() : 0, device_, tp_rank_};
    }
    /// Zeroes this rank's TP persistent state: the local GDN recurrent states and conv histories in the tp
    /// arena, which the session's `session_zero` does not reach (the TP window graphs read the arena, not
    /// `ss.gdn_state`).  Call on BOTH ranks at a fresh session, beside `session_zero` for each rank's session.
    bool tp_zero_persistent(std::string& err);
    /// The peer's `bo_` buffer as a P2P pointer on this rank's device (rank 0's allreduce reads and writes it).
    /// The coordinator sets BOTH directions after the two verifiers have init'd.
    void set_tp_bo_peer(float* peer_bo) { tp_bo_peer_ = peer_bo; }
    /// The peer's full-width value buffer, resolved on THIS rank's device (P2P).  The column-parallel
    /// out-projections each need the peer's half of the reduction vector, so BOTH ranks write into the other's
    /// y_full_ (rank 1 -> rank 0's [ZVt, ZV) via broadcast_to_a, rank 0 -> rank 1's [0, ZVt) via broadcast).
    void set_tp_y_peer(float* peer_y) { y_peer_ = peer_y; }
    /// The peer's LOCAL-width gate/up hq (its `tp_moe_scratch_`'s hq region) resolved on THIS rank's device
    /// over P2P; the MoE exchange reads it.  The coordinator sets BOTH directions after init (each rank reads
    /// the other's local hq).
    void set_tp_hq_peer(uint8_t* peer_hq, uint8_t* peer_hq2) {
        tp_moe_hq_peer_ = peer_hq;
        tp_moe_hq2_peer_ = peer_hq2;
    }
    /// This rank's LOCAL-width gate/up hq for group set `set` (0 or 1), for the peer's exchange to read (P2P).
    /// `native_expert_hq_off(capx, n_ff_local)` bytes into the set's scratch.
    uint8_t* tp_moe_local_hq(int set) const;
    /// The peer's compact pitched band half (`tp_moe_hitc_`), resolved on THIS rank's device over P2P.  Rank 0
    /// stitches it into its own full-width band before the combine so the chain's fp32 association matches the
    /// single card's exactly.  Set by set_tp_hitc_peer; only rank 0 reads it.
    void set_tp_hitc_peer(float* peer_hitc) { tp_moe_hitc_peer_ = peer_hitc; }
    float* tp_moe_hitc() const { return tp_moe_hitc_; }
    float* tp_y_full() const { return y_full_; }
    float* tp_bo() const { return bo_; }
    /// Rank 1 only: open the P2P pair in the mirrored order (dev_a = rank 0's card, dev_b = this card) and record
    /// the local card.  The fused handshake runs on dev_b; the full-width GDN out-projection writes rank 0's
    /// buffer over P2P (broadcast_to_a).  The shared flag words still belong to rank 0's tp_setup_flags.
    bool tp_set_local_device(std::string& err) {
        if (tp_rank_ == 0) return true;
        return tp_ar_.open(tp_peer_dev_, device_, err);
    }

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);

    // ================================ SEVERAL SEQUENCES IN ONE WINDOW ================================
    //
    // A batch window holds S INDEPENDENT sequences, one token each: row s is slot s, at slot s's own position,
    // reading and writing slot s's own state (GDN recurrence and conv history, QSA K/V and indexer, PLE history),
    // which lives in `slots[s]` - a session carved like this verifier's own (same layer range, same max_cells).
    // Everything that is per row already (hyper-connections, dense projections, router, shared expert, the
    // routed experts on the GPU and the CPU, the head) runs once over the S rows, so the weights are read once
    // per window for all the sequences.  Row s's arithmetic is the single-token window's, so a slot's greedy
    // tokens are its solo greedy tokens (modulo the multi-token CPU kernel choice: STRATA_IQ_MT_MIN=1).
    //
    // Greedy only, no drafts (MTP) in a batch window.  `init_slots` once after `init` (S <= max_t); a layer
    // split's stages each get their own sessions, and run_slots/commit_slots continue into the next stage.
    bool init_slots(const std::vector<SessionState*>& slots, std::string& err);
    int n_slots() const { return (int) slots_.size(); }
    /// One batch window over slots [0, S): tokens[s] at positions pos[s]; out[s] = the greedy pick after it.
    bool run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err);
    /// The same over the S slots `rows` (row t is slot rows[t], any distinct slots in any order): the slots not
    /// listed are not touched, so an idle slot keeps its state (a finished conversation it may continue later).
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    /// Keep every row of the last batch window.
    bool commit_slots(std::string& err);
    /// Commit an accepted prefix in each contiguous slot group of the last window. `keep` has
    /// one entry per slot (indexed by slot ID), each in 1..that slot's group length.
    bool commit_slot_prefixes(const int* keep, std::string& err);
    /// --batch-mtp: slot rotation makes many row layouts, so bound the captured batch graph pairs (LRU). 0 = unbounded.
    void set_batch_graph_limit(size_t n) { batch_graph_limit_ = n; }

    // ---- The stages of a layer split as a PIPELINE.  A batch window over the slot GROUP
    // [base, base + S) is launched on ONE stage with its commit right behind it on the stage's stream (a batch window
    // keeps every row, so the commit needs no host decision), and the host serves the rings of every stage that has
    // a window in flight from one thread (batch_poll does not block).  Stage k can then run group g while stage k+1
    // runs group g-1.  Rows of group `base` use hand-off rows [base, base + S), so groups never share a hand-off row.
    bool batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = this stage's window and commit are done (the last stage's picks are in batch_out), 0 = still running,
    /// -1 = an error (err).  Serves every layer that has rung so far.
    int batch_poll(PoolMultiFn pool, void* user, std::string& err);
    bool batch_busy() const { return b_running_.load(std::memory_order_acquire); }
    /// Serves the layers of this stage's window that have rung (the CPU pool and the flags), with no CUDA call: 1 once
    /// every layer is served (or the window raises no host doorbells), 0 while one is still to ring.  batch_poll
    /// calls it, unless set_external_serve(true) moved it to a thread of its own (STRATA_PIPED_CONCURRENT: a CUDA
    /// call can wait for the driver's lock while another thread holds it waiting for the GPU, whose window waits
    /// for its doorbell, #31); batch_poll then makes no CUDA call before every layer is served.
    int batch_serve(PoolMultiFn pool, void* user);
    void set_external_serve(bool on) { ext_serve_.store(on, std::memory_order_release); }
    /// A slot's sampling (temperature / top_p / top_k / min_p / seed; penalties are not applied in batch windows):
    /// its row is drawn again on the last stage with Philox(seed, position), as a solo window draws it.  Greedy by
    /// default.  Set on the first stage, it reaches the last.
    void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) {
        if (slot >= 0 && slot < (int) slot_sp_.size()) slot_sp_[(size_t) slot] = sp;
        if (next_) next_->set_slot_sampling(slot, sp);
    }
    const int32_t* batch_out() const { return b_out_; }
    bool last_stage() const { return g_ != nullptr && le_ == g_->n_layers; }
    /// commit() returns without waiting for its graph (a single-GPU session sets it): the next window follows it on
    /// the same stream and the drafter reads nothing it writes, so it overlaps the draft. Whoever reads or writes
    /// the session from another stream or the host afterwards (a new request, a checkpoint, a snapshot, the prompt
    /// path, the end of a run) calls wait_commit() first.  STRATA_COMMIT_SYNC=1 keeps the wait.
    static void set_commit_async(bool on);
    /// Waits for the last commit graph when commit() did not (an event recorded after it, not the whole device);
    /// false with `err` when it failed.  Free when nothing is pending.
    bool wait_commit(std::string& err);

    // ---- PIPELINED WINDOWS (--pipeline-windows, a layer split on two GPUs).  One conversation's windows with the
    // stages overlapped: stage 0 runs window K+1 while stage 1 still runs window K.  The same window as `run`, driven
    // without blocking the host, so one host thread keeps a window in flight on each stage (the batch pipeline's
    // pattern, batch_launch / batch_poll, for one sequence with drafts).  Two verifiers per stage (one per window
    // parity) share the stage's stream and each has its own hand-off.  `pl_launch` stages and launches (it never
    // captures: `capture_all` first, with nothing in flight), `service` serves the layers whose doorbells have rung
    // and returns at once, `done` polls the window's completion, `pl_finish` reads the picks, `pl_commit_async`
    // queues the commit.  Nothing chains to `next_`: the caller drives every stage.
    /// The compute stream to use instead of a private one (the two verifiers of one stage share it).  Before `init`.
    void set_stream(cudaStream_t s) { ext_stream_ = s; }
    /// Every layer copies the token rows to the host (doorbell_publish), not only the layers with a routed expert
    /// outside the VRAM tier by the device's residency table: with windows in flight the adaptive tier marks an
    /// expert evicted on the host (the pool then computes it on the CPU) before the device table follows.  Also turns
    /// the device-planned layers (E-6) off.  Before `init`.
    /// Before the first window (its graphs are captured there): on also turns E-6's device plan off, which reads the
    /// device table this mode lets lag.
    void set_always_publish(bool on) {
        always_publish_ = on;
        if (on) device_plan_ = false;
    }
    /// 0.1.43: the adaptive tier's overlap gate (see AdaptGate).  Refused (false) with the device-planned layers (E-6),
    /// which never ask the host.  The pointer must outlive the verifier's windows.
    bool set_adapt_gate(AdaptGate* g) {
        if (g != nullptr && device_plan_) return false;
        adapt_gate_ = g;
        return true;
    }
    double ms_gate = 0;   ///< host time spent holding a layer's plan for the adaptive copies (decode timing prints it)
    cudaStream_t stream() const { return cs_; }
    int device() const { return device_; }
    /// Capture every window size and the commit graph now (a capture syncs the stream: never with a window in flight).
    bool capture_all(std::string& err);
    /// Free the idle solo window graphs; the next window captures them again under whatever the kernels read THEN.
    /// Returns how many executables were freed, 0 when there was nothing to free or the graphs are not ours to drop
    /// (`--pipeline-windows` prepares them once).  A kernel switch that changes the summation ORDER (`select_simt`,
    /// §8.3-3d) calls this so this request's prefill and its decode share one order.  Safe from any device: it enters
    /// the verifier's own.
    int drop_solo_graphs();
    bool pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    /// Stage a window ahead of its launch (positions, and the PLE rows from `ple_prev` = the two tokens before the
    /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
    /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
    bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
    /// 1: every layer served; 0: the GPU has not reached the next layer yet; -1: an error (`err`).
    int service(PoolMultiFn pool, void* user, std::string& err);
    bool in_flight() const { return fl_active_; }
    /// The window's graph (and its profile copy) completed; false while it runs.  An error sets `err`.
    bool done(std::string& err);
    /// After `done`: the profile, the last stage's host sampling and picks (`out` may be null on an earlier stage).
    bool pl_finish(int32_t* out, std::string& err);
    /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
    /// (after its state was restored) replays it with another count.
    bool pl_commit_async(int n_keep, std::string& err);
    /// Fold another verifier's counters and GPU profile into this one's (the two verifiers of one stage report once).
    void absorb_stats(Verifier& o);
    /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
    void diag_pipelined(std::FILE* f, const char* name) const;

    /// Measurement hook (STRATA_LOGPOS): after run(), write one line per row t of the last window's head -
    /// "pos target logprob top top_logprob hit extra_logprob target_logprob_without_extra" - where row t is the
    /// distribution at pos0 + t, targets[t] is the token at pos0 + t + 1, extra_logprob is the log-probability of
    /// `extra_id` in the same row, and the last column is the target's log-probability in that row renormalized
    /// over every token but `extra_id` (both nan when they do not apply).  A layer split's earlier stage forwards
    /// to the stage that holds the head.
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return &sink_; }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { sink_.pcie_mode = mode; }
    /// Called before every window this verifier launches (run, the batch and pipelined launches): the device's
    /// residency table catches up with the host's (the adaptive tier's evictions, STRATA_ADAPT_FETCH's admissions).
    /// It gets this verifier's stream; false (with the message in `err`) fails the launch.
    void set_pre_launch(std::function<bool(cudaStream_t cs, std::string& err)> f) { pre_launch_ = std::move(f); }
    /// A graph that finds no VRAM after the verifier freed its own graphs asks this for `bytes` more (the engine's
    /// expert cache gives up slots); it returns what it freed, 0 when it cannot.  Runs on a quiet stream.
    void set_vram_relief(std::function<size_t(size_t bytes)> f) { vram_relief_ = std::move(f); }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    /// The PCIe count of each layer from measured costs (PcieModel) instead of a fixed share: the spin kernels stamp
    /// each step's GPU work (the plan's arrival, the PCIe part's start, the wait for the CPU's rows; GPU clock), the host
    /// times its pool, and after each window *model (the dispatch's) takes the costs. One GPU, no batch slots. Set
    /// before `init` (the stamps are captured). `paused`: a request's own --pcie-frac holds meanwhile.
    void set_pcie_balance(PcieModel* model) { bal_ = model; }
    void pause_pcie_balance(bool paused) {
        bal_paused_ = paused;
        if (bal_ != nullptr && m_bal_ != nullptr) bal_->on = !paused;
    }
    bool pcie_balance_on() const { return bal_ != nullptr && m_bal_ != nullptr; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    /// STRATA_PCIE_TIMING=1: the ms this verifier spent copying missed experts over PCIe (pcie_mode 2's
    /// fetch_blobs + rebase_ptrs, on the compute stream), summed over every window so far - the same running
    /// total as ms_wait, so a request reads it as a delta. 0 when the timing is off.
    double ms_pcie = 0;
    int64_t windows = 0;
    /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
    /// window), as one line; empty when off.
    std::string profile_report();
    /// The same line for the TP peer (rank 1): its own stamps, read back from its device after a sync. Rank 0's
    /// report shows the time it spends waiting for the peer; only this one says what the peer was doing instead.
    /// Empty when the peer is not profiling.
    std::string peer_profile_report();

private:
    RemoteExpertOpt* remote_opt_ = nullptr;
    bool capture(int T, std::string& err);
    bool sample_window(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    // batch windows (see init_slots)
    std::vector<SessionState*> slots_;
    bool batch_rec_ = false;               ///< record_window is capturing a batch window
    int row_base_ = 0;                     ///< ... its hand-off rows start here (a pipeline group's own rows)
    int brow_[8] = {};                     ///< ... and row t is slot brow_[t]
    bool last_batch_ = false;              ///< the last run was a batch window (set_plan_slot: one group)
    std::map<std::vector<int>, cudaGraphExec_t> exec_bm_, commit_bm_;   ///< full row layout avoids slot-ID collisions
    std::map<std::vector<int>, uint64_t> bm_used_;   ///< last use of each captured layout (LRU)
    uint64_t bm_tick_ = 0;
    size_t batch_graph_limit_ = 0;         ///< 0: keep every captured batch graph (0.1.39); N: LRU-evict beyond N layouts
    int last_rows_[8] = {};                ///< the slots of the last batch window's rows
    std::vector<int> bkey(const int* rows, int S, int hbase) const {   // #871: the doorbell variant has its own graphs
        std::vector<int> k = batch_key(rows, S, hbase);
        k.push_back(ar_off_ ? 1 : 0);
        return k;
    }
    static std::vector<int> batch_key(const int* rows, int S, int hbase) {
        std::vector<int> k;
        k.reserve((size_t) S + 1);
        k.push_back(hbase);
        for (int t = 0; t < S; ++t) k.push_back(rows[t]);
        return k;
    }
    // batch_launch / batch_poll (with external serving, batch_serve runs on another thread: running and the served
    // layer count are atomics, set by batch_launch before running)
    std::atomic<bool> b_running_{false};
    std::atomic<int64_t> b_k_{0};
    int64_t b_steps_ = 0;
    std::atomic<bool> ext_serve_{false};
    // when a layer was last served (steady clock ticks): the serving thread writes it, batch_poll's checks read it
    std::atomic<int64_t> b_last_{0};
    void ring_now() {
        b_last_.store((int64_t) std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
    }
    std::chrono::steady_clock::time_point last_ring() const {
        return std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(b_last_.load(std::memory_order_acquire)));
    }
    std::chrono::steady_clock::time_point ext_checked_{};   ///< batch_poll's last check with external serving
    int32_t b_out_[8] = {};
    std::vector<strata::kernels::SamplerParams> slot_sp_;
    bool sample_rows(int S, std::string& err);   ///< the sampled slots' rows of the last batch window
    int32_t* h_commitb_ = nullptr; int32_t* m_commitb_ = nullptr;   // per slot [1, 0, pos, -1 ..], stride 2 + max_t
    int32_t* commitb_ = nullptr;
    float* tail_snap_b_ = nullptr;         ///< per (slot, QSA layer) indexer tail snapshot
    void* arena_b_ = nullptr;
    int64_t last_pos_b_[8] = {};
    bool capture_batch(const int* rows, int S, int hbase, std::string& err);
    /// Free the graph pair of the least recently used batch layout other than `keep` (`evicted`: there was one).
    bool evict_batch_graph(const std::vector<int>& keep, bool& evicted, std::string& err);
    /// cudaGraphInstantiate; out of VRAM: batch layouts other than `key` (LRU), then (a batch graph) the solo window
    /// graphs, then (a single-request window, `keep_t` > 0) the graphs of the other window sizes, largest first
    /// (#3702): a window size whose graph does not fit used to end the engine ("verify: instantiate: out of memory");
    /// a freed size is captured again when it is next used.  Then the relief hook.  Fails only when none frees enough.
    bool instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t graph, const std::vector<int>& key, const char* what,
                              std::string& err, int keep_t = -1, unsigned long long flags = 0);
    /// Free the graph of the largest window size other than `keep_t` (and the prelaunched one); `evicted`: there was one.
    bool evict_window_graph(int keep_t, bool& evicted, std::string& err);
    bool no_window_eviction_ = false;   ///< capture_all wants every size at once: it must not free one to capture another
    void note_prio_fork(cudaStream_t from);   ///< STRATA_GRAPH_PRIO: where a branch leaves the main stream
    void note_prio_tail(cudaStream_t on, int prio);   ///< STRATA_GRAPH_PRIO: a priority stream's tail at its join
    void apply_node_priorities(cudaGraph_t graph, int T);   ///< STRATA_GRAPH_PRIO: repair + report the node priorities
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    // STRATA_GRAPH_PRIO (record_window / capture): the captured tails of the priority streams at their joins and the
    // main stream's tails at their forks, so that a capture that dropped the stream priorities can be repaired per
    // node (apply_node_priorities walks each tail back to its fork)
    std::vector<cudaGraphNode_t> prio_forks_;
    std::vector<std::pair<cudaGraphNode_t, int>> prio_tails_;
#endif
    void collect_profile();   ///< STRATA_VERIFY_PROFILE: add the last window's stamps to prof_sum_
    void accumulate_profile(const unsigned long long* stamps);   ///< one window's stamps (host copy) into prof_sum_
    // pipelined windows (pl_launch ...)
    cudaStream_t ext_stream_ = nullptr;   ///< set_stream: the stage's shared stream (not destroyed here)
    bool always_publish_ = false;
    bool adapt_gate_wait(int32_t layer, std::string& err);
    AdaptGate* adapt_gate_ = nullptr;     ///< 0.1.43: STRATA_ADAPT_OVERLAP's gate (null: off)
    void pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]);
    cudaEvent_t ev_done_ = nullptr, ev_commit_ = nullptr;
    unsigned long long* prof_pin_ = nullptr;   ///< pinned host copy of the stamps (pipelined windows)
    bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
    int fl_T_ = 0;
    int64_t fl_k_ = 0, fl_total_ = 0;
    double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
    int32_t pl_prev_[2] = {-1, -1};
    std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
    bool capture_commit_batch(const int* rows, int S, int hbase, std::string& err);
    bool stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    const int32_t* spec_q_ = nullptr;   ///< set_spec_q: the drafter's q rows for the next window (host)
    int spec_nq_ = 0;
    int32_t* d_spec_ = nullptr;          ///< device: kVerifyMaxT draft ids, then kVerifyMaxT q rows
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
#if defined(STRATA_GPU_METAL)
    void* metal_ = nullptr;              ///< the Metal decode window (metal/src/engine/window.cpp): init/run/commit go there
#endif
    std::vector<float> logit_bias_host_;
    float* logit_bias_device_ = nullptr;
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    unsigned long long* rr_stats_ = nullptr;   ///< STRATA_ROUTE_RESIDENT's counters on device_ (#1578); null: none
    std::atomic<bool> released_{false};  ///< #267: release_gpu_waits ran (maybe on the watchdog thread): no more windows
    bool all_resident_ = false;           ///< 100% of experts in [lb_, le_) resident in VRAM: zero-doorbell graph
    /// #871: the zero-doorbell graph plans from the device residency table alone, so it is only right while every
    /// expert of the stage is in VRAM.  A prompt loan, a VRAM shrink or an adaptive swap marks some -1 for a while:
    /// every window then runs the doorbell graph (exec_nr_ / the batch key's top bit), the same as a stage that never
    /// was all-resident.  refresh_ar() looks at the host table before each window.
    bool ar_off_ = false;
    bool ar_on() const { return all_resident_ && !ar_off_; }
    void refresh_ar();
    uint32_t* h_plan_err_ = nullptr; uint32_t* m_plan_err_ = nullptr;   // set by resident_plan on a -1 (all-resident graph)
    /// STRATA_TP_DUMP=1: the first 8 floats of every layer's `bo_` (the combined residual) in mapped host
    /// memory, so a TP run and a single-card run of the same window can be compared layer by layer to find
    /// the first one that diverges.  Null unless the variable is set.
    float* h_tpdump_ = nullptr;   float* m_tpdump_ = nullptr;
    /// STRATA_TP_VEC=N: the number of floats each dump site copies to `h_tpdump_ + kTpVecOff + site*N`, so a
    /// whole vector can be diffed between the two paths (0: the scalar dumps only, the buffer is then sized as
    /// it always was).  Read once at init; both paths use the same length, which is what makes the two dumps
    /// comparable.
    static int tp_vec_len();
    static constexpr int kTpVecSites = 4;    ///< sites per layer: mixer input, pre-allreduce, post-combine, residual
    static constexpr int64_t kTpVecOff = 4096;   ///< past every scalar slot of a solo-sized dump block (48*8+564 = 948)
    /// The float offset of (layer, site)'s vector copy, or -1 when STRATA_TP_VEC is off.
    int64_t tp_vec_at(int64_t l, int site) const {
        const int n = tp_vec_len();
        if (n <= 0 || h_tpdump_ == nullptr) return -1;
        return kTpVecOff + (l * kTpVecSites + site) * (int64_t) n;
    }
    const int32_t* h_res_ = nullptr;      ///< the host residency table (VerifyHits::h_res)
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    uint32_t* skip_ = nullptr;            ///< E-6: per group, the ring whose plan the device built (0: the host's)
    unsigned* qcnt_ = nullptr;             ///< S26 STRATA_QFUSE: the HC read's q8_1 group counters (n_embd / 32)
    unsigned long long* slot_off_d_ = nullptr;   ///< E-6: the slot offsets on the device
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    int handoff_rows_ = 0;               ///< set_handoff_rows (0: kVerifyMaxT)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    uint32_t* hflag_in_h_ = nullptr;     ///< set_handoff_flags: the incoming hand-off's ready word (host / device)
    uint32_t* hflag_in_d_ = nullptr;
    uint32_t* hflag_out_h_ = nullptr;    ///< ... and the outgoing one's
    uint32_t* hflag_out_d_ = nullptr;
    uint32_t* hcount_d_ = nullptr;       ///< handoff_publish's block counter (device memory)
    uint32_t* herr_h_ = nullptr;         ///< the bounded hand-off wait gave up (mapped; the reading stage's)
    uint32_t* herr_d_ = nullptr;
    uint32_t* hdrop_h_ = nullptr;        ///< STRATA_TEST_HANDOFF_DROP: the writing stage withholds its flag (mapped)
    uint32_t* hdrop_d_ = nullptr;
    bool handoff_failed(std::string& err) const;   ///< the bounded hand-off wait timed out in the last window
    int prelaunched_ = 0;                ///< prelaunch ran for a window of this size: run skips staging + launch
    /// The overlapped split: stage this window's inputs and launch its graph now (the graph waits on the hand-off
    /// flag); `run` then only serves the host side.  Called on a helper thread while the previous stage runs.
    bool prelaunch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    bool chain_overlap() const;          ///< this stage hands off through a flag to a next stage that waits on it
    Verifier* next_ = nullptr;
    void* next_user_ = nullptr;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    void stage_inputs(int T, const int32_t* tokens, int64_t pos0);
    bool staged_ = false;
    bool copy_used_ = false;
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);
    // --tp: the tensor-parallel window.  The default decode path (all env-gated branches off) with the dense
    // GDN/QSA projections split across the two cards and a per-layer allreduce of the out-proj partial.  Captured
    // on this rank's device; the coordinator launches both ranks' graphs concurrently (run_tp).
    bool record_window_tp(int T, cudaStream_t cs, std::string& err);
    bool capture_tp(int T, std::string& err);
    // #649: STRATA_VERIFY_TRACE=1 - a host event ring (trace_ev) and GPU breadcrumbs: the profiler's stamp points,
    // per layer and token group, written to mapped memory (null when the trace is off)
    void trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const;
    unsigned long long* trace_h_ = nullptr;
    unsigned long long* trace_m_ = nullptr;
    size_t trace_n_ = 0;
    static constexpr int kProfPer = 33;              // stamps per layer (32 left the hc-read second
                                      // half's up-stamp at slot 32 = the next layer's slot 0: D8)
    bool prof_on_ = false;
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    float* ple_key_ = nullptr;   ///< STRATA_PLE_BATCH: the window rows' PLE key / value projections
    float* ple_val_ = nullptr;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaStream_t sh_cs_ = nullptr;
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr;
    // STRATA_DF_BRANCH: a layer's mixer work that reads only the layer's input, captured as parallel graph branches
    // on these side streams (record_window); the same kernels on the same inputs, only their order is freer
    bool df_branch_ = false;
    cudaStream_t df_side_[2] = {};
    cudaEvent_t df_fork_ = nullptr;
    cudaEvent_t df_join_[2] = {};
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t exec_nr_[9] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
    // fork 34a35db (STRATA_SIDE_MIXER=1): the GDN gates and the QSA indexer on a branch of the window's graph beside the
    // mixer's projections, joined before the recurrence / the attention
    cudaStream_t side_ = nullptr;
    cudaEvent_t mfork_ = nullptr, mjoin_ = nullptr;
    // fork b063c13 (STRATA_SIDE_MIXER=2): a second branch for the GDN gate projection and the QSA queries' chain
    cudaStream_t side2_ = nullptr;
    cudaEvent_t pfork_ = nullptr, pjoin_ = nullptr;
    cudaGraphExec_t commit_exec_ = nullptr;

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    uint32_t* h_nf_ = nullptr;   uint32_t* m_nf_ = nullptr;     // #879: kVerifyMaxT * kNonfiniteBlocks flags
    int nf_row_ = -1;                     ///< nonfinite_row()
    int64_t nf_pos_ = -1;                 ///< nonfinite_pos()
    uint32_t nf_mask_ = 0;                ///< nonfinite_in(): bit t = row t
    void note_nonfinite(int T, const int64_t* pos_b);   ///< the window's flags (+ STRATA_NAN_INJECT) -> nf_row_
    float* h_x_ = nullptr;       float* m_x_ = nullptr;         // doorbell payload: T * n_embd
    int32_t* h_ids_ = nullptr;   int32_t* m_ids_ = nullptr;     // T * k
    float* h_w_ = nullptr;       float* m_w_ = nullptr;         // T * k
    uint32_t* h_seq_ = nullptr;  uint32_t* m_seq_ = nullptr;
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr;
    uint32_t* h_flagA_ = nullptr; uint32_t* m_flagA_ = nullptr;  // the GPU plan is in place
    uint32_t* h_flagB_ = nullptr; uint32_t* m_flagB_ = nullptr;  // the PCIe share's DMA copies have landed
    // set_pcie_balance: per step (layer x group, stride 2 groups) the GPU's clock as the plan arrives, as the PCIe
    // part starts and as it waits for the CPU rows (4 words a step); the pool's ms, its experts, the PCIe ones, and
    // whether the step had a plan
    unsigned long long* h_bal_ = nullptr; unsigned long long* m_bal_ = nullptr;
    std::vector<double> bal_p_;
    std::vector<int32_t> bal_nc_, bal_np_;
    std::vector<char> bal_ok_;
    PcieModel* bal_ = nullptr;
    bool bal_paused_ = false;
    void pcie_balance_window(int64_t steps, int G);
    cudaEvent_t commit_done_ = nullptr;   // recorded after an async commit (set_commit_async); see wait_commit
    bool commit_pending_ = false;
    cudaStream_t copy_ = nullptr;                                 // the copy engine's stream (DMA of missed experts)
    // STRATA_PCIE_TIMING: a device stamp before and after each fetch_blobs/rebase_ptrs pair (pcie_mode 2's H2D),
    // summed at the window's sync into ms_pcie.  Device stamps, not cuda events: these calls run inside the
    // captured window graph, and an event recorded in a graph cannot be read back (cudaEventElapsedTime says
    // invalid argument).  gpu_stamp writes %globaltimer, which a graph node may do.  Off by default.
    static constexpr int kPcieTimingMax = 256;                    // >= layers per stage x groups per layer
    bool pcie_timed_ = false;
    int pcie_n_ = 0;                                              // stamp slots used while recording this window
    int pcie_pairs_ = 0;                                          // pairs in the recorded graph (every launch restamps)
    bool pcie_drained_ = false;                                   // this window's stamps were summed already (done())
    unsigned long long* pcie_st_ = nullptr;                       // device: 2 stamps per fetch (start, end), ns
    std::vector<unsigned long long> pcie_h_;                      // the same, read back after the window's sync
    void drain_pcie_timing();                                     // after the window finished: sum the pairs into ms_pcie
    struct FlagSet { uint32_t* flag; uint32_t value; };
    FlagSet flag_sets_[2 * 64 * 2] = {};                          // host-function arguments, one per (layer, group)
    static void fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes);
    static void raise_flag(uint32_t* flag, uint32_t value);
    int32_t* h_plan_ = nullptr;  int32_t* m_plan_ = nullptr;     // counts | start | dst | tok | ptr (as int32 pairs)
    int64_t plan_i32_ = 0;                                        // int32 words in the plan block
    int64_t dst2_off_ = 0;                                        // its PCIe groups' slot targets (GpuPlanSink::dst2)
    GpuPlanSink sink_;
    uint32_t cur_layer_ = 0;
    static void publish_plan(void* ctx);
    void set_plan_slot(int grp);
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    // --tp: tensor parallelism across two GPUs.  tp_on_ is set by set_tp; tp_rank_ is 0 (CUDA0, the
    // single-writer allreduce device and the coordinator) or 1 (CUDA1).  tp_peer_ is the other rank's
    // verifier (set on the coordinator).  tp_ar_ is the P2P allreduce; tp_exec_[T] is this rank's window
    // graph (captured on this rank's device).  The coordinator launches both ranks' graphs concurrently.
    bool tp_on_ = false;
    int tp_rank_ = 0;
    int tp_peer_dev_ = -1;
    Verifier* tp_peer_ = nullptr;
    TpAllreduce tp_ar_;
    cudaGraphExec_t tp_exec_[9] = {};
    // TP per-layer handshake flags (mapped host memory, shared by both ranks; the device pointer is the same on
    // both cards).  flag_b is raised by rank 1 after its out-proj partial; rank 0 waits on it before the allreduce.
    // flag_a is raised by rank 0 after the allreduce; rank 1 waits on it before its next layer.  Values are the
    // global layer index + 1 (0..n_layers), reset to 0 between runs.
    uint32_t* tp_flag_a_h_ = nullptr;   uint32_t* tp_flag_a_d_ = nullptr;
    uint32_t* tp_flag_b_h_ = nullptr;   uint32_t* tp_flag_b_d_ = nullptr;
    /// flag_c: raised by rank 0 after the layer's MoE residual is the same on both cards.  With the experts'
    /// FFN split (tp_moe_) it is raised after the combine's allreduce; rank 1 waits on it before the FFN half
    /// read.  Kept for the pair's single-broadcast handshake.
    uint32_t* tp_flag_c_h_ = nullptr;   uint32_t* tp_flag_c_d_ = nullptr;
    /// flag_d / flag_e: the MoE combine's allreduce handshake.  Both cards compute their own half of every
    /// routed expert (the intermediate width is cut in two), combine it into their own `bo_`, and the two
    /// partials are summed:
    ///   rank 1: combine -> raise(flag_d, l+1) -> wait(flag_e, l+1)
    ///   rank 0: wait(flag_d, l+1) -> allreduce(bo_, peer bo_) -> raise(flag_e, l+1)
    /// The allreduce is the single-writer form: rank 0's kernel writes BOTH cards' `bo_`.
    uint32_t* tp_flag_d_h_ = nullptr;   uint32_t* tp_flag_d_d_ = nullptr;
    uint32_t* tp_flag_e_h_ = nullptr;   uint32_t* tp_flag_e_d_ = nullptr;
    /// flag_f: rank 0 has finished staging the layer's PCIe share (`fetch_blobs` + `rebase_ptrs`) and rank 1
    /// may read the rebased pointers.  Rank 0 owns that staging area, so rank 1's read of it has to wait.
    uint32_t* tp_flag_f_h_ = nullptr;   uint32_t* tp_flag_f_d_ = nullptr;
    uint32_t* tp_flag_g_h_ = nullptr;   uint32_t* tp_flag_g_d_ = nullptr;   // GDN full-width out-projection
    /// MoE hq exchange: flag_h is rank 1 -> rank 0 ("rank 1's gate/up1 local hq is written"), flag_i is
    /// rank 0 -> rank 1 (the mirror).  Each rank raises its own AFTER its gate/up1 and waits the other's BEFORE
    /// the exchange reads the peer's local hq over P2P - one flag per direction, as the GDN exchange.
    uint32_t* tp_flag_h_h_ = nullptr;   uint32_t* tp_flag_h_d_ = nullptr;
    uint32_t* tp_flag_i_h_ = nullptr;   uint32_t* tp_flag_i_d_ = nullptr;
    /// MoE band stitch: flag_j is rank 1 -> rank 0 ("my pitched band half is written").  Rank 0 waits on it
    /// before reading rank 1's half over P2P and stitching it into its own full-width band, so its combine input
    /// matches the single card's byte for byte (see moe_hit_stitch).
    uint32_t* tp_flag_j_h_ = nullptr;   uint32_t* tp_flag_j_d_ = nullptr;
    /// The experts' FFN split, per rank: gate/up by the LOCAL intermediate width (n_ff/2), down by the LOCAL
    /// OUTPUT width (n_embd/2) with the FULL n_ff reduction.  Set by tp_setup_scratch; `tp_moe_ok_` is false
    /// when the pack's expert geometry cannot be split (the window then refuses rather than silently running
    /// the whole MoE on one card).
    TpMoeSplit tp_moe_;
    bool tp_moe_ok_ = false;
    /// Per layer, the byte spans of an expert blob THIS rank's kernels read (its gate/up rows, its down rows and
    /// the NVFP4 tail), for the PCIe share's fetch: copying only those halves moves half the bytes over the link.
    /// Offsets are into the packed blob and 16-byte aligned; `tp_half_fetch_` is false when any layer's pack makes
    /// that impossible (the fetch then copies the whole blob, as before).  See fetch_blobs_spans.
    std::vector<strata::kernels::FetchSpans> tp_fetch_spans_;
    bool tp_half_fetch_ = false;
    /// True when the TWO ranks' spans together TILE every layer's blob (no holes; overlaps - the NVFP4 tail both
    /// ranks read - are harmless, every window copies source -> the same offsets).  Then a cache-ADMITTED blob's
    /// slot can be filled by both ranks at once (fetch_blobs_spans dst2_mode 2) instead of one rank copying the
    /// whole blob, which is what kept an admitted entry at a full-blob upload on one link.  Checked at startup,
    /// because a hole in the slot is silent corruption for a later hit.
    bool tp_slot_cover_ = false;
    /// `STRATA_TP_TWO_RANK_SLOT=1` + tp_slot_cover_: fill an admitted slot from both ranks (dst2_mode 2) instead of
    /// rank 0 copying the whole blob (mode 0) while rank 1 skips it (mode 1).  MEASURED SLOWER off by default: at 8K
    /// the GPU-busy time per window went 104.52 -> 108.54 ms over both requests (and with FEWER misses, 7415 vs
    /// 9415), because this rank's half reaches rank 0's slot as 16-byte peer stores over NVLink, which costs more
    /// than the PCIe bytes it saves.  Left in for a future DMA-based fill (one cudaMemcpyPeerAsync per blob).
    bool tp_two_rank_slot_ = false;
    /// Rank 1's view of rank 0's staging area (tp_share_doorbell).  Rank 0 stages the PCIe share there when the HOST
    /// does the DMA (pcie_mode 0/1); with pcie_mode 2 (`auto`) each rank stages its OWN half into its own `staging_`
    /// over its own PCIe link, so the two links carry the fetch in parallel and NVLink keeps carrying the resident
    /// experts' halves.  `staging_` on rank 1 stays this rank's own buffer.
    uint8_t* tp_stage0_ = nullptr;
    /// This rank's gate/up scratch at the LOCAL width (n_ff/2), ONE PER GROUP SET (a layer's two group sets -
    /// the VRAM/doorbell share and the PCIe share - must not clobber the first's local hq before the exchange
    /// reads it).
    void* tp_moe_scratch_ = nullptr;
    void* tp_moe_scratch2_ = nullptr;
    /// The FULL-width hq (n_ff/32 blocks per entry) the down's `hq + e*(n_ff/32)` read expects; moe_hq_exchange
    /// assembles the two ranks' halves into it over P2P.  One per group set.
    void* tp_moe_hq_ = nullptr;
    void* tp_moe_hq2_ = nullptr;
    /// The split down's COMPACT [cap][n_embd/2] output (it writes `y[j*n_out+row]` with n_out halved);
    /// moe_hit_pitch pitches each row into hit_out_'s own half.  ONE buffer for both group sets: their entries'
    /// dst rows are disjoint (an entry is in exactly one tier), so the second down cannot clobber the first's.
    float* tp_moe_hitc_ = nullptr;
    /// DIAG (STRATA_MOE_HQ_DUMP=1): this rank's pinned staging for the capture-safe layer-0 hq dump.  The
    /// in-graph D2H node (record_window_tp's dump_moe_hq_diag) fills the first moe_hq_diag_hq_bytes_ with the
    /// ASSEMBLED set0 hq (cap x n_ff/32 blocks of 36B) and the rest with this rank's LOCAL set0 gate/up hq
    /// (cap x (n_ff/2)/32 blocks of 36B).  run_tp writes the two ranks' buffers to disk after the window join.
    uint8_t* h_moe_hq_diag_ = nullptr;
    uint8_t* m_moe_hq_diag_ = nullptr;
    int64_t moe_hq_diag_hq_bytes_ = 0;
    int64_t moe_hq_dump_cap_ = -1;   // the prefill window's entry count (T*K), set when the D2H node fires at l==0
    size_t moe_hq_dump_hq_sz_ = 0;   // bytes per .hq file (cap * (n_ff*2/32) * 36)
    size_t moe_hq_dump_hitc_sz_ = 0; // bytes of this rank's down-output buffer (capx * n_embd/2 * 4)
    bool moe_hq_dump_written_ = false;
    /// The peer's LOCAL-width hq (its gate/up scratch's hq region), resolved on THIS rank's device over P2P;
    /// moe_hq_exchange reads it.  One per group set.  Set by set_tp_hq_peer.
    uint8_t* tp_moe_hq_peer_ = nullptr;
    uint8_t* tp_moe_hq2_peer_ = nullptr;
    /// Private copies of a layer plan's group-pointer arrays, shifted by this rank's offsets.  The plan block
    /// is shared with the peer (tp_share_doorbell) and its `p_ptr2` is rewritten in place by rank 0's
    /// `rebase_ptrs`, so the split's pointers are built beside it, never on top of it.  The GATE/UP call uses
    /// these (the row offset is a group-pointer shift); the DOWN call uses the UNshifted plan pointers, its row
    /// offset living in Ldown.down_off instead.
    unsigned long long* tp_ptr_ = nullptr;    // capx entries, from the plan's `p_ptr` (gate/up)
    unsigned long long* tp_ptr2_ = nullptr;   // capx entries, from the plan's rebased `p_ptr2` (gate/up)
    // TP local GDN state: per GDN layer, the recurrent state over the LOCAL v-heads (HV/2) plus the LOCAL conv
    // history (7168 channels).  Layout per layer: [S*(HV/2)*S] recurrent + [7168*3] conv, contiguous.
    float* gdn_state_tp_ = nullptr;
    int64_t gdn_state_tp_floats_ = 0;   // per-layer local GDN state size (floats)
    // TP local scratch, one arena: the half-width GDN pipeline (qkv/h at 7168 conv channels, gate/beta at
    // HV/2 heads, z/y at ZV/2) and the QSA out-projection's contiguous half of attn32_.  None of these may
    // alias the full-width buffers: the row-parallel kernels want contiguous rows of the LOCAL width, while
    // the full buffers are laid out at the global stride.
    void* tp_arena_ = nullptr;
    float* qkv_tp_ = nullptr;    // nG * MT * 7168
    float* h_tp_ = nullptr;      // nG * MT * 7168
    float* gate_tp_ = nullptr;   // nG * MT * (HV/2)
    float* beta_tp_ = nullptr;   // nG * MT * (HV/2)
    float* z_tp_ = nullptr;      // MT * (ZV/2)
    float* y_tp_ = nullptr;      // MT * (ZV/2)
    float* y_full_ = nullptr;    // MT * ZV: rank 0's full-width GDN value buffer (rank 1's half arrives over P2P)
    float* y_peer_ = nullptr;    // rank 0's y_full_ resolved on rank 1's device (P2P), set by set_tp_y_peer
    /// The column-parallel out-projections' scratch: the kernel writes a COMPACT [T][N/2] (its index is
    /// `y[j * n_out + row]`), and each token is then pitched into bo_'s [T][N] layout.  MT * (N/2).
    float* half_out_ = nullptr;
    /// The TP commit graph: this rank's GDN conv history + recurrence state advance by the window's kept
    /// tokens, plus the (replicated) QSA tail restore and indexer append.
    bool capture_commit_tp(std::string& err);
    cudaGraphExec_t tp_commit_exec_ = nullptr;
    /// Both cards' commit graphs, launched together after a TP window.
    bool tp_state_advance(int n_keep, std::string& err);
    /// A TP pair runs ONE doorbell: rank 0's.  The MoE experts are replicated (both cards route identically),
    /// so rank 1 reads rank 0's plan block and its flag/sequence words straight out of rank 0's mapped host
    /// memory instead of asking the pool for a second copy.  `tp_share_doorbell` installs them on the peer
    /// before the window's graphs are captured; the plan's staging pointers are reached over P2P.
    void tp_share_doorbell(Verifier& peer);
    // TP local GDN conv weights: per GDN layer, 7168 channels x 4 taps (F32), extracted from the full 10240-channel
    // weight by tp_local_conv_weights.
    float* conv_w_tp_ = nullptr;
    // The peer's compact pitched band half, as a P2P pointer on this rank's device (rank 0's stitch reads it).
    float* tp_moe_hitc_peer_ = nullptr;
    // The peer's bo_ buffer, as a P2P pointer on this rank's device (rank 0's allreduce reads/writes it).
    float* tp_bo_peer_ = nullptr;
    std::function<bool(cudaStream_t, std::string&)> pre_launch_;   // set_pre_launch
    std::function<size_t(size_t)> vram_relief_;                   // set_vram_relief
    bool graph_oom_said_ = false;          ///< the escalation below said once
    int groups_[9] = {};
    float* h_ymiss_ = nullptr;   float* m_ymiss_ = nullptr;     // T * k * n_embd

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    uint8_t* xil_ = nullptr;                                  // fork F4: the interleaved copy of xq_'s 2-4 columns
    uint8_t* sh_xq_ = nullptr;                                // T columns of q8_1 for shared expert branch
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *hit_out_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr;
    int32_t* plan_ = nullptr;                                     // device copy of the plan block
    uint8_t* staging_ = nullptr;                                  // VRAM slots for the PCIe share of the misses
    static constexpr int64_t kStagingBlobs = 16;
    static constexpr int64_t kPcieGroupRows = 4;                  // the PCIe call's groups side by side (of <= 16)
    uint8_t* hit_xq_ = nullptr;
    uint8_t* nat_xq_ = nullptr;   // plan v0.3 P6: q8_1 activations for a native pack's grouped experts
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    uint16_t* sh_bf16_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' partials and counters
    int32_t* one_ = nullptr;           ///< device {1}: the n_keep of a one-token window, which commits itself
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
