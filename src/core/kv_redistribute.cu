// src/core/kv_redistribute.cu - TP_RESULTS §8.3 step 2.  See the header for what moves and why.
//
// THE THREE TRANSPORTS.  Everything here is a byte copy or a fixed permutation, so the only interesting
// decisions are which device runs the work and how the bytes cross:
//
//   * same device - `cudaMemcpyAsync` D2D on the default stream;
//   * different devices - `cudaMemcpyPeerAsync`, with peer access enabled from the DESTINATION (the copy's
//     stream belongs to the device that will read).  2080 Ti + NVLink2: ~1 GB in 0.04-0.08 s (§8.1-3);
//   * the GDN permute - a kernel launched on the DESTINATION device reading the source through peer access,
//     which is exactly how TP decode already reads its peer's buffers.  A host round trip would be 3.3 MB per
//     layer each way; this is one pass.
//
// Everything is synchronous: the caller guarantees both devices are idle, and the function returns only after
// both have been synchronised.  This happens once per request, at the phase boundary - not on the token path.
#include "strata/core/kv_redistribute.hpp"

#include "strata/core/layer.hpp"   // qsa_kv_elastic_cells: how far the elastic K/V is mapped
#include "strata/core/layout.hpp"
#include "strata/core/tp_split_plan.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace strata::core {
namespace {

constexpr int kThreads = 256;

/// `gdn_state_floats` in session.cpp is file-local; the same sum, named here for what the source side holds.
int64_t full_gdn_floats(const ModelGeometry& g) {
    return g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size + g.ssm_conv_channels * (g.ssm_d_conv - 1);
}

/// The recurrence half: local `[row][local head][col]` <- global `[row][head_off + local head][col]`.
__global__ void gdn_split_rec_kernel(float* __restrict__ dst, const float* __restrict__ src, int S, int HV,
                                     int HVt, int head_off, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t col = i % S;
    const int64_t h = (i / S) % HVt;
    const int64_t row = i / ((int64_t) HVt * S);
    dst[i] = src[(row * HV + head_off + h) * (int64_t) S + col];
}

/// The conv history: local channels [0, qk) are the replicated q/k rows; a local channel past them is global
/// `qk + head_off * S + (c - qk)` - `tp_local_conv_kernel`'s own mapping, with `head_off * S` the v band's
/// channel offset.  Both sides are channel-major `[channel][d_conv-1]`.
__global__ void gdn_split_conv_kernel(float* __restrict__ dst, const float* __restrict__ src, int qk,
                                      int head_off, int S, int taps, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t j = i % taps;
    const int64_t c = i / taps;
    const int64_t full_c = c < qk ? c : qk + (int64_t) head_off * S + (c - qk);
    dst[i] = src[full_c * taps + j];
}

struct OnDevice {
    int prev = 0;
    explicit OnDevice(int d) {
        cudaGetDevice(&prev);
        cudaSetDevice(d);
    }
    ~OnDevice() { cudaSetDevice(prev); }
};

bool ck(cudaError_t status, const char* what, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("kv_redistribute: ") + what + ": " + cudaGetErrorString(status);
    return false;
}

bool enable_peer(int dst_device, int src_device, std::string& err) {
    if (dst_device == src_device) return true;
    const OnDevice on(dst_device);
    int ok = 0;
    if (!ck(cudaDeviceCanAccessPeer(&ok, dst_device, src_device), "cudaDeviceCanAccessPeer", err)) return false;
    if (!ok) {
        err = "kv_redistribute: device " + std::to_string(dst_device) + " cannot read device " +
              std::to_string(src_device) + " (no peer access); the layer-split to TP handover needs both cards " +
              "to see each other";
        return false;
    }
    const cudaError_t e = cudaDeviceEnablePeerAccess(src_device, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        (void) cudaGetLastError();   // enabled by an earlier phase: the state this wants
        return true;
    }
    return ck(e, "cudaDeviceEnablePeerAccess", err);
}

/// The device a pointer's memory really belongs to, or -1 when the driver will not say (a VMM range mapped from
/// another card's allocation reports what the mapping says, if anything).
int ptr_device(const void* p) {
    cudaPointerAttributes a{};
    const cudaError_t e = cudaPointerGetAttributes(&a, p);
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    return a.device >= 0 ? a.device : -1;
}

/// One buffer, from wherever the source stage lives to this rank's device.  A span ALREADY IN PLACE is skipped
/// rather than copied onto itself: the phased run's sources are the destination's own sessions (§8.4), so the
/// handover is "bring the other side's half", and `moved` reports what actually crossed into `stats`.
bool copy_span(int dst_device, void* dst, int src_device, const void* src, uint64_t bytes, uint64_t& moved,
               std::string& err) {
    if (bytes == 0) return true;
    if (dst == nullptr || src == nullptr) {
        err = "kv_redistribute: a state buffer is null (an uninitialised session?)";
        return false;
    }
    if (dst == src && dst_device == src_device) return true;   // same allocation, same device: already there
    // The DECLARED device of a side is not always the device its memory is on: a TP rank's elastic K/V is a VMM
    // range taken from the other card's cache and mapped here, so a call made on the declared pair would ask for
    // a peer copy between two pointers of one card - or peer-copy a mapping.  Ask the driver where the memory is
    // and copy on that device when it says something definite; fall back to the declared pair otherwise.
    const int ddev = ptr_device(dst), sdev = ptr_device(src);
    if (ddev >= 0 && ddev == sdev) {
        const OnDevice on(ddev);
        if (!ck(cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToDevice, nullptr), "state copy", err))
            return false;
        moved += bytes;
        return true;
    }
    const cudaError_t e = dst_device == src_device
                              ? cudaMemcpyAsync(dst, src, (size_t) bytes, cudaMemcpyDeviceToDevice, nullptr)
                              : cudaMemcpyPeerAsync(dst, dst_device, src, src_device, (size_t) bytes, nullptr);
    if (!ck(e, "state copy", err)) return false;
    moved += bytes;
    return true;
}

/// A state's bytes for `used_slots` of its `n_slots` slots: every slot is the same size, so this is a prefix.
uint64_t slot_prefix(uint64_t bytes, int64_t used_slots, int64_t n_slots) {
    if (n_slots <= 0 || used_slots <= 0) return 0;
    if (used_slots >= n_slots) return bytes;
    return bytes / (uint64_t) n_slots * (uint64_t) used_slots;
}

/// A source's owned layers: what the caller declared, or the session's own slice (`KvSrcStage::layer_lo`).
void src_range(const KvSrcStage& s, int64_t n_layers, int64_t& lo, int64_t& hi) {
    lo = s.layer_lo < 0 ? s.session->layer_lo : s.layer_lo;
    hi = s.layer_hi < 0 ? s.session->layer_hi : s.layer_hi;
    if (hi < 0 || hi > n_layers) hi = n_layers;
    if (lo < 0) lo = 0;
}

/// A destination's received layers: what the caller declared, or the whole model (`KvDstRank::layer_lo`).
void dst_range(const KvDstRank& d, int64_t n_layers, int64_t& lo, int64_t& hi) {
    lo = d.layer_lo < 0 ? 0 : d.layer_lo;
    hi = d.layer_hi < 0 ? n_layers : d.layer_hi;
    if (hi < 0 || hi > n_layers) hi = n_layers;
    if (lo < 0) lo = 0;
}

}  // namespace

bool kv_redistribute(const ModelGeometry& g, int64_t tokens, const KvSrcStage* src, int n_src, KvDstRank* dst,
                     int n_dst, KvRedistStats* stats, std::string& err) {
    if (stats != nullptr) *stats = KvRedistStats{};
    if (src == nullptr || n_src <= 0 || dst == nullptr || n_dst <= 0) {
        err = "kv_redistribute: no source stages or no destination ranks";
        return false;
    }
    if (tokens < 0) {
        err = "kv_redistribute: negative token count";
        return false;
    }
    const int64_t n_layers = g.n_layers, n_gdn = g.n_gdn_layers();
    const int64_t page_size = strata::kernels::qsa_real_shapes().page_size;
    const int64_t idx_block = std::max<int64_t>(strata::kernels::qsa_real_shapes().idx_block, 1);

    // ---- 1. the shape of every session, before a single byte moves -------------------------------------
    const int64_t cells = dst[0].session != nullptr ? dst[0].session->max_cells : -1;
    if (cells < 0 || tokens > cells) {
        err = "kv_redistribute: the destination session is missing or smaller than the tokens it must hold";
        return false;
    }
    for (int s = 0; s < n_src; ++s) {
        const SessionState* ss = src[s].session;
        if (ss == nullptr) {
            err = "kv_redistribute: a source stage has no session";
            return false;
        }
        if (ss->max_cells != cells) {
            err = "kv_redistribute: the layer-split stage and the TP rank are sized for different contexts (" +
                  std::to_string((long long) ss->max_cells) + " vs " + std::to_string((long long) cells) +
                  " cells); the redistribution copies a prefix of the identity layout and needs the same carve";
            return false;
        }
    }
    for (int d = 0; d < n_dst; ++d) {
        SessionState* ss = dst[d].session;
        if (ss == nullptr || ss->qsa_states == nullptr) {
            err = "kv_redistribute: a destination rank has no session";
            return false;
        }
        if (ss->max_cells != cells) {
            err = "kv_redistribute: the TP ranks are sized for different contexts";
            return false;
        }
        if (ss->layer_lo != 0 || ss->layer_hi != n_layers || ss->qsa_ord0 != 0) {
            // a SLICED destination (§ the dual cut) is a layer-split stage receiving the layers that changed
            // owner: it must still be able to hold them, but it is not the model
            int64_t dlo = 0, dhi = 0;
            dst_range(dst[d], n_layers, dlo, dhi);
            if (dst[d].layer_lo < 0) {
                err = "kv_redistribute: a destination session must cover the whole model (TP decode runs every "
                      "layer on every card)";
                return false;
            }
            if (dhi <= dlo) {
                err = "kv_redistribute: destination " + std::to_string(d) + " receives no layers";
                return false;
            }
            if (dlo < ss->layer_lo || dhi > ss->layer_hi) {
                err = "kv_redistribute: destination " + std::to_string(d) + " receives layers [" +
                      std::to_string((long long) dlo) + ", " + std::to_string((long long) dhi) +
                      "), which its session ([" + std::to_string((long long) ss->layer_lo) + ", " +
                      std::to_string((long long) ss->layer_hi) + ")) does not carve";
                return false;
            }
        }
    }

    // ---- 2. which stage owns which layer ---------------------------------------------------------------
    std::vector<int> owner((size_t) n_layers, -1);
    for (int s = 0; s < n_src; ++s) {
        int64_t lo = 0, hi = 0;
        src_range(src[s], n_layers, lo, hi);
        if (hi <= lo) {
            err = "kv_redistribute: source stage " + std::to_string(s) + " owns no layers";
            return false;
        }
        for (int64_t l = lo; l < hi; ++l) {
            if (owner[(size_t) l] >= 0) {
                err = "kv_redistribute: two stages both run layer " + std::to_string((long long) l);
                return false;
            }
            owner[(size_t) l] = s;
        }
    }
    for (int64_t l = 0; l < n_layers; ++l) {
        if (owner[(size_t) l] < 0) {
            err = "kv_redistribute: no stage runs layer " + std::to_string((long long) l);
            return false;
        }
    }

    // ---- 3. the GDN split every rank must end up with ------------------------------------------------
    const bool want_gdn = dst[0].gdn_state != nullptr;
    std::vector<GdnSplitPlan> plan((size_t) n_dst);
    for (int d = 0; d < n_dst; ++d) {
        plan[(size_t) d] = gdn_split_plan(g, dst[d].rank, n_dst);
        if ((dst[d].gdn_state != nullptr) != want_gdn) {
            err = "kv_redistribute: some destinations have a split GDN arena and others do not";
            return false;
        }
        if (!want_gdn || n_gdn == 0) continue;
        if (plan[(size_t) d].heads <= 0) {
            err = "kv_redistribute: " + std::to_string((long long) g.ssm_v_heads) + " v-heads do not split " +
                  std::to_string(n_dst) + " ways";
            return false;
        }
        if (dst[d].gdn_floats != plan[(size_t) d].floats()) {
            err = "kv_redistribute: rank " + std::to_string(dst[d].rank) + "'s GDN arena holds " +
                  std::to_string((long long) dst[d].gdn_floats) + " floats per layer, the split needs " +
                  std::to_string((long long) plan[(size_t) d].floats());
            return false;
        }
        if (n_dst == 2 && plan[(size_t) d].channels != kTpGdnConvChannels) {
            err = "kv_redistribute: the GDN conv split (" + std::to_string((long long) plan[(size_t) d].channels) +
                  " channels) disagrees with kTpGdnConvChannels (" + std::to_string(kTpGdnConvChannels) +
                  "); the TP kernels would read a different layout than this copier writes";
            return false;
        }
    }
    // The GDN half is a PERMUTE, so it reads one arena while writing another.  A phased run shares sessions
    // between the two roles, which is why this is checked: sharing the GDN arena itself would make the permute
    // overwrite rows it has not read yet, and the copy half's skip rule cannot see that (the kernels, not
    // `copy_span`, do the moving).  A WHOLE-WIDTH destination (one rank: the cut-to-cut move) is a plain copy
    // from another card's rows, so sharing is harmless there as long as the rows it writes are not rows any
    // source still has to read: the destination's layer range must be disjoint from the shared stage's.
    if (want_gdn && n_gdn > 0) {
        for (int d = 0; d < n_dst; ++d) {
            int64_t dlo = 0, dhi = 0;
            dst_range(dst[d], n_layers, dlo, dhi);
            for (int s = 0; s < n_src; ++s) {
                if (src[s].session->gdn_state != dst[d].gdn_state) continue;
                if (plan[(size_t) d].heads == plan[(size_t) d].HV) {
                    int64_t slo = 0, shi = 0;
                    src_range(src[s], n_layers, slo, shi);
                    if (shi <= dlo || slo >= dhi) continue;   // disjoint rows: nothing it still reads is written
                }
                err = "kv_redistribute: source stage " + std::to_string(s) + " and rank " +
                      std::to_string(dst[d].rank) +
                      " share one GDN arena; the full-width to split permute needs distinct buffers";
                return false;
            }
        }
    }

    // ---- 4. move ---------------------------------------------------------------------------------------
    // cells [0, tokens) are whole pages from the start of an identity-layout pool: a byte prefix
    const int64_t used_pages = (tokens + page_size - 1) / page_size;
    const uint64_t ple_bytes =
        (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * sizeof(float);
    // the PLE is layer 1's module: that stage's history is the one the sequence advanced
    const int ple_stage = owner[(size_t) std::min<int64_t>(1, n_layers - 1)];

    KvRedistStats out;
    // The LAYER counts describe the model, not one rank's share: every destination covers every layer (the
    // coverage check above refuses anything else), and a phased call splits the layers between the cards, so a
    // sum over ranks would not be a layer count either.  The byte figures below ARE totals.  A SLICED call
    // (the cut-to-cut move) has no model-wide destination, so it reports the layers it looked at instead.
    bool sliced = false;
    for (int d = 0; d < n_dst; ++d) sliced = sliced || dst[d].layer_lo >= 0;
    if (!sliced) {
        for (int64_t l = 0; l < n_layers; ++l)
            if (is_qsa_layer(g, l)) ++out.qsa_layers;
        if (want_gdn) out.gdn_layers = (int) n_gdn;
    }
    for (int d = 0; d < n_dst; ++d) {
        SessionState* ds = dst[d].session;
        const int dd = dst[d].device;
        for (int s = 0; s < n_src; ++s)
            if (!enable_peer(dd, src[s].device, err)) return false;
        const OnDevice on(dd);
        const SessionState* ple_src = src[ple_stage].session;
        const int ple_src_device = src[ple_stage].device;
        // the layers this destination receives; a whole-model one takes the lot
        int64_t dlo = 0, dhi = n_layers;
        dst_range(dst[d], n_layers, dlo, dhi);
        // the PLE history is ONE buffer on the card that runs layer 1: a destination that does not own that
        // layer has no history carved (`session_bytes` only carves it for the range containing layer 1)
        const bool ple_here = dlo <= 1 && 1 < dhi;

        int64_t gdn_ord = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const bool mine = l >= dlo && l < dhi;
            const SessionState* ss = src[owner[(size_t) l]].session;
            const int sd = src[owner[(size_t) l]].device;
            if (is_qsa_layer(g, l)) {
                if (!mine) continue;   // this destination does not receive the layer: leave it alone
                const int64_t q = l / std::max<int64_t>(g.qsa_interval, 1);
                if (q < ss->qsa_ord0 || q >= ss->qsa_ord0 + ss->qsa_alloc) {
                    err = "kv_redistribute: the stage that runs layer " + std::to_string((long long) l) +
                          " does not hold QSA state " + std::to_string((long long) q);
                    return false;
                }
                const QsaState& a = ss->qsa_states[(size_t) q];
                const QsaState& b = ds->qsa_states[(size_t) q];
                if (a.kv_mode != 0 || b.kv_mode != 0) {
                    err = "kv_redistribute: QSA layer " + std::to_string((long long) l) +
                          " is streamed; only a fully resident K/V is redistributable (a streamed "
                          "state's VRAM pools are slots, not identity-ordered pages)";
                    return false;
                }
                // An ELASTIC pool (--kv-grow) keeps the same identity order - only its mapping grows, and the
                // addresses never move - so it IS redistributable once both sides have mapped this request's cells.
                // The caller grows them: the serve loop's kvg_ensure does it for every request before the prompt is
                // read.  Not grown here, because a growth takes VRAM from the expert cache and needs the devices
                // quiesced, which is the caller's contract.
                if ((a.kv_elastic >= 0 || b.kv_elastic >= 0) &&
                    strata::core::qsa_kv_elastic_cells(0) < tokens) {
                    err = "kv_redistribute: QSA layer " + std::to_string((long long) l) + ": the elastic K/V holds " +
                          std::to_string((long long) strata::core::qsa_kv_elastic_cells(0)) + " cells of the " +
                          std::to_string((long long) tokens) + " this request needs; grow it before redistributing";
                    return false;
                }
                if (qsa_kv_format(a) != qsa_kv_format(b) || a.n_slots != b.n_slots) {
                    err = "kv_redistribute: QSA layer " + std::to_string((long long) l) +
                          " holds a different K/V format or slot count on the two sides";
                    return false;
                }
                if (used_pages > b.n_slots) {
                    err = "kv_redistribute: QSA layer " + std::to_string((long long) l) + " holds " +
                          std::to_string((long long) b.n_slots) + " pages, the request needs " +
                          std::to_string((long long) used_pages);
                    return false;
                }
                void* ap[4];
                uint64_t ab[4];
                void* bp[4];
                uint64_t bb[4];
                const int na = qsa_kv_spans(a, g, ap, ab);
                const int nb = qsa_kv_spans(b, g, bp, bb);
                if (na != nb) {
                    err = "kv_redistribute: the QSA span count differs between the two sides";
                    return false;
                }
                uint64_t moved = 0;
                for (int i = 0; i < na; ++i) {
                    const uint64_t n = slot_prefix(std::min(ab[i], bb[i]), used_pages, a.n_slots);
                    if (!copy_span(dd, bp[i], sd, ap[i], n, moved, err)) return false;
                }
                // the indexer: the page table, the block tail, the dead key, the pooled ring's used rows and
                // the block position.  All of it is what `qsa_state_zero` would have to leave alone for the
                // next token to attend the same cells.
                const uint64_t table = std::min((uint64_t) a.n_pages, (uint64_t) b.n_pages) * sizeof(int32_t);
                const uint64_t tail = (uint64_t) (idx_block - 1) * g.idx_key_dim * sizeof(float);
                const uint64_t dead = (uint64_t) g.idx_key_dim * sizeof(float);
                const int64_t pooled_rows =
                    std::min<int64_t>(tokens / idx_block + 2, std::min(a.idx_pooled_rows, b.idx_pooled_rows));
                const uint64_t pooled = (uint64_t) std::max<int64_t>(pooled_rows, 0) * g.idx_key_dim * sizeof(float);
                if (!copy_span(dd, b.page_table, sd, a.page_table, table, moved, err) ||
                    !copy_span(dd, b.idx_tail, sd, a.idx_tail, tail, moved, err) ||
                    !copy_span(dd, b.idx_dead, sd, a.idx_dead, dead, moved, err) ||
                    !copy_span(dd, b.idx_pooled, sd, a.idx_pooled, pooled, moved, err) ||
                    !copy_span(dd, b.idx_block_pos, sd, a.idx_block_pos, sizeof(int32_t), moved, err))
                    return false;
                out.qsa_bytes += moved;
            } else {
                if (want_gdn && mine) {
                    const GdnSplitPlan& p = plan[(size_t) d];
                    if (gdn_ord < ss->gdn_ord0 || gdn_ord >= ss->gdn_ord0 + ss->gdn_alloc) {
                        err = "kv_redistribute: the stage that runs layer " + std::to_string((long long) l) +
                              " does not hold GDN state " + std::to_string((long long) gdn_ord);
                        return false;
                    }
                    if (gdn_ord < ds->gdn_ord0 || gdn_ord >= ds->gdn_ord0 + ds->gdn_alloc) {
                        err = "kv_redistribute: the destination of layer " + std::to_string((long long) l) +
                              " does not carve GDN state " + std::to_string((long long) gdn_ord);
                        return false;
                    }
                    const float* a = ss->gdn_state + (size_t) (gdn_ord - ss->gdn_ord0) * full_gdn_floats(g);
                    // row 0 of the destination's arena is global ordinal `gdn_ord0` (0 for a whole-model one,
                    // so the TP path reads exactly what it always did)
                    float* b = dst[d].gdn_state + (size_t) (gdn_ord - ds->gdn_ord0) * dst[d].gdn_floats;
                    const int64_t rec = p.S * p.heads * p.S;
                    if (p.heads == p.HV) {
                        // ONE rank: the layout is full width on both sides, so this is a plain copy and not the
                        // TP permute (and it may cross devices, which a kernel's peer read should not)
                        uint64_t moved = 0;
                        if (!copy_span(dd, b, sd, a, (uint64_t) p.floats() * sizeof(float), moved, err))
                            return false;
                        out.gdn_bytes += (uint64_t) p.floats() * sizeof(float);
                        ++gdn_ord;
                        continue;
                    }
                    if (rec > 0) {
                        const unsigned grid = (unsigned) ((rec + kThreads - 1) / kThreads);
                        gdn_split_rec_kernel<<<grid, kThreads>>>(b, a, (int) p.S, (int) p.HV, (int) p.heads,
                                                                (int) p.head_off, rec);
                        if (!ck(cudaGetLastError(), "the GDN recurrence permute", err)) return false;
                    }
                    const int64_t conv = p.channels * (p.d_conv - 1);
                    if (conv > 0) {
                        const unsigned grid = (unsigned) ((conv + kThreads - 1) / kThreads);
                        gdn_split_conv_kernel<<<grid, kThreads>>>(
                            b + rec, a + (int64_t) p.S * p.HV * p.S, (int) p.qk_channels, (int) p.head_off,
                            (int) p.S, (int) (p.d_conv - 1), conv);
                        if (!ck(cudaGetLastError(), "the GDN conv permute", err)) return false;
                    }
                    out.gdn_bytes += (uint64_t) p.floats() * sizeof(float);
                }
                ++gdn_ord;
            }
        }
        // the PLE's history and its token window: sequence state, replicated like the K/V.  A destination that
        // RECEIVES layers (a cut-to-cut move) only takes it when it owns layer 1, whose module the history is.
        if (ple_here) {
            if (!copy_span(dd, ds->ple_hist, ple_src_device, ple_src->ple_hist, ple_bytes, out.ple_bytes, err))
                return false;
            ds->ple_prev[0] = ple_src->ple_prev[0];
            ds->ple_prev[1] = ple_src->ple_prev[1];
            ds->ple_token = ple_src->ple_token;
        }

        if (!ck(cudaDeviceSynchronize(), "the redistribution's copies", err)) return false;
    }
    if (sliced) {
        // no destination covers the model, so the layer counts are the layers this move actually touched
        int qsa_moved = 0, gdn_moved = 0;
        for (int d = 0; d < n_dst; ++d) {
            int64_t dlo = 0, dhi = n_layers;
            dst_range(dst[d], n_layers, dlo, dhi);
            for (int64_t l = dlo; l < dhi; ++l) (is_qsa_layer(g, l) ? qsa_moved : gdn_moved) += 1;
        }
        out.qsa_layers = qsa_moved;
        if (want_gdn) out.gdn_layers = gdn_moved;
    }

    if (stats != nullptr) *stats = out;   // totals across the destination ranks - see the header
    return true;
}

}  // namespace strata::core
