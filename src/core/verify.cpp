// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/batch_rows.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/window_evict.hpp"
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include "strata/core/host_graph_launch.hpp"
#endif
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/gpu_busy.hpp"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/route_prior.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/tp_split.hpp"
#include "strata/core/tp_split_plan.hpp"
#include "strata/core/tp_blob_ranges.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <optional>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

#if defined(STRATA_GPU_METAL)
// The Metal backend's decode window (metal/src/engine/window.cpp): this file's init/run/commit hand over to it.
namespace strata::metal_engine {
void window_destroy(void* w);
void* window_create();
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err);
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err);
bool window_commit(void* w, int n_keep, std::string& err);
bool window_copy_logits(void* w, int t, float* host);
}  // namespace strata::metal_engine
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
inline bool g_lfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE"); return v != nullptr && v[0] == '1'; }(); return on; }

/// STRATA_POOL_ORDER=1: run the resident experts through `native_expert_grouped_q8_0` instead of
/// `native_expert_grouped`.  The two differ only in ARITHMETIC, not in result: the q8_0 path is a bit-exact
/// transcription of the CPU pool (same Q8_0 blocks, same fp32 FMA order, same hsum8 tree, glibc's SiLU expf),
/// while the q8_1 path keeps 16-lane fp16 partials and CUDA's `__expf`.  With it on, a resident expert and a
/// missed one return the same number, so changing which experts are resident no longer changes the tokens -
/// which is what has blocked raising residency for speed.  OFF by default: it changes the output tokens of the
/// existing path, so it must be opted into (and the acceptance run compares against the solo tokens WITH it on).
inline bool g_pool_order() {
    static const bool on = [] { const char* v = std::getenv("STRATA_POOL_ORDER"); return v != nullptr && v[0] == '1'; }();
    return on;
}
// STRATA_ROUTE_RESIDENT=<margin logits> (+ STRATA_ROUTE_RESIDENT_RANKS=lo-hi, default 6-9): residency-biased routing.
struct RouteResidentCfg;
RouteResidentCfg& route_resident_cfg();
struct RouteResidentCfg {
    float margin = 0.0f;
    int lo = 6, hi = 9;
    // #1578: one counter block per GPU.  A layer split inits a verifier per stage; a block allocated on whichever device
    // was current at the first init is unreachable from the other stage's kernels without P2P (illegal memory access).
    // The Verifier asks for its device's block in init (outside graph capture) and keeps the pointer: a launch never
    // picks the block by the host thread's current device.  nullptr (the allocation failed): the kernel runs without counters.
    std::mutex mu;
    std::map<int, unsigned long long*> blocks;
    unsigned long long* stats_on(int dev) {
        if (margin <= 0.0f || dev < 0) return nullptr;
        std::lock_guard<std::mutex> lk(mu);
        const auto it = blocks.find(dev);
        if (it != blocks.end()) return it->second;
        unsigned long long* p = nullptr;
        {
            const OnDevice on(dev);
            if (cudaMalloc((void**) &p, 8 * sizeof(unsigned long long)) != cudaSuccess ||
                cudaMemset(p, 0, 8 * sizeof(unsigned long long)) != cudaSuccess) {
                (void) cudaGetLastError();
                if (p != nullptr) cudaFree(p);
                p = nullptr;
                std::fprintf(stderr, "route-resident: the counters could not be allocated on GPU %d; it runs without them\n", dev);
            }
        }
        if (blocks.empty()) {
            std::atexit([] {
                RouteResidentCfg& c = route_resident_cfg();
                for (const auto& kv : c.blocks) {
                    if (kv.second == nullptr) continue;
                    unsigned long long h[8] = {};
                    if (cudaSetDevice(kv.first) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
                        cudaMemcpy(h, kv.second, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        (void) cudaGetLastError();
                        continue;
                    }
                    char who[32] = "";
                    if (c.blocks.size() > 1) std::snprintf(who, sizeof who, " (GPU %d)", kv.first);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T>8 (prompt reads): tail_nonres=%llu swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[0], h[1], h[0] ? 100.0 * h[1] / h[0] : 0.0, h[2], h[3]);
                    std::fprintf(stderr, "route-resident%s: margin=%g ranks=%d-%d windows T<=8 (decode): tail_nonres=%llu swaps=%llu (%.1f%%) nonres_entries before=%llu after=%llu\n",
                                 who, c.margin, c.lo, c.hi, h[4], h[5], h[4] ? 100.0 * h[5] / h[4] : 0.0, h[6], h[7]);
                }
            });
        }
        blocks[dev] = p;
        return p;
    }
};
RouteResidentCfg& route_resident_cfg() {
    static RouteResidentCfg c;
    static const bool parsed = [] {
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT")) c.margin = (float) std::atof(v);
        if (const char* v = std::getenv("STRATA_ROUTE_RESIDENT_RANKS")) std::sscanf(v, "%d-%d", &c.lo, &c.hi);
        return true;
    }();
    (void) parsed;
    return c;
}
inline bool g_lfuse_gate() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_GATE"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_lfuse_pair() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_PAIR"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_qdedup() { static const bool on = [] { const char* v = std::getenv("STRATA_VERIFY_QDEDUP"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
// S26 STRATA_QFUSE=1: activation q8_1 images written by their producers (the GDN output norm) - the same bytes
// fork F4 (Eddoursul): 2-4 token dense projections read an interleaved copy of the q8_1 activations (bitwise the same
// outputs); STRATA_MMVQ_IL=0 keeps native_mmvq's multi-column kernels
inline bool g_mmvq_il() { static const bool on = [] { const char* v = std::getenv("STRATA_MMVQ_IL"); return v == nullptr || v[0] != '0'; }(); return on; }
// 143-amdk: STRATA_QFUSE_HC - only the hyper-connection read writes its consumer's q8_1 image (not the GDN recurrence): gfx12 default (=0 / 1 forces off / on)
inline bool g_qfuse_hc() {
#if defined(STRATA_USE_HIP)
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QFUSE_HC");
        if (v != nullptr) return std::atoi(v) != 0;
        int dev = 0;
        cudaDeviceProp p;
        return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
               strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
    }();
    return on;
#else
    return false;
#endif
}
// STRATA_GRAPH_PRIO=1 (opt-in): the window graph's kernel nodes keep their capturing stream's
// priority (side_ high, sh_cs_ low).  Unset / 0: every node at the launch stream's priority, as before.
[[maybe_unused]] inline bool graph_prio_on() {   // (every HIP caller is guarded off)
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_GRAPH_PRIO"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
#endif
}
// STRATA_GRAPH_PRIO: the capture and graph queries by CUDA version (13 renamed the edge-data forms: the same split as
// verify_kernels.cu's and the PDL edge check above).  Nothing is returned for the edge data.
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
inline cudaError_t prio_capture_deps(cudaStream_t s, cudaStreamCaptureStatus* st, const cudaGraphNode_t** deps, size_t* n) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(s, st, nullptr, nullptr, deps, nullptr, n);
#elif CUDART_VERSION >= 12030
    return cudaStreamGetCaptureInfo_v3(s, st, nullptr, nullptr, deps, nullptr, n);
#else
    return cudaStreamGetCaptureInfo_v2(s, st, nullptr, nullptr, deps, n);
#endif
}
inline cudaError_t prio_node_deps(cudaGraphNode_t node, cudaGraphNode_t* deps, size_t* n) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependencies(node, deps, nullptr, n);
#elif CUDART_VERSION >= 12030
    return cudaGraphNodeGetDependencies_v2(node, deps, nullptr, n);
#else
    return cudaGraphNodeGetDependencies(node, deps, n);
#endif
}
#endif
// fork 34a35db + b063c13 (Eddoursul), STRATA_SIDE_MIXER=1 / =2 (opt-in): the mixer's work that reads only the layer's
// input on branches of the window's graph (bitwise the same outputs); 0 / unset = in line, and no branch stream is created
inline int g_side_mixer() { static const int m = [] { const char* v = std::getenv("STRATA_SIDE_MIXER"); return v ? std::max(0, std::min(2, std::atoi(v))) : 0; }(); return m; }
inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// #649 (HIP, opt-in A/Bs for the gfx1030 verify timeouts; CUDA never reads them)
//   STRATA_VERIFY_COHERENT=1  the handshake words and rows in explicitly coherent (fine-grained) host memory
//   STRATA_DOORBELL_STORE=1   the GPU stores each step's ring instead of read-modify-writing it over PCIe
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] != 0 && e[0] != '0';
}
#if defined(STRATA_USE_HIP)
const bool g_coherent = env_on("STRATA_VERIFY_COHERENT");
const bool g_doorbell_store = env_on("STRATA_DOORBELL_STORE");
#endif
const bool g_trace = env_on("STRATA_VERIFY_TRACE");
const bool g_no_multi_gr = env_on("STRATA_NO_MULTI_GR");   // #783 PR-g: per-token GR reads/writes and generic-T kernels again
const bool g_no_batch_kv = env_on("STRATA_NO_BATCH_KV_STEP");   // #783 PR-d: per-token K/V and indexer appends again
// STRATA_BATCH_SELF_COMMIT=1 (opt-in): a batch window whose rows are each another slot's one token keeps them itself,
// as a one-token window does (record_window): the GDN conv histories and states advance inside the window graph, and
// no commit graph runs after it.  Such a window's commit is unconditional (each slot keeps its one row), and the rest
// of the commit graph only repeats what the window wrote: the indexer tails are put back and appended again from the
// same rows, the PLE histories are put back to the window's own snapshot.
const bool g_batch_self_commit = env_on("STRATA_BATCH_SELF_COMMIT");
// STRATA_QSA_SLOTS=1 (opt-in): in a batch window, a QSA layer's block scores, top-k and attention run as one launch
// each over the window's rows (up to 8 per launch, each row reading its own slot's indexer keys and K/V) instead of
// one call per row; every row is computed by the same kernel with the same arithmetic as its own call.  Not with a
// streamed KV cache (a row's block resolve could evict the blocks an earlier row of the same slot selected).
const bool g_qsa_slots = env_on("STRATA_QSA_SLOTS");

// every row of a batch window is another slot's (no slot has proposal rows)
bool one_row_per_slot(const int* rows, int S) {
    for (int i = 0; i < S; ++i)
        for (int j = i + 1; j < S; ++j)
            if (rows[i] == rows[j]) return false;
    return S >= 1;
}
bool batch_self_commits(const int* rows, int S) { return g_batch_self_commit && one_row_per_slot(rows, S); }

// The shared expert runs on its own stream, forked off and joined back per layer (`sh_fork` in record_window).  The
// overlap pays off on CUDA, where it was added (cfd3b72).  On HIP a cross-stream event dependency costs more than the
// shared expert takes to run, so there the fork starts off: same binary, same 48,067-token prompt, three runs each on a
// Radeon AI PRO R9700 (gfx1201, ROCm 6.4.3), the fork off decodes at 62.9 t/s against 43.4 t/s with it on.  The
// profiler turns the fork off to time the stages, so a profiled run never showed that.  #816 reports the same fall on a
// gfx1030 (RX 6800, Windows, 42.5 -> 56.4 t/s).  STRATA_SH_STREAM overrides either default: `=1` forks, `=0` does not.
// Strix Halo (gfx1151, unified memory) is the exception: the fork pays there (+1.8% UD-IQ4_XS, +6.7% UD-Q4_K_XL decode
// at 8K, ids identical), so the gfx1151 arch table sets STRATA_SH_STREAM=1.  Read on first use, after that table ran.
bool sh_stream_on() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_SH_STREAM")) return v[0] != '0';
#if defined(STRATA_USE_HIP)
        return false;
#else
        return true;
#endif
    }();
    return on;
}

// A one-token window always keeps its token, so its graph advances the sequence state itself (the GDN conv history
// and recurrence state) and Verifier::commit launches no commit graph after it (eddoursul's fork, F7).  CUDA only:
// HIP keeps the commit graph after every window (STRATA_ONE_TOKEN_COMMIT=0 does too).
// The head's hyper-connection read for the window's tokens in one launch set (CUDA; HIP keeps gr_read per token;
// STRATA_HEAD_MIX_MULTI=0 does too): bitwise the same sums.
bool head_mix_multi_enabled() {
#if defined(STRATA_USE_HIP)
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");   // HIP: opt-in (=1)
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

// 143-amdk: the wait for a host flag folded into the copy that follows it (wait_copy_i32_from_mapped,
// copy_rows_from_mapped_after): same reads, same values, one kernel node fewer each.  A dependent kernel costs ~2.4 us in a
// gfx1201 graph whatever it does; measured on gfx12 only, so it is on there by default.  STRATA_FUSE_WAIT=0 / 1 forces it
// off / on (HIP builds; a CUDA build never fuses).
// the Turing series (STRATA_SM75_PERF=1, sm75_series.hpp) defaults of the CUDA build's switches below, read per device
bool device_is_sm75() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = ok && strata::sm75_default(maj, mnr) ? 2 : 1;   // a Turing card with STRATA_SM75_PERF=1
    }
    return per_dev[dev] == 2;
#endif
}

bool fuse_wait_on() {
#if defined(STRATA_USE_HIP)
    static const int env = [] {
        const char* v = std::getenv("STRATA_FUSE_WAIT");
        return v != nullptr ? (std::atoi(v) != 0 ? 1 : 0) : -1;
    }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        cudaDeviceProp p;
        per_dev[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#else
    // CUDA: STRATA_FUSE_WAIT=1 folds both waits (the plan's and the CPU rows'), =2 only the plan's, =0 neither.  Turing
    // default 2 (RTX 2080 Ti, Flash-Next Q2_0, two runs each: +0.6..+1.4 % decode); the rows' version makes every row
    // block of the copy poll the flag over PCIe and was -1.5 % there, so it stays opt-in.
    static const int env = [] { const char* v = std::getenv("STRATA_FUSE_WAIT"); return v != nullptr ? std::atoi(v) : -1; }();
    if (env >= 0) return env == 1 || env == 2;
    return device_is_sm75();
#endif
}
// 143-amdk's rows version of the folded wait (copy_rows_from_mapped_after): HIP with fuse_wait_on, CUDA STRATA_FUSE_WAIT=1
bool fuse_rows_on() {
#if defined(STRATA_USE_HIP)
    return fuse_wait_on();
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_FUSE_WAIT"); return v != nullptr && std::atoi(v) == 1; }();
    return on;
#endif
}

// 143-amdk: STRATA_FUSE_CONVAB=1 - the GDN conv and the a/b projections in one launch.  Opt-in: measured -0.3% on gfx1201 (the a/b dot chain is the longer one).
bool fuse_convab_on() {
#if defined(STRATA_USE_HIP)
    static const bool on = [] { const char* v = std::getenv("STRATA_FUSE_CONVAB"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
#else
    return false;
#endif
}

// 143-amdk: STRATA_FUSE_HITS - moe_hit_add folded into the combine (gfx12 default; =0 / 1 forces off / on)
bool fuse_hits_on() {
#if defined(STRATA_USE_HIP)
    static const int env = [] { const char* v = std::getenv("STRATA_FUSE_HITS"); return v != nullptr ? (std::atoi(v) != 0 ? 1 : 0) : -1; }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!per_dev[dev]) {
        cudaDeviceProp p;
        per_dev[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#else
    return false;
#endif
}

bool one_token_self_commit() {
#if defined(STRATA_USE_HIP)
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");   // HIP: opt-in (=1)
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool mapped(size_t bytes, void** h, void** d) {
#if defined(STRATA_USE_HIP)
    if (g_coherent) {
        if (hipHostMalloc(h, bytes, hipHostMallocMapped | hipHostMallocCoherent |
                                        (peer_portable() ? hipHostMallocPortable : 0)) != hipSuccess)
            return false;
        std::memset(*h, 0, bytes);
        return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
    }
#endif
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;   // multi-GPU: portable only with a peer - the peer card writes its rows into them
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
// #649: the host side of STRATA_VERIFY_TRACE - a ring of the last kTraceN handshake events of every verifier
struct TraceEv {
    int64_t t_ns;
    const void* who;
    int64_t window, step, layer, aux;
    uint32_t seq, flag, a, b;
    const char* what;
};
constexpr uint64_t kTraceN = 4096;
TraceEv g_trace_ring[kTraceN];
std::atomic<uint64_t> g_trace_next{0};
int64_t trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
// The overlapped split's hand-off wait on the GPU is bounded: STRATA_SPLIT_WAIT_MS (default 30000, 0 = no bound).
// The host's own per-layer wait (20 s, #267) normally ends a stalled window first; this bound is the backstop for
// a host that is gone (a killed process must not leave a kernel spinning on its card).
const unsigned long long g_split_wait_ns = [] {
    const char* e = std::getenv("STRATA_SPLIT_WAIT_MS");
    return (unsigned long long) (e != nullptr ? std::atoll(e) : 30000) * 1000000ull;
}();
// test hook: STRATA_TEST_HANDOFF_DROP=N - the writing stage's N-th window (1-based) leaves its hand-off flag down, as
// if the publish were lost; the reading stage's bounded wait (or the host's) must then end the window cleanly.
const int64_t g_test_drop = [] {
    const char* e = std::getenv("STRATA_TEST_HANDOFF_DROP");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

// STRATA_TP_VEC=N: floats per dump site (0 = the scalar dumps only).  Both the solo and the TP window read it,
// so their vector dumps have the same length and can be diffed element by element.
int Verifier::tp_vec_len() {
    static const int n = [] {
        const char* v = std::getenv("STRATA_TP_VEC");
        const int k = v != nullptr ? std::atoi(v) : 0;
        // The sites are [T][N] buffers, so a width below T*N silently drops whole tokens: N alone (2560) covers
        // only token 0, and the 2048 this used to default to hid both token 1 and columns 2048..2559 - an
        // "l0 is bit-exact" reading off that dump was only 80% of one token.  The ceiling is the largest window
        // (6 tokens x 2560 = 15360) so any window can be dumped in full.
        return k > 0 && k <= 16384 ? k : 0;
    }();
    return n;
}

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    trace_ev("RELEASE", -1, -1, timeout_ms);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_, hflag_in_h_})   // + an overlapped split's hand-off wait
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_, copy_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                trace_ev("RELEASE-NOT-DRAINED", -1, -1, (int64_t) ms_since(t0));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    trace_ev("RELEASE-DRAINED", -1, -1, (int64_t) ms_since(t0));   // aux: ms the GPU took to finish once released
    return true;
}

void Verifier::trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const {
    if (!g_trace) return;
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    TraceEv& e = g_trace_ring[g_trace_next.fetch_add(1) % kTraceN];
    e = {trace_now_ns(), this, windows, step, layer, aux, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), what};
}

void Verifier::trace_dump(std::FILE* f) const {
    if (!g_trace || f == nullptr) return;
    static const char* names[kProfPer] = {"pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx",
                                          "k/v rope", "kv append", "q", "scores+topk", "kv-resolve", "attention",
                                          "gate", "", "out-proj", "router+ring", "shared", "waitA", "VRAM hits", "waitB",
                                          "PCIe grp", "waitCPU", "combine", "", "head", "", "", "", "", "", ""};
    const uint64_t end = g_trace_next.load();
    const uint64_t n = end < 160 ? end : 160;
    std::fprintf(f, "strata verify trace (#649): the last %llu handshake events (host clock, us before the last; "
                    "words: seq = GPU rang, flag = CPU served, A = plan, B = copies)\n", (unsigned long long) n);
    const int64_t t_last = n ? g_trace_ring[(end - 1) % kTraceN].t_ns : 0;
    for (uint64_t i = end - n; i < end; ++i) {
        const TraceEv& e = g_trace_ring[i % kTraceN];
        if (e.what == nullptr) continue;
        std::fprintf(f, "  %10.1f %-20s v%p window %lld step %lld layer %lld aux %lld | seq %u flag %u A %u B %u\n",
                     (double) (t_last - e.t_ns) / 1000.0, e.what, e.who, (long long) e.window, (long long) e.step,
                     (long long) e.layer, (long long) e.aux, e.seq, e.flag, e.a, e.b);
    }
    if (trace_h_ == nullptr || g_ == nullptr) return;
    // the breadcrumbs: every stamp point the GPU passed in this window (ns of the GPU clock after the window's first)
    unsigned long long t0 = ~0ull;
    for (size_t i = 0; i < trace_n_; ++i) {
        const unsigned long long t = *(const volatile unsigned long long*) (trace_h_ + i);
        if (t != 0 && t < t0) t0 = t;
    }
    if (t0 == ~0ull) {
        std::fprintf(f, "strata verify trace: no GPU breadcrumb in this window (the GPU never started it)\n");
        return;
    }
    int64_t last_l = -1;
    for (int64_t l = 0; l <= g_->n_layers; ++l)
        for (int i = 0; i < kProfPer; ++i)
            for (int grp = 0; grp < 2; ++grp)
                if (trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)] != 0) last_l = l;
    std::fprintf(f, "strata verify trace: GPU breadcrumbs (us after the window's first), the GPU got as far as layer "
                    "%lld:\n", (long long) last_l);
    for (int64_t l = std::max<int64_t>(0, last_l - 2); l <= std::min<int64_t>(g_->n_layers, last_l + 1); ++l)
        for (int grp = 0; grp < 2; ++grp) {
            std::string line;
            char b[64];
            for (int i = 0; i < kProfPer; ++i) {
                const unsigned long long t = trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)];
                if (t == 0) continue;
                std::snprintf(b, sizeof b, " %s(%d)=%.1f", names[i], i, (double) (t - t0) / 1000.0);
                line += b;
            }
            if (!line.empty())
                std::fprintf(f, "  layer %lld group %d:%s\n", (long long) l, grp, line.c_str());
        }
    std::fflush(f);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    trace_dump(f);   // #649: STRATA_VERIFY_TRACE=1 only
}


Verifier::~Verifier() {
#if defined(STRATA_GPU_METAL)
    strata::metal_engine::window_destroy(metal_);
    metal_ = nullptr;
#endif
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    if (sh_cs_) cudaStreamSynchronize(sh_cs_);
    if (pcie_st_) cudaFree(pcie_st_);
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    for (auto& kv : exec_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    if (arena_b_) cudaFree(arena_b_);
    if (h_commitb_) cudaFreeHost(h_commitb_);
    if (qcnt_) cudaFree(qcnt_);
    if (d_spec_) cudaFree(d_spec_);
    if (logit_bias_device_) cudaFree(logit_bias_device_);
    if (cs_ && cs_ != ext_stream_) cudaStreamDestroy(cs_);   // set_stream: the stage's stream, shared, not ours
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (side_) cudaStreamDestroy(side_);
    if (side2_) cudaStreamDestroy(side2_);
    for (cudaEvent_t e : {mfork_, mjoin_, pfork_, pjoin_})
        if (e) cudaEventDestroy(e);
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (ev_done_) cudaEventDestroy(ev_done_);
    if (ev_commit_) cudaEventDestroy(ev_commit_);
    if (prof_pin_) cudaFreeHost(prof_pin_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    for (cudaStream_t s : df_side_)
        if (s) cudaStreamDestroy(s);
    if (df_fork_) cudaEventDestroy(df_fork_);
    for (cudaEvent_t e : df_join_)
        if (e) cudaEventDestroy(e);
    if (arena_) cudaFree(arena_);
    if (hcount_d_) cudaFree(hcount_d_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_plan_err_, h_nf_, herr_h_, hdrop_h_, h_bal_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

// STRATA_PCIE_TIMING: turn this window's stamp pairs into ms_pcie.  Only call it once cs_ has been synchronized:
// every pair brackets a fetch on that stream, so the stamps are final.
void Verifier::drain_pcie_timing() {
    if (!pcie_timed_) return;
    if (pcie_pairs_ == 0 || pcie_st_ == nullptr) {
        static bool said = false;   // a zero here means the capture recorded no fetch at all: say WHY once
        if (!said) {
            said = true;
            std::fprintf(stderr,
                         "strata pcie timing: the recorded window carries no stamp pairs (pcie_mode %d, pcie_n %d)\n",
                         (int) sink_.pcie_mode, pcie_n_);
        }
        return;
    }
    const size_t words = (size_t) 2 * pcie_pairs_;
    if (pcie_h_.size() < words) return;
    if (cudaMemcpy(pcie_h_.data(), pcie_st_, words * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();   // a window cut short by an error: count nothing
        return;
    }
    double sum = 0;
    size_t nonmono = 0;
    for (size_t i = 0; i < (size_t) pcie_pairs_; ++i) {
        const unsigned long long t0 = pcie_h_[2 * i], t1 = pcie_h_[2 * i + 1];
        if (t1 > t0) sum += (double) (t1 - t0) / 1e6;   // %globaltimer counts ns (gpu_stamp normalizes HIP's too)
        else ++nonmono;
    }
    static bool said = false;   // first read-back of a stamped window: the numbers, once
    if (!said) {
        said = true;
        std::fprintf(stderr, "strata pcie timing: %d pairs, first t0=%llu t1=%llu (%zu non-monotonic), %.3f ms\n",
                     pcie_pairs_, pcie_h_[0], pcie_h_[1], nonmono, sum);
    }
    ms_pcie += sum;
}

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_logit_bias(bias, err)) return false;
    if (le_ < g_->n_layers) return true;
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias vocabulary size mismatch"; return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if ((!logit_bias_device_ && cudaMalloc((void**) &logit_bias_device_, bytes) != cudaSuccess) ||
            cudaMemcpy(logit_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "cannot upload logit_bias"; return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    rr_stats_ = route_resident_cfg().stats_on(device_);   // STRATA_ROUTE_RESIDENT: this GPU's counters, outside graph capture
    strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    strata::kernels::fused_gr_persist_init();   // opt-in STRATA_HC_PERSIST=1 (HIP)
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
#if defined(STRATA_GPU_METAL)
    if (metal_ == nullptr) metal_ = strata::metal_engine::window_create();
    return strata::metal_engine::window_init(metal_, g, ss, max_t, ss.qsa_states[ss.qsa_primary()].max_cells, &n_vocab_,
                                             err);
#endif
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped((size_t) strata::kernels::kVerifyMaxT * strata::kernels::kNonfiniteBlocks * 4, (void**) &h_nf_,
                     (void**) &m_nf_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_plan_err_, (void**) &m_plan_err_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // STRATA_TP_DUMP=1: a per-layer copy of the residual's head, to diff a TP run against a single-card one.
    // STRATA_TP_VEC=N (>0) appends N floats per dump site past kTpVecOff: the head of a layer whose values are
    // near zero says nothing about the layer, and a per-layer scalar cannot tell a wrong sum from a wrong shape.
    if (std::getenv("STRATA_TP_DUMP") != nullptr) {
        // the vectors need (n_layers + 1) rows: the head's input is dumped as one more layer's site 2.  They start
        // at kTpVecOff, not right after the scalars, so the block is kTpVecOff + vec floats.
        const int64_t vec = tp_vec_len() > 0 ? (g.n_layers + 1) * kTpVecSites * tp_vec_len() : 0;
        const int64_t nfl = vec > 0 ? kTpVecOff + vec : (int64_t) g.n_layers * 8 + 564;
        if (!mapped((size_t) nfl * sizeof(float), (void**) &h_tpdump_, (void**) &m_tpdump_)) {
            err = "verify: the TP diagnosis buffer could not be allocated";
            return false;
        }
    }
    if (bal_ != nullptr) {   // set_pcie_balance: four stamp words per (layer, group)
        if (!mapped((size_t) g.n_layers * 2 * 4 * 8, (void**) &h_bal_, (void**) &m_bal_)) {
            err = "verify: mapped balance stamps allocation failed";
            return false;
        }
        bal_p_.assign((size_t) g.n_layers * 2, 0.0);
        bal_nc_.assign((size_t) g.n_layers * 2, 0);
        bal_np_.assign((size_t) g.n_layers * 2, 0);
        bal_ok_.assign((size_t) g.n_layers * 2, 0);
        bal_->on = true;
    }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    // | pad | dst2(kStagingBlobs u64)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        dst2_off_ = (ptr_off + 4 * cap + (cap + 1) + 1 + 1) & ~1ll;
        plan_i32_ = dst2_off_ + 2 * kStagingBlobs;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.dst2 = (unsigned long long*) (h_plan_ + dst2_off_);
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        xil_ = b.take<uint8_t>(strata::kernels::native_q8_1_il_bytes(max_in, 4));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        ple_key_ = b.take<float>(T * (uint64_t) strata::kernels::NG_HC_DIM); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    if (hflag_out_d_ != nullptr && hcount_d_ == nullptr) {   // the overlapped split's hand-off counter
        if (cudaMalloc((void**) &hcount_d_, 64) != cudaSuccess) { err = "verify: hand-off counter allocation failed"; return false; }
        cudaMemset(hcount_d_, 0, 64);
        if (!mapped(64, (void**) &hdrop_h_, (void**) &hdrop_d_)) { err = "verify: hand-off test word allocation failed"; return false; }
    }
    if (hflag_in_d_ != nullptr && herr_h_ == nullptr &&
        !mapped(64, (void**) &herr_h_, (void**) &herr_d_)) { err = "verify: hand-off error word allocation failed"; return false; }
    if (g_trace && trace_h_ == nullptr) {   // #649: the breadcrumbs, mapped so they read while the GPU hangs
        trace_n_ = (size_t) (g.n_layers + 1) * kProfPer * 2;
        if (!mapped(trace_n_ * 8, (void**) &trace_h_, (void**) &trace_m_)) {
            (void) cudaGetLastError();
            trace_h_ = trace_m_ = nullptr;
            trace_n_ = 0;
        }
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr, "strata verify trace (#649): on; coherent words %s, doorbell %s, HIP_HOST_COHERENT=%s "
                             "HSA_ENABLE_SDMA=%s GPU_MAX_HW_QUEUES=%s\n", g_coherent ? "explicit" : "default",
                     g_doorbell_store ? "stored" : "incremented", std::getenv("HIP_HOST_COHERENT") ? std::getenv("HIP_HOST_COHERENT") : "-",
                     std::getenv("HSA_ENABLE_SDMA") ? std::getenv("HSA_ENABLE_SDMA") : "-",
                     std::getenv("GPU_MAX_HW_QUEUES") ? std::getenv("GPU_MAX_HW_QUEUES") : "-");
#else
        std::fprintf(stderr, "strata verify trace (#649): on\n");
#endif
    }
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    pcie_timed_ = std::getenv("STRATA_PCIE_TIMING") != nullptr;
    if (pcie_timed_ && pcie_st_ == nullptr) {
        const size_t words = (size_t) 2 * kPcieTimingMax;
        if (cudaMalloc((void**) &pcie_st_, words * sizeof(unsigned long long)) != cudaSuccess) {
            pcie_timed_ = false; pcie_st_ = nullptr; cudaGetLastError();
            std::fprintf(stderr, "strata pcie timing: cudaMalloc failed, off\n");
        } else {
            cudaMemset(pcie_st_, 0, words * sizeof(unsigned long long));
            pcie_h_.assign(words, 0);
            std::fprintf(stderr, "strata pcie timing: armed on device %d from layer %d (mode %d)\n",
                         device_, (int) le_, (int) sink_.pcie_mode);
        }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    {
        const int32_t one = 1;
        if (cudaMemcpy(one_, &one, sizeof one, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the arena could not be set";
            return false;
        }
    }
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    // STRATA_GRAPH_PRIO=1 (opt-in): the window's streams carry priorities that their captured kernel
    // nodes keep (the window graph is instantiated with cudaGraphInstantiateFlagUseNodePriority): the GDN gates / QSA
    // indexer branch (side_) at the device's highest, cs_ and side2_ in the middle, the shared expert (sh_cs_) at the
    // lowest (= the default).  The indexer chain's 1-block kernels are then dispatched ahead of the q GEMV's queued
    // blocks, and the shared expert behind the experts.  Unset: every stream at the default priority, as before.
    int gp_lo = 0, gp_hi = 0;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    const bool gprio = graph_prio_on() && cudaDeviceGetStreamPriorityRange(&gp_lo, &gp_hi) == cudaSuccess && gp_lo != gp_hi;
    if (graph_prio_on() && !gprio) (void) cudaGetLastError();   // (only a failed range query's error: unset clears nothing)
#else
    const bool gprio = false;   // (HIP: the default priority, as make_stream's below)
#endif
    const int gp_mid = gprio ? (gp_lo + gp_hi) / 2 : 0;
    if (gprio) std::fprintf(stderr, "strata verify: STRATA_GRAPH_PRIO: stream priorities side %d, cs / side2 %d, shared expert %d\n",
                            gp_hi, gp_mid, gp_lo);
    // STRATA_PIPED_CONCURRENT=1 (opt-in): a prompt can run on the same GPU while the batch windows run, so the windows'
    // streams get the device's highest priority (HIP: the default priority, as the drafter's)
    auto make_stream = [&](cudaStream_t* s, int gp = 0) -> cudaError_t {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
        static const bool hi = [] {   // (STRATA_PIPED_CONCURRENT_PRIO=0: the default priority)
            const char* v = std::getenv("STRATA_PIPED_CONCURRENT");
            const char* p = std::getenv("STRATA_PIPED_CONCURRENT_PRIO");
            return v != nullptr && std::atoi(v) != 0 && (p == nullptr || std::atoi(p) != 0);
        }();
        int lo = 0, top = 0;
        if (hi) {
            if (cudaDeviceGetStreamPriorityRange(&lo, &top) == cudaSuccess &&
                cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, top) == cudaSuccess)
                return cudaSuccess;
            cudaGetLastError();   // (only the failed priority call's error: the default path clears nothing)
        }
        if (gprio) {   // STRATA_GRAPH_PRIO: this stream's own priority (gp)
            if (cudaStreamCreateWithPriority(s, cudaStreamNonBlocking, gp) == cudaSuccess) return cudaSuccess;
            cudaGetLastError();
        }
#else
        (void) gp;
#endif
        return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking);
    };
    bool copy_made = false;
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    // engine-latency, STRATA_COPY_HIP_PRIO=1 (opt-in): the copy stream at high priority, so it gets a hardware queue of
    // its own (ROCm hands queues out per priority).  Sharing one with the window's stream, --pcie-mode dma's copies and
    // their host function queue behind the window's graph, which spins on the flag they raise (a timeout at layer 1).
    {
        static const bool cp = [] { const char* v = std::getenv("STRATA_COPY_HIP_PRIO"); return v != nullptr && std::atoi(v) != 0; }();
        int lo = 0, hi = 0;
        if (cp && hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess)
            copy_made = hipStreamCreateWithPriority(&copy_, hipStreamNonBlocking, hi) == hipSuccess;
        (void) cudaGetLastError();
    }
#endif
    if (!copy_made && make_stream(&copy_) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (ext_stream_ != nullptr) cs_ = ext_stream_;   // set_stream (pipelined windows): the stage's shared stream
    else if (make_stream(&cs_, gp_mid) != cudaSuccess) cs_ = nullptr;
    if (cs_ == nullptr || make_stream(&sh_cs_, gp_lo) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
        // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
        // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
        const char* v = std::getenv("STRATA_DF_BRANCH");
        df_branch_ = v != nullptr && std::atoi(v) != 0;
        if (df_branch_ && df_fork_ == nullptr) {
            bool okb = cudaEventCreateWithFlags(&df_fork_, cudaEventDisableTiming) == cudaSuccess;
            for (int i = 0; i < 2 && okb; ++i)
                okb = cudaStreamCreateWithFlags(&df_side_[i], cudaStreamNonBlocking) == cudaSuccess &&
                      cudaEventCreateWithFlags(&df_join_[i], cudaEventDisableTiming) == cudaSuccess;
            if (!okb) { cudaGetLastError(); df_branch_ = false; }
        }
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    ar_off_ = false;
    h_res_ = hits.h_res;
    if (h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        const char* v_ar = std::getenv("STRATA_VERIFY_ALL_RESIDENT");
        if (v_ar == nullptr || std::atoi(v_ar) != 0) {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    if (g_side_mixer() > 0 &&
        ((gprio ? make_stream(&side_, gp_hi) : cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking)) != cudaSuccess ||
        cudaEventCreateWithFlags(&mfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&mjoin_, cudaEventDisableTiming) != cudaSuccess ||
        (gprio ? make_stream(&side2_, gp_mid) : cudaStreamCreateWithFlags(&side2_, cudaStreamNonBlocking)) != cudaSuccess ||
        cudaEventCreateWithFlags(&pfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&pjoin_, cudaEventDisableTiming) != cudaSuccess)) {
        err = "verify: the mixer branch's stream / events could not be created";
        return false;
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        // (not with set_always_publish: the device table may lag the host's while windows are in flight)
        device_plan_ = !all_resident_ && !always_publish_ && (v != nullptr && std::atoi(v) != 0);
    }
    // (halo's STRATA_VERIFY_RESIDENT=1 is this window's all_resident_ graph above, which 0.1.39 has on by default)
    if (g_qfuse() || g_qfuse_hc()) {   // S26: the HC read's q8_1 group counters, zeroed once (each launch leaves them zero)
        if (cudaMalloc((void**) &qcnt_, sizeof(unsigned) * (size_t) (g.n_embd / 32)) != cudaSuccess ||
            cudaMemset(qcnt_, 0, sizeof(unsigned) * (size_t) (g.n_embd / 32)) != cudaSuccess) {
            cudaGetLastError();
            if (qcnt_) cudaFree(qcnt_);
            qcnt_ = nullptr;
        }
    }
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); all_resident_ = false; device_plan_ = false; }
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers%s\n", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    // STRATA_PCIE_TIMING: once this returns, the stamp pairs are graph nodes - every launch of the graph rewrites
    // them, so the window's drain reads that launch.  A scope guard because the function has an early return per
    // error, and a half-recorded graph must not leave a count behind.
    struct PcieStampScope {
        int* n;
        int* pairs;
        ~PcieStampScope() {
            if (*n > 0) *pairs = *n;
            static bool said = false;   // what the capture actually recorded, once per process
            if (!said) {
                said = true;
                std::fprintf(stderr, "strata pcie timing: the recorded window carries %d stamp pairs\n", *n);
            }
            *n = 0;
        }
    } pcie_stamp_scope{&pcie_n_, &pcie_pairs_};
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2 && !batch_rec_) ? 2 : 1;   // a batch window is one group
    // Pipelined windows must wait for the host verdict/snapshot fence before mutating state.
    const bool self_commit = T == 1 && !always_publish_ && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    // fork 34a35db: the GDN gates and the QSA indexer beside the projections (STRATA_SIDE_MIXER=1; 0 / unset = in line)
    const int side_mixer_mode = g_side_mixer();
    const bool side_mixer = side_mixer_mode >= 1;
    const bool side_proj = side_mixer_mode >= 2 && !batch_rec_;   // fork b063c13: the GDN gate projection, the QSA queries
    auto branch = [&](cudaEvent_t e, cudaStream_t from, cudaStream_t to, const char* what) {
        if (to == side_) note_prio_fork(from);   // STRATA_GRAPH_PRIO: where the high-priority branch leaves cs
        if (cudaEventRecord(e, from) == cudaSuccess && cudaStreamWaitEvent(to, e, 0) == cudaSuccess) return true;
        err = std::string("verify: the mixer branch could not ") + what;
        return false;
    };
    // a batch window's rows each read their own slot's indexer: in line there
    const bool sidem = side_mixer && !batch_rec_;
    auto stamp = [&](int64_t l, int i, int grp) {
        if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
        if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2 + grp), cs);   // #649
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    // S26 STRATA_LFUSE=1: where the shared expert's gate / scale fusions apply (the paths they replace are the ones taken)
    auto lfuse_on = [&](int n) {
        return g_lfuse() && ar_on() && native_moe_combine_enabled() && dec_batch && n > 1 && n <= 8 &&
               shared_expert_native_bf16_enabled();
    };
    bool sg_gated_[2] = {false, false};   // per group: the combine applies the shared gate
    if (!batch_rec_) groups_[T] = G;
    // Programmatic dependent launch (pdl.hpp; sm_90+, opt-in: STRATA_DF_PDL=1): the window's quantizations and dense
    // projections may start while the kernel before them finishes, loading their weights before they wait for its
    // output.  Only a launch whose predecessors in the graph are all kernels gets programmatic edges; every value is
    // the same as without them.
    struct PdlScope {
        bool prev;
        explicit PdlScope(bool on) : prev(pdl_scope()) { pdl_scope() = on; }
        ~PdlScope() { pdl_scope() = prev; }
    } pdl_window(G == 1 && pdl_supported());
    // STRATA_DF_BRANCH: mixer work that reads only the layer's input runs on a side stream inside the graph.  fork(i)
    // makes side i depend on everything captured on `cs` so far, join(i) makes `cs` depend on everything captured on
    // side i: a branch never races with what came before its fork or after its join.  Not in a split or batch window,
    // nor under the stage profiler (whose stamps are on `cs`).
    const bool br = df_branch_ && G == 1 && !batch_rec_ && !prof_on_ && !side_mixer;   // STRATA_SIDE_MIXER wins
    auto fork = [&](int i) {
        cudaEventRecord(df_fork_, cs);
        cudaStreamWaitEvent(df_side_[i], df_fork_, 0);
    };
    auto join = [&](int i) {
        cudaEventRecord(df_join_[i], df_side_[i]);
        cudaStreamWaitEvent(cs, df_join_[i], 0);
    };
    const int64_t nQall = g.n_qsa_layers();
    // a batch window: row t is slot t, whose state lives in its own session
    auto slot_ss = [&](int t) -> SessionState& { return batch_rec_ ? *slots_[(size_t) brow_[t]] : ss; };
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows
    // STRATA_BATCH_SELF_COMMIT: the window keeps its rows (no commit graph, nothing to snapshot for one)
    const bool keep_rows = batch_rec_ && batch_self_commits(brow_, T);

    // ---- the window's inputs, from mapped staging.  With branches the steps and positions (first read by the first
    // layer's mixer) cross PCIe on side 1 beside the embedding and the first hyper-connection read.
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    bool inputs_pending = false;
    cudaStream_t in_s = cs;
    if (br) {
        fork(1);
        in_s = df_side_[1];
        inputs_pending = true;
    }
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, in_s);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), in_s);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        // the overlapped split: this graph was launched before the previous stage finished - wait for its hand-off
        if (hflag_in_d_ != nullptr && !batch_rec_) wait_handoff(hflag_in_d_, 1u, herr_d_, g_split_wait_ns, cs);
        copy_from_mapped(R_, hin, (int64_t) T * HC * N, cs);
        copy_from_mapped(bo_, hin + (size_t) T * HC * N, (int64_t) T * N, cs);
        copy_from_mapped(inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (grp == 0) {
                if (ar_on()) wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            static const bool ple_batch_env = [] {
                const char* e = std::getenv("STRATA_PLE_BATCH");
                return !e || e[0] != '0';
            }();
            const bool ple_batch_kv = ple_batch_env && dec_batch && n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                      ss.ple.w.value_bf16 != nullptr && ple_native_bf16_enabled() &&
                                      ple_native_postops_enabled();
            // S25 (STRATA_PLE_BATCH=1): the key and value projections depend only on each row's n-gram embedding, so
            // the group's rows run through the multi-row forms of the same kernels (the weights read once); every
            // row's arithmetic is the single-row call's. The history-dependent rest stays row by row.
            static const bool ple_batch = [] { const char* e = std::getenv("STRATA_PLE_BATCH"); return e && e[0] == '1'; }();
            PleWeights pw = ss.ple.w;
            const bool nkey = pw.key_native_data != nullptr && pw.key_bf16 == nullptr;
            const bool batched = !ple_batch_kv && ple_batch && n > 1 && (pw.key_bf16 != nullptr || nkey) && ple_native_bf16_enabled();
            if (ple_batch_kv) {
                try {
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.key_bf16,
                                              xn_ + (size_t) tb * HC * N, HC * N, N, HC * N, n, cs);
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.value_bf16,
                                              z_ + (size_t) tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE multi: ") + e.what();
                    return false;
                }
            } else if (batched) {
                try {
                    if (pw.key_bf16 != nullptr)
                        bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.key_bf16, ple_key_ + (size_t) tb * NG_HC_DIM,
                                                  NG_HC_DIM, N, NG_HC_DIM, n, cs);
                    else {
                        native_quantize_q8_1(ple_ + tb * N, xq_, (int) N, n, cs);
                        native_mmvq(pw.key_native_type, pw.key_native_data, xq_, ple_key_ + (size_t) tb * NG_HC_DIM,
                                    (int) N, NG_HC_DIM, n, cs);
                    }
                    bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.value_bf16, ple_val_ + tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE batch: ") + e.what();
                    return false;
                }
            }
            // STRATA_DF_PLE (opt-in: STRATA_DF_PLE=1): the window's post-projection steps in one pass, bitwise the per-token loop
            // below (the norms divide at run time as its norms do; the history advances token by token), which also
            // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
            // hyper-connection read's `lo_`, which the read below rewrites.
            static const bool ple_post_env = [] {
                const char* e = std::getenv("STRATA_DF_PLE");
                return e != nullptr && std::atoi(e) != 0;
            }();
            const bool ple_post = ple_batch_kv && ple_post_env && !batch_rec_;
            // #783 PR-g (stuchapin909): the window's residual writes in one launch (STRATA_NO_MULTI_GR=1: per token)
            const bool multi_write = !ple_post && dec_batch && !g_no_multi_gr && n > 1;
            if (multi_write) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            if (ple_post) {
                float* qn = parts_ + (size_t) tb * K * N;           // n x 10240 (K * N = 25600 per row)
                float* gated = hit_out_ + (size_t) tb * K * N;      // n x 10240
                float* pgate = lo_ + (size_t) tb * g.hc_lr;         // n x 4
                try {
                    if (gr_native_mmvf_enabled())   // gr_write's native write for the n rows in one launch
                        native_gr_post_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, Rt(tb), (int) N, (int) HC, n, HC * N,
                                             N, HC, cs);
                    else
                        for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                    native_ple_postops_batch_snap(xn_ + (size_t) tb * HC * N, Rt(tb), z_ + (size_t) tb * N, ss.ple.hist,
                                                  ss.ple.w, qn, gated, pgate, n, hist_snap_ + (size_t) tb * HS, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE (window): ") + e.what();
                    return false;
                }
            } else
            for (int t = tb; t < te; ++t) {
                if (batched) { pw.pre_key = ple_key_ + (size_t) t * NG_HC_DIM; pw.pre_value = ple_val_ + t * N; }
                if (!multi_write) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? slot_ss(t).ple_hist : ss.ple.hist;   // the row's own history
                try {
                    if (ple_batch_kv) {
                        ple_block_projected(xn_ + (size_t) t * HC * N, z_ + (size_t) t * N, Rt(t), hist,
                                            ss.ple.w, po, ss.ple.scratch, cs);
                    } else {
                        ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    }
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                if (!keep_rows) copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                a.q8_inject = (const uint8_t*) wi[half]->hc_q8;
                a.s_down = wd[half]->q8_scale; a.s_up = wu[half]->q8_scale;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
                if (qcnt_ != nullptr) {   // S26 STRATA_QFUSE: the consumer's q8_1 image written by the read itself
                    a.q8_cnt = qcnt_;
                    if (half == 0) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                    else if (strata::kernels::cpu::expert_layout().native) a.q8_mixed = nat_xq_ + (size_t) t * (N / 32) * 36;
                }
            }
            if (strata::kernels::fused_gr_read_persist(fa, n, cs)) return false;   // STRATA_HC_PERSIST=1; q8_mixed is never set there
            return fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        const bool q8_attn = gr_read_group(0, pending, inj2_, inj_);   // true: xq_ holds mixed's q8_1 (STRATA_QFUSE)
        {   // measurement only: STRATA_DUMMY_NODES=k adds k empty kernel nodes per layer (what one node costs in a window)
            static const int dummy = [] { const char* e = std::getenv("STRATA_DUMMY_NODES"); return e != nullptr ? std::atoi(e) : 0; }();
            for (int d = 0; d < dummy; ++d) gpu_noop(cs);
        }
        if (inputs_pending) {   // the window's steps and positions, before the first mixer reads them
            join(1);
            inputs_pending = false;
        }
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        bool il_ready = false;   // xil_ holds xq_'s interleaved copy (reset whenever xq_ is rewritten)
        // a projection input's q8_1 blocks and their interleaved copy in one launch (native_quantize_q8_1_il, the same
        // bytes): Turing default (+0.5..+1 % decode, two launches fewer per GDN layer and per QSA layer); STRATA_QIL=0 / 1
        // forces it off / on
        auto quant_in = [&](const float* x, int n_in) {
#if !defined(STRATA_USE_HIP)
            static const int qil_env = [] { const char* v = std::getenv("STRATA_QIL"); return v != nullptr ? std::atoi(v) : -1; }();
            const bool qil = qil_env >= 0 ? qil_env != 0 : device_is_sm75();
            if (qil && n >= 2 && n <= 4 && g_mmvq_il()) {
                strata::kernels::native_quantize_q8_1_il(x, xq_, xil_, n_in, n, cs);
                il_ready = true;
                return;
            }
#endif
            native_quantize_q8_1(x, xq_, n_in, n, cs);
            il_ready = false;
        };
        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {
            if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(w->native_type, n, n_out)) {
                if (!il_ready && n >= 2) {   // one column reads the plain q8_1 blocks
                    strata::kernels::native_q8_1_interleave(xq_, xil_, n_in, n, cs);
                    il_ready = true;
                }
                strata::kernels::native_mmvq_il(w->native_type, w->native_data, xq_, xil_, out, n_in, n_out, n, cs);
            } else {
                native_mmvq(w->native_type, w->native_data, xq_, out, n_in, n_out, n, cs);
            }
        };
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                if (sidem) {   // the gates (a few blocks, latency-bound) beside the projections, joined below
                    if (!branch(mfork_, cs, side_, "fork")) return false;
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, side_);
                    note_prio_tail(side_, 1);
                    if (cudaEventRecord(mjoin_, side_) != cudaSuccess) { err = "verify: the GDN gates' branch could not join"; return false; }
                }
                if (!q8_attn) quant_in(xm, (int) N);
                if (side_proj) {   // the gate projection (compute-bound for i-quants) beside qkv and the conv
                    // the branch reads F4's interleaved copy too (written on cs before the fork; bitwise native_mmvq's
                    // output), and qkv is recorded before the branch's launch, so its grid is dispatched first and the
                    // conv / recurrence that wait for it start sooner (z is read only after the recurrence)
                    const bool g_il = g_mmvq_il() && strata::kernels::native_mmvq_il_supported(wg->native_type, n, (int) ZV);
                    if (g_il && !il_ready && n >= 2) {
                        strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, n, cs);
                        il_ready = true;
                    }
                    if (!branch(pfork_, cs, side2_, "fork")) return false;
                    mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);
                    if (g_il) strata::kernels::native_mmvq_il(wg->native_type, wg->native_data, xq_, xil_, z_ + (size_t) tb * ZV,
                                                              (int) N, (int) ZV, n, side2_);
                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, side2_);
                    if (cudaEventRecord(pjoin_, side2_) != cudaSuccess) { err = "verify: the gate projection's branch could not join"; return false; }
                }
                if (br) {   // STRATA_DF_BRANCH: a/b and z beside q/k/v and the conv; all of them read only this layer's input
                    fork(0);
                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                 (int) HV, n, df_side_[0]);
                    fork(1);
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                df_side_[1]);
                }
                const bool fuse_ab = fuse_convab_on() && !br && !batch_rec_ && !prof_on_ && !sidem;   // the side mixer computes the gates on its branch
                if (!side_proj) mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);   // with side_proj: above, first
                stamp(l, 2, grp);
                if (batch_rec_) {   // contiguous rows may be proposals for the same slot
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        const float* cx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats +
                                          (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gdn_conv_l2_multi(cx, qkv + (size_t) first * C, (const float*) wc->data,
                                          hb + (size_t) first * C, (int) C, (int) (2 * HK), EPS, t - first, cs, 0,
                                          keep_rows);
                    }
                } else
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                if (fuse_ab)   // 143-amdk: the conv and the a/b projections in one launch
                    gdn_conv_ab_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, tb, self_commit,
                                      xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                      (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
                                      (int) HV, cs);
                else
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  self_commit);
                if (m_tpdump_ != nullptr && grp == 0 && l == 0) {   // DIAG: the solo GDN qkv + conv output
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 548), qkv, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 552), qkv + 2048, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 556), qkv + 4096, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 560), hb, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                }
                stamp(l, 3, grp);
                if (br) {   // gate, beta and z, before the recurrence reads them
                    join(0);
                    join(1);
                } else {
                if (!fuse_ab && !sidem)   // the gates: the HIP fused conv kernel (143-amdk) or the side mixer's branch, else here
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                if (!side_proj) mm(wg, z_ + (size_t) tb * ZV, (int) N, (int) ZV);
                stamp(l, 5, grp);
                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
                    err = "verify: the gate projection's branch could not join";
                    return false;
                }
                if (sidem && cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
                    err = "verify: the GDN gates' branch could not join";
                    return false;
                }
                }
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                static const bool gdn_groups_on = [] {   // STRATA_BATCH_GDN_GROUPS=1, all slots in one launch
                    const char* v = std::getenv("STRATA_BATCH_GDN_GROUPS");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                if (batch_rec_ && gdn_groups_on && te - tb <= 8) {
                    strata::kernels::GdnGroups gg;
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        gg.state[gg.n] = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gg.first[gg.n] = first;
                        gg.ntok[gg.n] = t - first;
                        ++gg.n;
                    }
                    strata::kernels::gdn_step_norm_groups(gg, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS,
                                                          y_, (int) HK, (int) HV, false, cs);
                } else
                if (batch_rec_) {   // each slot's recurrence over its own proposed-token group
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        float* stx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gdn_step_norm_multi(stx, hb + (size_t) first * C, (int) C, gate + (size_t) first * HV,
                                            beta + (size_t) first * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_ + (size_t) first * ZV,
                                            (int) HK, (int) HV, t - first, keep_rows ? one_ : nullptr, cs, 0);
                    }
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, g_qfuse() ? (void*) xq_ : nullptr);
                stamp(l, 6, grp);
                // STRATA_QFUSE: done above by gdn_step_norm_multi - but a batch's per-slot recurrence passes it no q8_1
                // destination (#1139: the out-projection read stale bytes), so that case quantizes here as without it
                if (!g_qfuse() || batch_rec_) quant_in(y_ + (size_t) tb * ZV, (int) ZV);
                else il_ready = false;
                mm(wout, bo_ + tb * N, (int) ZV, (int) N);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope_on = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos,
                                        cudaStream_t sx) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, sx);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, sx);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, sx);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, sx);
                };
                // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope in one launch, bit-identical to the pair
                // (rope_parity check 6); STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes
                const bool fuse_nr = native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, cs);
                        return;
                    }
                    norm_rope_on(data, norm, rows, cols, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                // STRATA_DF_QB_Q4 (default on since 0.1.45; STRATA_DF_QB_Q4=0 gives the old token-by-token path): a Q4_0 KV cache
                // takes the batched window too - its one per-token extra, the query's rotation, is row-wise and the batched
                // path rotates (270650e).  The generated ids are identical (p620 gfx1201, A4000 sm_86, P100 sm_60, short and
                // long prompts), and the 6-token verify graph is 1950 nodes instead of 2610 (#2403: an RX 6900 XT faulted in
                // the first verify graph launch with --kv q4_0 + KV streaming + MTP; whether the graph size is the cause is
                // not confirmed)
                static const bool qb_q4 = [] {
                    const char* e = std::getenv("STRATA_DF_QB_Q4");
                    return e == nullptr || std::atoi(e) != 0;
                }();
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && (!st.kv_q4 || qb_q4);
                if (!q8_attn) quant_in(xm, (int) N);
                // STRATA_DF_BRANCH: the query side (q and its norm/RoPE/rotation on side 0, the indexer query on side 1)
                // beside the K/V side; the indexer query joins before the block scores, the query before attention
                const bool qbr = br && qb;
                if (qbr) {
                    fork(0);
                    native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                (int) (NH * 2 * HD), n, df_side_[0]);
                    copy_rows_strided(qcur_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, (int64_t) n * NH, HD, 2 * HD,
                                      df_side_[0]);
                    norm_rope_on(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, df_side_[0]);
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, df_side_[0]);   // <Hq, Hk> = <q, k>
                    fork(1);
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID,
                                              IQ * ID, N, IQ * ID, n, df_side_[1]);
                    norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                }
                // STRATA_SIDE_MIXER=1 / =2 (fork Eddoursul 34a35db + b063c13): the indexer's key projection,
                // the tail's snapshot, its keys appended, its query projection, norm and RoPE, the block scores and the
                // top-k read only the mixer's input and the indexer's own state - =1 runs them on a branch beside the
                // attention's projections; =2 also the queries' chain on a second branch.  Each step below is
                // the in-line code with its stream as a parameter; with the switch off they run on cs in the same order
                // as before (STRATA_DF_BRANCH, the same idea for fewer of these steps, is off whenever this is on).
                auto norm_rope_s = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos, cudaStream_t ns) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, ns);
                        return;
                    }
                    norm_rope_on(data, norm, rows, cols, pos, ns);
                };
                auto idx_key = [&](cudaStream_t ns) {
                    if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, ns);
                    else for (int t = tb; t < te; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, ns);
                };
                auto idx_snap = [&](cudaStream_t ns) {
                    if (batch_rec_) {   // snapshot each slot's indexer tail before its first proposed row
                        for (int t = tb; t < te && !keep_rows; ++t)
                            if (t == tb || brow_[t] != brow_[t - 1])
                                copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS,
                                                 slot_ss(t).qsa_states[qi].idx_tail, TS, ns);
                    } else
                    if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, ns);
                };
                auto idx_append = [&](cudaStream_t ns) {
                    if (dec_batch && !g_no_batch_kv && !batch_rec_) {
                        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, kStepCount,
                                                        n, 0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                        rope_scaling(), ns);
                    } else {
                        for (int t = tb; t < te; ++t) {
                            const QsaState& sx = slot_ss(t).qsa_states[qi];
                            const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                            native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                      (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                                      rope_scaling(), ns);
                        }
                    }
                };
                auto idx_query = [&](cudaStream_t ns) {
                    if (qb) {
                        bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                                  N, IQ * ID, n, ns);
                        norm_rope_s(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, ns);
                    } else
                        for (int t = tb; t < te; ++t) {
                            float* qx = qidx_ + t * IQ * ID;
                            bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), ns);
                            norm_rope_s(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH, ns);
                        }
                };
                auto idx_select = [&](cudaStream_t ns) {
                    qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                     s, scores_ + (size_t) tb * max_blocks_, ns);
                    qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                                   sel_ + (size_t) tb * cap_, ns);
                };
                // the attention's queries: the projection (mm on cs, which may read F4's interleaved copy; on a branch
                // F4's kernel too, its copy written on cs before the fork - bitwise the same), the q/gate
                // split with norm and RoPE, the rotation
                const bool q_il = side_proj && g_mmvq_il() &&
                                  strata::kernels::native_mmvq_il_supported(wq->native_type, n, (int) (NH * 2 * HD));
                auto q_chain = [&](cudaStream_t ns) -> bool {
                    if (ns == cs) mm(wq, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD));
                    else if (q_il) strata::kernels::native_mmvq_il(wq->native_type, wq->native_data, xq_, xil_,
                                                                   qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD), n, ns);
                    else native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N,
                                     (int) (NH * 2 * HD), n, ns);
                    if (qb) {
                        if (fuse_nr) {   // the q/gate split reads q straight out of the q|gate rows (stride 2 * HD)
                            native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
                                                     qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
                                                     rope_scaling(), pos_ + tb * NH, ns);
                        } else {
                            copy_rows_strided(qcur_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, (int64_t) n * NH, HD, 2 * HD, ns);
                            norm_rope_s(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH, ns);
                        }
                        if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, ns);   // <Hq, Hk> = <q, k>
                    } else {
                        for (int t = tb; t < te; ++t) {
                            float* qc = qcur_ + t * NH * HD;
                            if (fuse_nr) {
                                native_qsa_rms_norm_rope(qfull_ + t * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data, qc,
                                                         (int) NH, (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos_ + t * NH, ns);
                            } else {
                                if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                                      (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, ns) != cudaSuccess) {
                                    err = "verify: the q/gate split failed";
                                    return false;
                                }
                                norm_rope_s(qc, wqn, (int) NH, (int) HD, pos_ + t * NH, ns);
                            }
                            if (st.kv_rot) fwht256_inplace_cuda(qc, NH, ns);   // <Hq, Hk> = <q, k>
                        }
                    }
                    return true;
                };
                if (sidem) {
                    if (!branch(mfork_, cs, side_, "fork")) return false;
                    idx_key(side_);
                    idx_snap(side_);
                    idx_append(side_);
                    idx_query(side_);
                    idx_select(side_);
                    note_prio_tail(side_, 1);
                    if (cudaEventRecord(mjoin_, side_) != cudaSuccess) { err = "verify: the indexer's branch could not join"; return false; }
                }
                if (side_proj) {   // the queries beside the keys and values (fork b063c13)
                    if (q_il && !il_ready && n >= 2) {   // the copy before the fork (mm(wk) / mm(wv) on cs reuse it)
                        strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, n, cs);
                        il_ready = true;
                    }
                    if (!branch(pfork_, cs, side2_, "fork") || !q_chain(side2_)) return false;
                    if (cudaEventRecord(pjoin_, side2_) != cudaSuccess) { err = "verify: the queries' branch could not join"; return false; }
                }
                if (!sidem) idx_key(cs);
                stamp(l, 7, grp);
                // STRATA_MMVQ_V2=1: k, v and q in one launch, the same bits (qfull_ is not read before q's own call was;
                // 1.05-1.08x the three calls at T = 2-5).  Not with the query branch (qbr: q is made on side 0)
                bool q_done = false;
                if (!qbr && native_mmvq_v2_enabled() && wk->native_type == 8 && wv->native_type == 8 &&
                    wq->native_type == 8) {
                    const void* gw[3] = {wk->native_data, wv->native_data, wq->native_data};
                    float* gy[3] = {kcur_ + tb * NKV * HD, vcur_ + tb * NKV * HD, qfull_ + tb * NH * 2 * HD};
                    const int go[3] = {(int) (NKV * HD), (int) (NKV * HD), (int) (NH * 2 * HD)};
                    q_done = native_q8_0_mmvq_group(3, gw, gy, go, xq_, (int) N, n, cs);
                }
                if (!q_done) {
                    mm(wk, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                    mm(wv, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD));
                }
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0) {   // DIAG: the k/v weight heads and the raw (pre-norm) k/v
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 528), wk->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 532), wv->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 536), wq->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 540), kcur_ + tb * NKV * HD, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 544), vcur_ + tb * NKV * HD, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                }
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0) {   // DIAG: the l3 intermediates, vs the TP path
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 492), kcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 496), vcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                }
                if (!sidem) idx_snap(cs);
                // #783 PR-d (stuchapin909): the window's K/V cells in one launch per pool instead of one per token (a
                // batch's rows each have their own K/V, so that case keeps the loop)
                if (dec_batch && !g_no_batch_kv && !batch_rec_) {
                    const int32_t* step_b = step_ + tb * kStepCount;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        // #1188: a streamed state's host copy gets the window's cells too, as the per-token path below
                        // writes them - without it a block that was not resident when its cells were appended came
                        // back from the host copy without them, and free generation degenerated into repetition
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();
                        kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b, kStepCount,
                                           kc_b, kc_b, (int) (NKV * HD), n, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, kStepCount, n, vc_b, vc_b, s, cs,
                                           mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, kStepCount, n, kc_b, vc_b, s, cs,
                                           &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b, kStepCount,
                                           kc_b, vc_b, (int) (NKV * HD), n, s, cs, &st.host);
                    else
                        for (int t = tb; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& st = slot_ss(t).qsa_states[qi];   // the row's own K/V (ss's outside a batch)
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                if (!sidem) idx_append(cs);
                stamp(l, 9, grp);
                if (qbr) {
                    join(1);   // the indexer query, for the block scores
                } else {
                    if (!side_proj && !q_chain(cs)) return false;
                    if (!sidem) idx_query(cs);
                }
                stamp(l, 10, grp);
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 504), qidx_ + tb * IQ * ID, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                bool streamed = false;   // a streamed KV cache keeps the per-row resolve-then-attend order
                if (batch_rec_ && g_qsa_slots)
                    for (int t = tb; t < te; ++t) streamed = streamed || slot_ss(t).qsa_states[qi].kv_mode == 1;
                // STRATA_BATCH_ATTN_ROWS=1: the window's rows in ONE launch per step - scores with each row's
                // own indexer keys, the top-k over all rows (row-independent), attention with each row's own pools -
                // instead of 3-4 one-row launches per row; the same per-row arithmetic.  Not with KV streaming
                // (qsa_kv_resolve has work per slot then).
                static const bool attn_rows = [] {
                    const char* v = std::getenv("STRATA_BATCH_ATTN_ROWS");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                bool rows_ok = batch_rec_ && attn_rows && te - tb <= strata::kernels::kAttnRowsMax;
                for (int t = tb; rows_ok && t < te; ++t) rows_ok = slot_ss(t).qsa_states[qi].kv_mode != 1;
                if (rows_ok) {
                    const int n_rows = te - tb;
                    strata::kernels::QsaScoreRows sr;
                    strata::kernels::QsaAttnPoolsRows pr;
                    sr.n = pr.n = n_rows;
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        sr.pooled[t - tb] = sx.idx_pooled;
                        sr.dead[t - tb] = sx.idx_dead;
                        pr.p[t - tb] = qsa_attn_pools(sx);
                    }
                    qsa_block_scores_rows(sr, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n_rows, max_blocks_, s,
                                          scores_ + (size_t) tb * max_blocks_, cs);
                    qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n_rows, max_blocks_, cap_,
                                   s, sel_ + (size_t) tb * cap_, cs);
                    qsa_decode_attn_batch_rows(qcur_ + tb * NH * HD, pr, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount,
                                               cap_, s, attn_scratch_ + (size_t) tb * attn_scratch_floats_,
                                               attn_ + tb * NH * HD, n_rows, cs);
                } else
                if (batch_rec_ && g_qsa_slots && !streamed) {
                    // the rows' block scores, top-k and attention, up to 8 rows per launch.  The rows only read
                    // their slots' keys and K/V here and write their own scores/sel/scratch/attn rows, so running
                    // each step for all rows before the next is the per-row loop's result
                    for (int t0 = tb; t0 < te; t0 += kQsaAttnMaxSlots) {
                        const int m = std::min(te - t0, kQsaAttnMaxSlots);
                        const float* pooled[kQsaAttnMaxSlots];
                        const float* dead[kQsaAttnMaxSlots];
                        QsaAttnSlotPools px;
                        for (int i = 0; i < m; ++i) {
                            const QsaState& sx = slot_ss(t0 + i).qsa_states[qi];
                            pooled[i] = sx.idx_pooled;
                            dead[i] = sx.idx_dead;
                            px.p[i] = qsa_attn_pools(sx);
                        }
                        // a launcher that would not run the per-row call's kernel launches nothing: per-row calls
                        if (!qsa_block_scores_slots(pooled, dead, qidx_ + t0 * IQ * ID, step_ + t0 * kStepCount, m,
                                                    max_blocks_, s, scores_ + (size_t) t0 * max_blocks_, cs))
                            for (int t = t0; t < t0 + m; ++t)
                                qsa_block_scores(pooled[t - t0], dead[t - t0], qidx_ + t * IQ * ID, step_ + t * kStepCount,
                                                 1, max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t0 * max_blocks_, step_ + t0 * kStepCount, m, max_blocks_,
                                       cap_, s, sel_ + (size_t) t0 * cap_, cs);
                        if (!qsa_decode_attn_slots(qcur_ + t0 * NH * HD, px, sel_ + (size_t) t0 * cap_,
                                                   step_ + t0 * kStepCount, cap_, s,
                                                   attn_scratch_ + (size_t) t0 * attn_scratch_floats_,
                                                   attn_ + t0 * NH * HD, m, cs))
                            for (int t = t0; t < t0 + m; ++t)
                                qsa_decode_attn_batch(qcur_ + t * NH * HD, px.p[t - t0], sel_ + (size_t) t * cap_,
                                                      step_ + t * kStepCount, cap_, s,
                                                      attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                                      attn_ + t * NH * HD, 1, cs);
                    }
                } else if (batch_rec_) {   // each row selects and attends over its own slot's K/V
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + t * IQ * ID, step_ + t * kStepCount, 1,
                                         max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t * max_blocks_, step_ + t * kStepCount, 1, max_blocks_, cap_, s,
                                       sel_ + (size_t) t * cap_, cs);
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_ + t * kStepCount, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + t * NH * HD, px, sel_ + (size_t) t * cap_, step_ + t * kStepCount,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
                if (!sidem) idx_select(cs);
                else if (cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
                    err = "verify: the indexer's branch could not join";
                    return false;
                }
                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
                    err = "verify: the queries' branch could not join";
                    return false;
                }
                stamp(l, 11, grp);
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0) {
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 508), scores_ + (size_t) tb * max_blocks_, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 512), sel_ + (size_t) tb * cap_, 4 * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, cs);
                }
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                if (qbr) join(0);   // the query, for attention
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 500), qcur_ + tb * NH * HD, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                const QsaAttnPools pools = qsa_attn_pools(st);
                // STRATA_QSA_SPLIT_ATTN=1: the split tensor-core attention where it applies (int8 KV, sm_80+)
                static const bool split_attn = [] { const char* v = std::getenv("STRATA_QSA_SPLIT_ATTN"); return v != nullptr && std::atoi(v) != 0; }();
                if (!(split_attn && qsa_decode_attn_split(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_,
                                                          step_ + tb * kStepCount, cap_, s,
                                                          attn_scratch_ + (size_t) tb * attn_scratch_floats_,
                                                          (uint64_t) n * (uint64_t) attn_scratch_floats_, attn_ + tb * NH * HD, n, cs)))
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                }
                stamp(l, 13, grp);
                if (m_tpdump_ != nullptr && grp == 0 && l == 3 && tb == 0)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 516), attn_ + tb * NH * HD, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                quant_in(attn32_ + tb * NH * HD, (int) (NH * HD));
                mm(wo, bo_ + tb * N, (int) (NH * HD), (int) N);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        const bool q8_ffn = gr_read_group(1, true, inj_, inj2_);   // true: nat_xq_ holds the MoE input's q8_1 (STRATA_QFUSE)
        stamp(l, 32, grp);   // the profile only: the FFN-side read ends here, the router starts
        if (m_tpdump_ != nullptr && grp == 0 && l < 2)   // DIAG: the single-card MoE input (half 1), same slot as the TP path
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 100) + (size_t) l * 4, mixed_ + tb * N, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (m_tpdump_ != nullptr && grp == 0 && l == 3)   // DIAG: the l3 MoE input, vs the TP path
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 524), mixed_ + tb * N, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        // DIAG (STRATA_TP_VEC=N): site 0, this layer's mixer output in full, so the two paths' MoE inputs can be
        // compared as vectors rather than by their first four floats
        if (const int64_t vat = tp_vec_at(l, 0); vat >= 0 && grp == 0)
            cudaMemcpyAsync(m_tpdump_ + vat, mixed_ + tb * N, (size_t) tp_vec_len() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        // with CPU experts in the window the shared expert forks off once the doorbell is published (it no longer
        // competes with it for the GPU's first microseconds); STRATA_SH_FORK_LATE=0 forks at the top as before
        static const bool sh_fork_late_env = [] {
            const char* e = std::getenv("STRATA_SH_FORK_LATE");
            return !e || e[0] != '0';
        }();
        const bool sh_fork = sh_stream_on() && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
        const bool sh_fork_late = sh_fork && sh_fork_late_env && !ar_on();
        cudaStream_t sh_stream = sh_fork ? sh_cs_ : cs;
        if (sh_fork && !sh_fork_late) {
            note_prio_fork(cs);
            cudaEventRecord(ev_fork_, cs);
            cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        // S26 STRATA_LFUSE=1: the shared expert's gate row rides in the router GEMV launch, its sigmoid + row scale
        // move into the combine, gate + up share one launch (all bitwise; resident combine path, 2-8 rows only)
        const WeightRef* w_sgi = lfuse_on(n) && g_lfuse_gate() ? v.get("ffn_gate_inp_shexp.weight") : nullptr;
        bool sg_ready = false;
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                if (w_sgi != nullptr)
                    sg_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                             logits_ + tb * NE, NE, N, NE, n,
                                                             (const uint16_t*) w_sgi->data, sh_g_ + tb, 1, cs);
                if (!sg_ready)
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
#if defined(STRATA_HIP_GFX906)
        } else if (dec_batch) {
            // gfx906: the Coder's 256-expert router (the native router is 512 x 10 only) for the whole window in one
            // multi-column BF16 projection and one top-k, bitwise the per-token calls (route_window_parity); ~3 ms
            // of a ~55 ms verify window on 2x MI50.  STRATA_ROUTE_PER_TOKEN=1: the per-token calls.
            if (!moe_route_window(wt, g, l, K, ss.moe, mixed_ + tb * N, logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n,
                                  cs, err))
                return false;
#endif
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        // STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): the top-10 picked again with a logit bonus for the
        // resident experts; the weights stay the router's (route_prior.cu)
        static const float route_prior = [] { const char* v = std::getenv("STRATA_ROUTE_PRIOR"); return v ? (float) std::atof(v) : 0.0f; }();
        if (route_prior > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr)
            route_prior_top10(logits_ + tb * NE, hits_.d_res + l * g.n_expert, route_prior, ids_ + tb * K, w_ + tb * K, n, cs);
        if (route_resident_cfg().margin > 0.0f && NE == 512 && K == 10 && hits_.d_res != nullptr) {
            // STRATA_ROUTE_RESIDENT: after the router, before the plan/doorbell read ids_/w_ (opt-in, changes the output)
            try {
                native_route_resident(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, hits_.d_res + l * g.n_expert, n,
                                      route_resident_cfg().margin, route_resident_cfg().lo, route_resident_cfg().hi,
                                      rr_stats_ != nullptr ? rr_stats_ + (n <= 8 ? 4 : 0) : nullptr, cs);
            } catch (const std::exception& e) { err = "verify route-resident: " + std::string(e.what()); return false; }
        }
        if (ar_on()) {
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
        } else {
            if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                              (uint32_t) ((l - lb_) * G + grp + 1), cs);
#if defined(STRATA_USE_HIP)
            if (g_doorbell_store)   // #649 A/B: the step's ring stored, not incremented over PCIe
                doorbell_publish_value(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                       m_ids_ + tb * K, m_w_ + tb * K, m_seq_, (uint32_t) ((l - lb_) * G + grp + 1), cs);
            else
#endif
            // #578: the helper GPUs reduce with the routing weights - publish them; set_always_publish: the rows
            // whatever the device's residency table says (pipelined windows)
            if (remote_opt_ || always_publish_)
                doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                 m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
            else {
                const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
                doorbell_publish_res(xm, ids_ + tb * K, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                     m_x_ + tb * N, m_ids_ + tb * K, m_seq_, cs);
            }
            if (sh_fork_late) {
                note_prio_fork(cs);
                cudaEventRecord(ev_fork_, cs);
                cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
            }
        }
        stamp(l, 17, grp);
        bool qdedup = false;   // STRATA_VERIFY_QDEDUP took effect: the experts' q8_1 image of xm is already made
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = sh_fork ? sh_xq_ : xq_;
            // STRATA_VERIFY_QDEDUP=1 (not with the forked shared-expert stream, which quantizes into its own buffer): the
            // experts' q8_1 image of xm is made first and the shared expert's gate/up read it (quantize_q8_1_rows and
            // native_quantize_q8_1 write the same bytes)
            qdedup = g_qdedup() && !sh_fork && strata::kernels::cpu::expert_layout().native;
            if (qdedup && !q8_ffn) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            if (!shared_expert_native_bf16_enabled()) {
                if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, sh_stream);   // contiguous rows
                else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, sh_stream);
            }
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, sh_stream,
                                    qdedup ? (const void*) (nat_xq_ + (size_t) tb * (N / 32) * 36) : nullptr,
                                    (sg_ready ? 1 : 0) | (lfuse_on(n) && g_lfuse_pair() ? 2 : 0));
                sg_gated_[grp] = sg_ready;
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
            if (sh_fork) {
                note_prio_tail(sh_cs_, -1);
                cudaEventRecord(ev_join_, sh_cs_);
            }
        }
        if (strata::kernels::cpu::expert_layout().native) {
            if (!qdedup)
                if (!q8_ffn) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            // STRATA_POOL_ORDER=1: also produce the pool's OWN activation contract.  `hit_xq_` already exists for
            // this (the non-native path's buffer) and `quantize_q8_0` is byte-identical to the pool's `from_float`
            // (nvfp4_q8_0_entry_parity asserts it), so a resident expert and a missed one then compute the same
            // number and the resident SET can change without changing the tokens.
            if (g_pool_order() && !q8_ffn)
                quantize_q8_0(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, (int64_t) n * N, cs);
        } else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        if (m_tpdump_ != nullptr && grp == 0 && l < 2) {   // DIAG: see the TP window's dump
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 72) + (size_t) l * 8, ids_ + tb * K,
                            4 * sizeof(int32_t), cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 72) + (size_t) l * 8 + 4, w_ + tb * K,
                            4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
        }
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    static const bool fuse_head_gr_env = [] {
        const char* e = std::getenv("STRATA_FUSE_HEAD_GR");
        return e && e[0] == '1';
    }();
    const bool fuse_head_gr = fuse_head_gr_env && (le_ == g.n_layers) && (head_ != nullptr && head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        float* parts_out = parts_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                if (g_pool_order() && f.gu_type == 40 && f.d_type == 40) {
                    // THE POOL-ORDER PATH: the same number a CPU miss produces, so residency stops being a
                    // correctness knob.  Its activation is `block_q8_0`, quantized here from the same `xm`.
                    native_expert_grouped_q8_0(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                               hit_xq_ + (size_t) tb * (N / 32) * 34, (int) ((N / 32) * 34),
                                               hit_scratch_, dst_buf, cs, 0, gy);
                } else
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, dst_buf, cs);
            }
        };
        bool hits_in_combine = false;
        if (ar_on()) {
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, parts_out);
            stamp(l, 20, grp);
        } else {
            if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
                wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
                copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
            } else {
                if (fuse_wait_on() && !prof_on_ && m_bal_ == nullptr) {   // 143-amdk: one launch for the wait and the copy
                    wait_copy_i32_from_mapped(m_flagA_, ring, pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
                } else {
                if (m_bal_ != nullptr)   // set_pcie_balance: the GPU's share starts as the plan arrives
                    wait_flag_ge_stamped(m_flagA_, ring, nullptr, m_bal_ + ((l - lb_) * 2 + grp) * 4, cs);
                else
                    wait_flag_ge(m_flagA_, ring, cs);              // the pool published this group's GPU plan
                copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
                }
            }
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, hit_out);
            stamp(l, 20, grp);
            if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
            else if (m_bal_ != nullptr)   // set_pcie_balance: the PCIe part starts
                wait_flag_ge_stamped(m_flagB_, ring, m_bal_ + ((l - lb_) * 2 + grp) * 4 + 1, nullptr, cs);
            else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                            // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                const unsigned long long* p_dst2 = (const unsigned long long*) (pl + dst2_off_);   // STRATA_ADAPT_FETCH
                const bool pcie_t = pcie_timed_ && pcie_n_ < kPcieTimingMax;   // STRATA_PCIE_TIMING: this H2D, timed
                if (pcie_t) gpu_stamp(pcie_st_, 2 * pcie_n_, cs);
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs, p_dst2);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs, p_dst2);
                if (pcie_t) {
                    gpu_stamp(pcie_st_, 2 * pcie_n_ + 1, cs);
                    ++pcie_n_;
                }
            }
            stamp(l, 21, grp);
            // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (always none at
            // pcie_frac 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
            grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
            stamp(l, 22, grp);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                         skip_ + grp, ring, cs);
            } else {
                const bool fuse_rows = fuse_rows_on() && !prof_on_ && !remote_opt_ && dec_batch && m_bal_ == nullptr;   // 143-amdk
                if (!fuse_rows) {
                    if (m_bal_ != nullptr)   // set_pcie_balance: ... and ends as it waits for the CPU's rows
                        wait_flag_ge_stamped(m_flag_, ring, m_bal_ + ((l - lb_) * 2 + grp) * 4 + 2, nullptr, cs);
                    else
                        wait_flag_ge(m_flag_, ring, cs);           // the CPU's share is in the mapped rows
                }
                stamp(l, 23, grp);
                if (remote_opt_)   // #578: the helper's rows come back reduced; skip them as well
                    remote_opt_->copy_rows(parts_out, m_ymiss_ + (size_t) tb * K * N, tb, n, p_dst, p_counts + 1, cs);
                else if (fuse_rows)   // the wait folded into the copy's blocks
                    copy_rows_from_mapped_after(m_flag_, ring, parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                                p_dst, p_counts + 1, cs);
                else if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            }
            // 143-amdk: the add of the GPU's hit rows is done by the combine that follows (one launch fewer)
            hits_in_combine = fuse_hits_on() && !prof_on_ && !remote_opt_ && !sg_gated_[grp] && dec_batch && n > 1 &&
                              native_moe_combine_enabled() && n <= 8 &&
                              native_moe_combine_hits_supported(parts_ + (size_t) tb * K * N, shared_ + tb * N, bo_ + tb * N, N, K, n);
            if (!hits_in_combine) moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        // Same condition as `sh_fork` above, which is per group and out of scope here: wait only when the fork
        // actually ran (a wait on an event never recorded is a no-op, but saying it outright reads better).
        if (sh_stream_on() && !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr) {
            cudaStreamWaitEvent(cs, ev_join_, 0);
        }
        if (sg_gated_[grp]) {   // STRATA_LFUSE: the shared expert's gate was not applied: the combine applies it (all rows are hits)
            try {
                native_moe_combine_multi_hits_gated(parts_out, w_ + tb * K, shared_ + tb * N, sh_g_ + tb, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        if (hits_in_combine) {
            try {
                native_moe_combine_multi_hits_added(parts_ + (size_t) tb * K * N, hit_out, p_dst, p_counts + 1, w_ + tb * K,
                                                    shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
                if (m_tpdump_ != nullptr && grp == 0 && l < 2)   // DIAG: this window's combine
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 40) + (size_t) l * 4, bo_ + tb * N,
                                    4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        if (remote_opt_) remote_opt_->combine(bo_ + tb * N, y_dummy_ + tb * N, tb, n,
                              device_plan_ ? skip_ + grp : nullptr, ring, cs);
        stamp(l, 24, grp);
        // STRATA_TP_DUMP=1: the residual's head, per layer (see h_tpdump_)
        if (m_tpdump_ != nullptr && grp == 0)
            cudaMemcpyAsync(m_tpdump_ + (size_t) l * 8, bo_ + tb * N, 8 * sizeof(float), cudaMemcpyDeviceToHost, cs);
        // DIAG (STRATA_TP_VEC=N): site 1, this layer's combined residual in full (the sum of both halves plus
        // the shared expert - the value every later layer reads)
        if (const int64_t vat = tp_vec_at(l, 1); vat >= 0 && grp == 0)
            cudaMemcpyAsync(m_tpdump_ + vat, bo_ + tb * N, (size_t) tp_vec_len() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                if (dec_batch && !g_no_multi_gr) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
                else for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = hand_out_ + (size_t) hrow0 * HB;
        if (hflag_out_d_ != nullptr && !batch_rec_) {   // the overlapped split: payload + the next stage's flag
            handoff_publish(hout, R_, (int64_t) T * HC * N, bo_, (int64_t) T * N, inj2_, (int64_t) T * HC, hcount_d_,
                            hflag_out_d_, cs, g_test_drop > 0 ? hdrop_d_ : nullptr);   // the test word only with the hook on
            return true;
        }
        copy_from_mapped(hout, R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        // S25 (STRATA_HC_Q8=1): the final mixer from the GGUF's Q8_0 projections too, through the same multi read
        // (no pending write, no inject)
        const bool mix_q8 = hd->hc_q8 != nullptr && hu->hc_q8 != nullptr && head_ != nullptr && head_->loaded();
        if (mix_q8) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = false;
                a.w_norm = (const float*) hn->data; a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data; a.w_inject = nullptr;
                a.q8_down = (const uint8_t*) hd->hc_q8; a.q8_up = (const uint8_t*) hu->hc_q8;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        else if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = lo_ + t * g.hc_lr;
                a.rs = rs_ + t * HC;
                a.inject_out = head_inj_;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_enabled() && head_ != nullptr && head_->loaded()) {
            // the final mixer, the window's tokens in one read (fused_gr_read_multi without the pending write: the
            // last layer's write was done above; its sums are gr_read's)
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = Rt(t); fa[t].R_out = Rt(t); fa[t].apply = false;
                fa[t].w_norm = (const float*) hn->data; fa[t].w_down = (const uint16_t*) hd->data;
                fa[t].w_up = (const uint16_t*) hu->data; fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (head_ != nullptr && head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                // DIAG (STRATA_TP_VEC=N): site 2, the head's input (the final mixer's output) in full: a head whose
                // logits come out too small is either fed a too-small vector or reading the wrong weights
                if (const int64_t vat = tp_vec_at(g.n_layers, 2); vat >= 0)
                    cudaMemcpyAsync(m_tpdump_ + vat, head_mixed_, (size_t) tp_vec_len() * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(head_->type(), (int) T, (int) n_vocab_)) {
                    if (T >= 2) strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, (int) T, cs);
                    strata::kernels::native_mmvq_il(head_->type(), head_->weights(), xq_, xil_, head_logits_, (int) N,
                                                    (int) n_vocab_, (int) T, cs);
                } else
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        if (argmax_rows_wanted()) {   // sm_80 to sm_89: the pick over many blocks a row (bitwise the one-block kernel's)
            argmax_rows(head_logits_, T, (int) n_vocab_, arg_scratch_, m_out_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        }
        // #879: whether each row's logits are all finite, beside the pick (read after the window's own sync)
        if (nan_guard_on()) strata::kernels::logits_nonfinite_rows(head_logits_, T, (int) n_vocab_, m_nf_, cs);
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

// The TP peer's own stage line.  Rank 0's report shows what rank 0 waited for; the wait's OTHER half (what rank 1 was
// doing while rank 0 spun) only exists on rank 1's device.  The stamps are written by rank 1's captured graph, so the
// device must be drained before the read-back (the graph's last kernels may still be in flight) - profiling only.
std::string Verifier::peer_profile_report() {
    if (!prof_on_ || tp_peer_ == nullptr || !tp_peer_->prof_on_) return std::string();
    {
        const OnDevice on_peer(tp_peer_->device_);
        (void) cudaDeviceSynchronize();
    }
    tp_peer_->collect_profile();
    return tp_peer_->profile_report();
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "router+plan+bell", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "  hc1 norm", "  hc1 down", "  hc1 up"};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

// #871: is the zero-doorbell graph right for the next window?  Only while every expert of [lb_, le_) is in VRAM now
// (the host table is the one the device copy was uploaded from; a loan, a VRAM shrink or a swap in flight has
// marked some -1).  Not: the doorbell graph, which asks the pool for what the device does not hold.
void Verifier::refresh_ar() {
    if (!all_resident_ || h_res_ == nullptr || g_ == nullptr) return;
    const int64_t ne = g_->n_expert;
    const int32_t* r = h_res_ + lb_ * ne;
    const int64_t n = (le_ < 0 ? g_->n_layers : le_) * ne - lb_ * ne;
    int32_t lo = 0;
    for (int64_t i = 0; i < n; ++i) lo = std::min(lo, r[i]);
    const bool off = lo < 0;
    if (off != ar_off_) {
        ar_off_ = off;
        std::fprintf(stderr, "strata verify: layers [%lld, %lld) %s\n", (long long) lb_, (long long) (le_ < 0 ? g_->n_layers : le_),
                     off ? "have experts out of VRAM (a prompt loan, a shrink or a swap): the doorbell graph runs the windows"
                         : "are 100% VRAM resident again: the zero-doorbell graph");
    }
}

bool Verifier::capture(int T, std::string& err) {
    cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] : exec_[T];
    if (exec_t != nullptr) return true;
    {   // said before the capture: a process that exits inside it (#1275: Windows, 313 MiB free) leaves this line as the trace
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            std::fprintf(stderr, "strata verify: capturing the %d-token window (%zu MiB of VRAM free)\n", T, free_b >> 20);
        else
            (void) cudaGetLastError();
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    prio_forks_.clear();
    prio_tails_.clear();
#endif
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
#if defined(STRATA_USE_HIP)   // 143-amdk: the same listing on HIP (kernel names by hipKernelNameRefByPtr); STRATA_VERIFY_NODES=2 also lists the first nodes in graph order
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {
        size_t nn = 0;
        hipGraphGetNodes(graph, nullptr, &nn);
        std::vector<hipGraphNode_t> nodes(nn);
        hipGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        std::string seq;
        size_t shown = 0;
        for (hipGraphNode_t nd : nodes) {
            hipGraphNodeType ty;
            hipGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == hipGraphNodeTypeKernel) {
                hipKernelNodeParams kp{};
                if (hipGraphKernelNodeGetParams(nd, &kp) == hipSuccess && kp.func != nullptr) {
#if defined(_WIN32)
                    using Fn_hipKernelNameRefByPtr = const char* (*)(const void*, hipStream_t);
                    static auto p_hipKernelNameRefByPtr = reinterpret_cast<Fn_hipKernelNameRefByPtr>(
                        GetProcAddress(GetModuleHandleA("amdhip64_7.dll"), "hipKernelNameRefByPtr"));
                    const char* fn = p_hipKernelNameRefByPtr ? p_hipKernelNameRefByPtr(kp.func, nullptr) : nullptr;
#else
                    const char* fn = hipKernelNameRefByPtr(kp.func, nullptr);
#endif
                    if (fn != nullptr) name = fn;
                }
            } else if (ty == hipGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == hipGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
            if (std::atoi(std::getenv("STRATA_VERIFY_NODES")) >= 2 && shown++ < 400) seq += "\n  " + name.substr(0, 110);
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 80; ++i) std::fprintf(stderr, "\n %4d x %.110s", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n%s\n", seq.c_str());
    }
#endif
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)   // a CUDA debug listing (node types, kernel names)
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            cudaGraphNodeGetType(nd, &ty);
            std::string name = "type" + std::to_string((int) ty);
            if (ty == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams kp{};
                if (cudaGraphKernelNodeGetParams(nd, &kp) == cudaSuccess) {
#if CUDART_VERSION >= 12030   // cudaFuncGetName arrived in CUDA 12.3
                    const char* fn = nullptr;
                    if (cudaFuncGetName(&fn, kp.func) == cudaSuccess && fn) name = fn;
#endif
                }
            } else if (ty == cudaGraphNodeTypeMemcpy) name = "memcpy";
            else if (ty == cudaGraphNodeTypeMemset) name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
#if CUDART_VERSION >= 12030   // the programmatic (PDL) edges: CUDA allows them only from a kernel node to a kernel node
        size_t ne = 0;
#if CUDART_VERSION >= 13000
        cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &ne);
#else
        cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &ne);
#endif
        std::vector<cudaGraphNode_t> from(ne), to(ne);
        std::vector<cudaGraphEdgeData> ed(ne);
#if CUDART_VERSION >= 13000
        const cudaError_t ge = cudaGraphGetEdges(graph, from.data(), to.data(), ed.data(), &ne);
#else
        const cudaError_t ge = cudaGraphGetEdges_v2(graph, from.data(), to.data(), ed.data(), &ne);
#endif
        size_t prog = 0, bad = 0;
        for (size_t i = 0; ge == cudaSuccess && i < ne; ++i) {
            if (ed[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++prog;
            cudaGraphNodeType ta, tb2;
            if (cudaGraphNodeGetType(from[i], &ta) != cudaSuccess || cudaGraphNodeGetType(to[i], &tb2) != cudaSuccess ||
                ta != cudaGraphNodeTypeKernel || tb2 != cudaGraphNodeTypeKernel)
                ++bad;
        }
        if (ge != cudaSuccess) cudaGetLastError();
        std::fprintf(stderr, "strata verify: %zu edges, %zu programmatic (PDL), %zu of them not kernel to kernel%s\n", ne,
                     prog, bad, ge != cudaSuccess ? " (the edge query failed)" : "");
#endif
    }
#endif
    unsigned long long iflags = 0;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    if (graph_prio_on()) {   // STRATA_GRAPH_PRIO: the nodes run at their own (captured or repaired) priorities
        apply_node_priorities(graph, T);
        iflags = cudaGraphInstantiateFlagUseNodePriority;
    }
#endif
    bool made = instantiate_evicting(exec_t, graph, {}, "verify: instantiate: ", err, T, iflags);
    if (!made && iflags != 0) {   // a driver / card that rejects the flag must not take the engine down (opt-in only)
        std::fprintf(stderr, "strata verify: instantiate with node priorities failed (%s); retried at the default\n", err.c_str());
        (void) cudaGetLastError();
        exec_t = nullptr;
        err.clear();
        made = instantiate_evicting(exec_t, graph, {}, "verify: instantiate: ", err, T, 0);
    }
    cudaGraphDestroy(graph);
    if (!made) return false;
    const cudaError_t ue = cudaGraphUpload(exec_t, cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    // The upload only warms the graph; the first launch does it otherwise. A failed one (out of memory on a full card)
    // must not stay pending: the next kernel wrapper's cudaGetLastError would read it and end the engine (#3702).
    if (ue != cudaSuccess || us != cudaSuccess) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata verify: the %d-token window's upload was skipped (the first launch does it)\n", T);
    }
    return true;
}

// ===================================== tensor parallelism (--tp) =====================================
// The coordinator (rank 0) owns the pair's shared handshake flags: one mapped host word each, the same device
// pointer on both cards.  `flag_b` is raised by rank 1 after its out-projection partial is complete; rank 0
// spins on it before the allreduce.  `flag_a` is raised by rank 0 after the allreduce; rank 1 spins on it
// before it reads the summed residual.  Both start at 0 and carry the global layer index + 1, so a window of
// L layers raises 1..L and the next window overwrites the same sequence.
void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    // the overlapped split: the next stage's graph is launched after this, and must see THIS window's raise
    if (hflag_out_h_ != nullptr) *(volatile uint32_t*) hflag_out_h_ = 0;
    if (herr_h_ != nullptr) *(volatile uint32_t*) herr_h_ = 0;
    if (hdrop_h_ != nullptr) *(volatile uint32_t*) hdrop_h_ = (g_test_drop > 0 && windows + 1 == g_test_drop) ? 1u : 0u;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

void Verifier::collect_profile() {
    const OnDevice on_self(device_);   // the peer's buffers are on ITS device: this runs for rank 1's object too
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    accumulate_profile(prof_h_.data());
}

void Verifier::accumulate_profile(const unsigned long long* stamps) {
    const ModelGeometry& g = *g_;
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return stamps[(size_t) (l * kProfPer + i)]; };
    // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
    // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
    // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
    // contributes nothing.
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    // only this stage's layers [lb_, le_) are ever stamped (a layer split); the head only on the last stage
    for (int64_t l = lb_; l < le_; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        {   // the FFN-side read split like hc-read0: norm 30, down 31, up 32; what is left of 17 is the router, plan, doorbell
            const double n1 = gap(at(l, 30), at(l, 16)), d1 = gap(at(l, 31), at(l, 30)), u1 = gap(at(l, 32), at(l, 31));
            if (n1 > 0 && d1 > 0 && u1 > 0) {
                prof_sum_[kind][30] += n1;
                prof_sum_[kind][31] += d1;
                prof_sum_[kind][32] += u1;
                prof_sum_[kind][17] -= gap(at(l, 32), at(l, 16));
            }
        }
        if (l + 1 < le_) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    if (le_ == L) prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    ++prof_windows_;
}

// 0.1.43 (STRATA_ADAPT_OVERLAP): hold the host until the adaptive tier's copies up to `layer` have landed.  The wait
// is a spin on an atomic the copy stream's host functions raise (no driver call here); 20 s is the window's own limit.
bool Verifier::adapt_gate_wait(int32_t layer, std::string& err) {
    const Clock::time_point t0 = Clock::now();
    uint32_t spins = 0;
    while (adapt_gate_->reached() < layer) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        if (Clock::now() - t0 > std::chrono::seconds(20)) {
            err = "verify: the adaptive tier's copies for layer " + std::to_string(layer) + " never landed" +
                  released_note(release_gpu_waits(5000));
            return false;
        }
    }
    ms_gate += ms_since(t0);
    return true;
}

bool Verifier::handoff_failed(std::string& err) const {
    if (herr_h_ == nullptr || *(const volatile uint32_t*) herr_h_ == 0) return false;
    err = "verify: layer split: the hand-off into layer " + std::to_string(lb_) + " did not arrive within " +
          std::to_string(g_split_wait_ns / 1000000ull) + " ms (STRATA_SPLIT_WAIT_MS); the window was not used. "
          "STRATA_SPLIT_OVERLAP=0 runs the serial order";
    trace_ev("HANDOFF-TIMEOUT", -1, lb_, (int64_t) (g_split_wait_ns / 1000000ull));
    return true;
}

bool Verifier::chain_overlap() const {
    return next_ != nullptr && hflag_out_h_ != nullptr && hflag_out_d_ != nullptr && hcount_d_ != nullptr &&
           next_->hflag_in_h_ == hflag_out_h_ && next_->device_ != device_ && !next_->ple_stage();
}

bool Verifier::prelaunch(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    prelaunched_ = 0;
    cudaGraphExec_t exec_t = ar_off_ ? exec_nr_[T] : exec_[T];   // #871: the variant the previous stage refreshed
    if (T < 1 || T > max_t_ || exec_t == nullptr) { err = "verify: prelaunch before the window was captured"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (pos0 + T > ss_->qsa_states[ss_->qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    const Clock::time_point t0 = Clock::now();
    stage_inputs(T, tokens, pos0);
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW-PRE", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    if (h_bal_ != nullptr) std::memset(h_bal_, 0, (size_t) bal_p_.size() * 4 * 8);   // this window's stamps only
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    // 0.1.43: the same gate run() applies before it launches (the adaptive tier's copies land first)
    if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && ar_on() &&
        !adapt_gate_wait((int32_t) g_->n_layers, err)) return false;
    const cudaError_t le = cudaGraphLaunch(exec_t, cs_);
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    prelaunched_ = T;
    return true;
}

bool Verifier::sample_window(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    using namespace strata::kernels;
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        bool judged = false;
        if (sampled && spec_q_ != nullptr && spec_nq_ > 0 && T > 1 && T <= kVerifyMaxT) {
            if (d_spec_ == nullptr &&
                cudaMalloc((void**) &d_spec_, (size_t) kVerifyMaxT * (1 + strata::core::kSpecQStride) * sizeof(int32_t)) !=
                    cudaSuccess) {
                (void) cudaGetLastError();
                d_spec_ = nullptr;
            }
            if (d_spec_ != nullptr) {
                const int nq = std::min(spec_nq_, T - 1);
                int32_t* d_q = d_spec_ + kVerifyMaxT;
                if (cudaMemcpyAsync(d_spec_, tokens + 1, (size_t) (T - 1) * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) ==
                        cudaSuccess &&
                    cudaMemcpyAsync(d_q, spec_q_, (size_t) nq * strata::core::kSpecQStride * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, cs_) == cudaSuccess)
                    judged = sample_tokens_spec(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, d_spec_, d_q, nq,
                                                m_out_, cs_);
            }
        }
        spec_q_ = nullptr;
        spec_nq_ = 0;
        if (judged) ++spec_windows;
        else sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
#if defined(STRATA_GPU_METAL)
    (void) pool;
    (void) user;
    return strata::metal_engine::window_run(metal_, T, tokens, pos0, head_sampling_, out, err);
#endif
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    // STRATA_GPU_BUSY (opt-in): holds this device's clock up across the window's host-fed waits.  The
    // scope is the window, so every early return below releases the kernel.  It runs on its own
    // lowest-priority stream and touches no arithmetic - a token change here is not this feature's doing.
    const GpuBusyScope gpu_busy_scope;
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    // the overlapped split launched this stage's window already: keep the graph variant (#871) chosen for it then,
    // or this thread would serve the window as the other variant
    const bool pre = prelaunched_ == T;
    prelaunched_ = 0;
    if (!pre && pre_launch_ && !pre_launch_(cs_, err)) return false;   // (a prelaunched window's was served with the first stage's)
    if (!pre) refresh_ar();
    if (!capture(T, err) || !capture_commit(err)) return false;
    const bool chain = chain_overlap();
    if (chain) {   // capture the next stage's window here, on this thread, so the helper only stages and launches
        const OnDevice on_next(next_->device_);
        next_->refresh_ar();                     // #871: the variant its prelaunch and its run() will both use
        if (!next_->capture(T, err) || !next_->capture_commit(err)) return false;
    }
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    if (!pre && !staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    std::optional<HostGraphLaunch> launch_o;   // (not set when the previous stage launched this window already)
#endif
    if (!pre) {
        if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
        std::atomic_thread_fence(std::memory_order_seq_cst);
        trace_ev("WINDOW", -1, -1, pos0 * 16 + T);
        ms_host += ms_since(t0);
        VDBG("staged; launching\n");
        if (h_bal_ != nullptr) std::memset(h_bal_, 0, (size_t) bal_p_.size() * 4 * 8);   // this window's stamps only
        if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
        // 0.1.43: with the all-resident graph the host serves no layer, so nothing can hold a layer back: every copy first
        if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && ar_on() &&
            !adapt_gate_wait((int32_t) g.n_layers, err)) return false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
        // A blocking instrumented launch cannot also serve the graph's CPU doorbells.
        static const bool threaded_launch = [] {
            const char* value = std::getenv("STRATA_VERIFY_ASYNC_LAUNCH");
            return value != nullptr && value[0] == '1';
        }();
        launch_o.emplace(ar_off_ ? exec_nr_[T] : exec_[T], cs_, device_, threaded_launch,
                         [this] { (void) release_gpu_waits(5000); });
        const cudaError_t le = launch_o->status();
#else
        const cudaError_t le = cudaGraphLaunch(ar_off_ ? exec_nr_[T] : exec_[T], cs_);
#endif
        trace_ev("LAUNCHED", -1, -1, (int64_t) le);
        if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
        (void) launch_o->query();
#else
        (void) cudaStreamQuery(cs_);
#endif
        VDBG("launched\n");
    }
    // The overlapped split: the next stage's window is staged and launched on a helper thread while this thread
    // serves this stage's layers; its graph waits on the GPU for this stage's hand-off flag.  The guard joins the
    // helper on every return, and on an early return (an error after the launch) releases the next stage's GPU
    // waits so no spin kernel outlives the window (#267).
    struct NextLaunch {
        Verifier* next = nullptr;
        std::thread th;
        bool ok = true;
        bool handed = false;
        std::string err;
        bool join(std::string& e) {
            if (th.joinable()) th.join();
            if (!ok) e = err;
            return ok;
        }
        ~NextLaunch() {
            if (th.joinable()) th.join();
            if (next != nullptr && !handed) (void) next->release_gpu_waits(5000);
        }
    } nl;
    if (chain) {
        nl.next = next_;
        nl.th = std::thread([&nl, T, tokens, pos0] { nl.ok = nl.next->prelaunch(T, tokens, pos0, nl.err); });
    }
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    if (ar_on() && !test_stall) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
        }
    } else
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
                const cudaError_t q = launch_o ? launch_o->query() : cudaStreamQuery(cs_);
#else
                const cudaError_t q = cudaStreamQuery(cs_);
#endif
                if (q != cudaErrorNotReady && *seq < want) {
                    trace_ev("NEVER-RANG", k, l, (int64_t) q);
                    trace_dump(stderr);
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                // #267: the caller ends the engine; no spin kernel may outlive it
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
                trace_ev("TIMEOUT", k, l, (int64_t) (launch_o ? launch_o->query() : cudaStreamQuery(cs_)));
#else
                trace_ev("TIMEOUT", k, l, (int64_t) cudaStreamQuery(cs_));
#endif
                if (g_trace) {
                    diag(stderr);   // the words and the breadcrumbs before the release ...
                    const bool drained = release_gpu_waits(5000);
                    trace_dump(stderr);   // ... and after it: did the GPU move once its waits were raised?
                    err = "verify: timed out at layer " + std::to_string(l) + released_note(drained);
                    return false;
                }
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        if (g_trace) trace_ev("RANG", k, l, (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
        VDBG("layer %lld rang\n", (long long) l);
        if (adapt_gate_ != nullptr && adapt_gate_->active.load(std::memory_order_acquire) && l >= 0 &&
            (size_t) l < adapt_gate_->has.size() && adapt_gate_->has[(size_t) l] != 0 &&
            !adapt_gate_wait((int32_t) l, err)) return false;
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window: the CPU experts of layer", l);
        if (remote_opt_) remote_opt_->begin(h_w_ + (size_t) tb * ss.k, tb, n);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (remote_opt_) remote_opt_->end();
        VDBG("layer %lld served\n", (long long) l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", k, l,
                              (int64_t) ms_since(b));   // aux: ms the CPU experts took
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (h_bal_ != nullptr) {   // set_pcie_balance: the step's plan, its CPU and its PCIe experts
            const size_t bi = (size_t) ((k / G) * 2 + grp);
            bal_ok_[bi] = *(volatile uint32_t*) h_flagA_ == want;
            bal_nc_[bi] = bal_ok_[bi] ? sink_.cpu_jobs : 0;
            bal_np_[bi] = bal_ok_[bi] ? sink_.counts[2] : 0;
            sink_.cpu_jobs = 0;
        }
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        if (!(test_stall && k + 1 == steps)) *flag = want;
        if (h_bal_ != nullptr) bal_p_[(size_t) ((k / G) * 2 + grp)] = ms_since(b);
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    // (#646 staged the next stage's inputs here; 0.1.39b keeps the layer split's order: each stage stages its own)
    if (chain) {
        // The overlapped split: no host sync between the stages.  The next stage's GPU starts on this stage's flag;
        // this thread goes straight to serving the next stage's layers, and syncs this stage once the chain is done.
        if (!nl.join(err)) return false;
        nl.handed = true;
        ++windows;
        const bool ok = next_->run(T, tokens, pos0, pool, next_user_, out, err);
        const OnDevice back(device_);
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (copy_used_) {
            cudaStreamSynchronize(copy_);
            copy_used_ = false;
        }
        drain_pcie_timing();   // cs_ is synced: this window's H2D stamps are final
        if (!ok) return false;
        if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
        if (handoff_failed(err)) return false;   // a middle stage of three or more: its own hand-in
        if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871, as below
            *(volatile uint32_t*) h_plan_err_ = 0;
            err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
            return false;
        }
        commit_pending_ = false;
        if (prof_on_ && G == 1) collect_profile();
        return true;
    }
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process - the case that left Windows GPUs "lost" until a power cycle.  The wait
    // itself stays a blocking sync: a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver
    // beside the expert workers).
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    const cudaError_t launch_result = launch_o ? launch_o->finish() : cudaSuccess;
    if (launch_result != cudaSuccess) {
        err = std::string("verify: launch: ") + cudaGetErrorString(launch_result);
        return false;
    }
#endif
    trace_ev("SYNC", -1, -1, 0);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    trace_ev("SYNCED", -1, -1, (int64_t) se);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    if (handoff_failed(err)) return false;   // a stale hand-off: nothing of this window is used
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871: refresh_ar saw the table whole
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
    }
    commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (copy_used_) {
        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
        copy_used_ = false;
    }
    drain_pcie_timing();   // cs_ is synced: this window's H2D stamps are final
    if (h_bal_ != nullptr && !all_resident_ && !bal_paused_) pcie_balance_window(steps, G);
    if (prof_on_ && G == 1) collect_profile();   // the window's GPU stage stamps
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    if (!sample_window(T, tokens, pos0, err)) return false;
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    if (h_tpdump_ != nullptr) {
        for (int64_t ll = 0; ll < 2; ++ll) {
            const int32_t* id = (const int32_t*) (h_tpdump_ + (size_t) g.n_layers * 8 + 72 + (size_t) ll * 8);
            const float* ww = h_tpdump_ + (size_t) g.n_layers * 8 + 72 + (size_t) ll * 8 + 4;
            std::fprintf(stderr, "TPDUMP solo CMB l%lld %.9e  ROUTE ids=%d,%d,%d,%d w=%.6e,%.6e,%.6e,%.6e\n",
                         (long long) ll, (double) h_tpdump_[(size_t) g.n_layers * 8 + 40 + (size_t) ll * 4],
                         id[0], id[1], id[2], id[3], (double) ww[0], (double) ww[1], (double) ww[2], (double) ww[3]);
        }
        {
            const float* m0 = h_tpdump_ + (size_t) g.n_layers * 8 + 100;
            std::fprintf(stderr, "TPDUMP solo MOEIN l0 %.9e,%.9e | l1 %.9e,%.9e\n", (double) m0[0], (double) m0[1],
                         (double) m0[4], (double) m0[5]);
        }
        {
            const float* q3 = h_tpdump_ + (size_t) g.n_layers * 8 + 492;
            const int32_t* sel3 = (const int32_t*) (h_tpdump_ + (size_t) g.n_layers * 8 + 512);
            std::fprintf(stderr,
                         "TPDUMP solo QSA3 k=%.9e,%.9e v=%.9e,%.9e q=%.9e,%.9e qi=%.9e,%.9e s=%.6e,%.6e sel=%d,%d,%d,%d attn=%.9e,%.9e\n",
                         (double) q3[0], (double) q3[1], (double) q3[4], (double) q3[5],
                         (double) q3[8], (double) q3[9], (double) q3[12], (double) q3[13],
                         (double) q3[16], (double) q3[17],
                         sel3[0], sel3[1], sel3[2], sel3[3], (double) q3[24], (double) q3[25]);
            const float* m3 = h_tpdump_ + (size_t) g.n_layers * 8 + 524;
            std::fprintf(stderr, "TPDUMP solo M3 %.9e,%.9e\n", (double) m3[0], (double) m3[1]);
            const float* w3 = h_tpdump_ + (size_t) g.n_layers * 8 + 528;
            std::fprintf(stderr, "TPDUMP solo W3 wk=%.9e,%.9e wv=%.9e,%.9e wq=%.9e,%.9e rk=%.9e,%.9e rv=%.9e,%.9e\n",
                         (double) w3[0], (double) w3[1], (double) w3[4], (double) w3[5], (double) w3[8], (double) w3[9],
                         (double) w3[12], (double) w3[13], (double) w3[16], (double) w3[17]);
            const float* g0 = h_tpdump_ + (size_t) g.n_layers * 8 + 548;
            std::fprintf(stderr, "TPDUMP solo GDN0 q=%.9e,%.9e k=%.9e,%.9e v=%.9e,%.9e h=%.9e,%.9e\n",
                         (double) g0[0], (double) g0[1], (double) g0[4], (double) g0[5],
                         (double) g0[8], (double) g0[9], (double) g0[12], (double) g0[13]);
        }
        std::fprintf(stderr, "TPDUMP solo T=%d pos0=%lld", T, (long long) pos0);
        for (int64_t l = 0; l < g.n_layers; ++l)
            std::fprintf(stderr, " %.6e", (double) h_tpdump_[(size_t) l * 8]);
        std::fprintf(stderr, "\n");
        // STRATA_TP_VEC: the vector dumps as raw fp32, one file per window, appended in site order
        if (const int64_t vat = tp_vec_at(0, 0); vat >= 0) {
            const int nv = tp_vec_len();
            static const char* vf = std::getenv("STRATA_TP_VEC_FILE");
            static int vn = 0;
            if (vf != nullptr) {
                const std::string path = std::string(vf) + ".solo." + std::to_string(vn++);
                if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                    for (int64_t l = 0; l <= g.n_layers; ++l)
                        for (int site = 0; site < kTpVecSites; ++site)
                            std::fwrite(h_tpdump_ + tp_vec_at(l, site), sizeof(float), (size_t) nv, f);
                    std::fclose(f);
                }
            }
        }
    }
    // debug: the first non-finite head (STRATA_DBG_NAN_VERIFY: this check alone, without the prompt path's)
    static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr || std::getenv("STRATA_DBG_NAN_VERIFY") != nullptr;
    if (dbg) {
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            cudaMemcpy(h.data(), head_logits_, h.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

// set_pcie_balance: each stamped step's times into the model's fits (PcieModel::sample), solved once a window.
void Verifier::pcie_balance_window(int64_t steps, int G) {
    const volatile unsigned long long* st = h_bal_;
    PcieModel& M = *bal_;
    for (int64_t k = 0; k < steps; ++k) {
        const size_t i = (size_t) ((k / G) * 2 + k % G);
        if (!bal_ok_[i]) continue;
        const unsigned long long ta = st[4 * i], tb = st[4 * i + 1], tw = st[4 * i + 2];
        if (ta == 0 || tb < ta || tw < tb || tw - ta > 1000000000ull) continue;   // not stamped in this window
        M.sample(bal_nc_[i], bal_np_[i], bal_p_[i], (double) (tw - ta) * 1e-6);
    }
    M.solve();
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    sink_.dst2 = (unsigned long long*) (base + dst2_off_);
    const int G = last_batch_ ? 1 : (groups_[last_t_] > 0 ? groups_[last_t_] : 1);
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    v->copy_used_ = true;
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    // STRATA_DMA_BATCH (midhatn's #807, F12): the group's independent uploads as one cudaMemcpyBatchAsync.  Mode 0 is
    // the loop the engine always ran, errors unchecked (it never stops the engine).  A batch mode whose uploads
    // cannot be submitted stops the engine: flag B must not release the consumers to expert weights that were never
    // copied.
    const int batch_mode = strata::core::dma_batch_mode();
    if (batch_mode == 0) {
        for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    } else {
        const auto copied = strata::core::copy_expert_blobs(stage, src, n, bytes, v->copy_, batch_mode);
        if (copied != cudaSuccess) {
            std::fprintf(stderr, "strata DMA: expert upload failed: %s\n", cudaGetErrorString(copied));
            std::fflush(stderr);
            v->release_gpu_waits(5000);
            std::abort();
        }
    }
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
#if defined(STRATA_GPU_METAL)
    // the head logits are the Metal window's (window_copy_logits, row by row)
    if (n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    for (int t = 0; t < T; ++t)
        if (!strata::metal_engine::window_copy_logits(metal_, t, h.data() + (size_t) t * (size_t) n_vocab_)) {
            err = "window_logprobs: the head logits copy failed";
            return false;
        }
#else
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    if (cudaMemcpy(h.data(), head_logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "window_logprobs: the head logits copy failed";
        return false;
    }
#endif
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        // engine-latency (STRATA_COMMIT_BATCH, default on; =0 keeps the per-layer loop): every GDN layer's conv history and
        // state in two launches instead of two per layer (~70 graph nodes fewer; the same arithmetic per layer, bitwise).
        // The commit runs between a window and the draft round, which on HIP waits for it.
#if defined(STRATA_GPU_METAL)   // no batched commit kernels on Metal: the per-layer loop (the Mac engine's gated path)
        static const bool commit_batch = false;
#else
        static const bool commit_batch = [] { const char* e = std::getenv("STRATA_COMMIT_BATCH"); return !e || e[0] != '0'; }();
#endif
        bool gdn_batched = false;
        if (commit_batch) {
            int64_t n_gdn = 0;
            bool norms_ok = true;   // the per-layer loop checks each layer's ssm_norm.weight: keep that check
            for (int64_t l = lb_; l < le_ && norms_ok; ++l)
                if (!is_qsa_layer(g, l)) {
                    ++n_gdn;
                    norms_ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
                }
            if (!norms_ok) ok = false;
            else if (n_gdn > 0)
                gdn_batched = gdn_commit_layers(
                    ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats, (int64_t) gdn_floats,
                    (int64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size, qkv_L_ + (size_t) gdn_index * MT * C,
                    MT * C, h_L_ + (size_t) gdn_index * MT * C, MT * C, gate_L_ + (size_t) gdn_index * MT * HV,
                    beta_L_ + (size_t) gdn_index * MT * HV, MT * HV, (int) C, (int) g.ssm_k_heads, (int) HV, (int) MT,
                    (int) n_gdn, commit_, cs_);
        }
        // the QSA layers' tail restore + indexer append, one launch for all of them (one block per layer)
        bool qsa_batched = false;
        if (commit_batch && ok && !g_no_batch_kv) {
            QsaIndexerLayers il;
            int nq = 0;
            int64_t mc = -1;
            bool fits = true;
            for (int64_t l = lb_, qi = qsa_index; l < le_ && fits; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; fits = false; break; }
                const QsaState& st = ss.qsa_states[qi];
                if (nq >= QsaIndexerLayers::kMax || (mc >= 0 && st.max_cells != mc)) { fits = false; break; }
                mc = st.max_cells;
                il.gamma[nq] = (const float*) wikn->data;
                il.tail_src[nq] = tail_snap_ + (size_t) qi * TS;
                il.tail[nq] = st.idx_tail; il.dead[nq] = st.idx_dead; il.pooled[nq] = st.idx_pooled;
                il.block_pos[nq] = st.idx_block_pos;
                ++nq;
                ++qi;
            }
            if (ok && fits && nq > 0) {
                native_qsa_indexer_append_steps_layers(idx_raw_L_ + (size_t) qsa_index * MT * ID, MT * ID, commit_ + 2, 1,
                                                       (int) MT, 0, EPS, il, nq, s, mc, rope_scaling(), cs_);
                qsa_batched = true;
            }
        }
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (is_qsa_layer(g, l) && qsa_batched) {
                ++qsa_index;
            } else if (!is_qsa_layer(g, l) && gdn_batched) {
                ++gdn_index;
            } else if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_,
                                    (int) MT);   // S26: t_out_begin = MT - the replay's outputs (y_dummy_) are never read
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                if (!g_no_batch_kv) {
                    native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT, 0,
                                                    (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                } else {
                    for (int64_t t = 0; t < MT; ++t)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    // (through the same path as the window graphs: on a full card it frees older batch layouts and retries, and the error
    // it reports is the instantiate's own)
    const bool made = instantiate_evicting(commit_exec_, graph, {}, "verify: commit instantiate: ", err, 0);
    cudaGraphDestroy(graph);
    return made;
}

// ---- STRATA_GDN_VALDUMP staging (file-scope, allocated in tp_setup_scratch OUTSIDE capture, D2H'd in-capture,
// written at window-end).  The GDN out-projection runs INSIDE the CUDA graph capture, so the value vectors
// (y_full_/y_tp_, DEVICE pointers in the cudaMalloc'd take() arena) must be pulled out with cudaMemcpyAsync into
// PINNED host memory - exactly like the STRATA_TP_VEC site dumps (which fwrite the mapped buffer only OUTSIDE
// capture, in run_tp).  A cudaStreamSynchronize + fwrite, or even a cudaHostAlloc, INSIDE capture is illegal: it
// aborts the capture and poisons the stream ("operation failed due to a previous error during capture" at layer 0's
// allreduce).  ONE process runs BOTH ranks (ver + tps.ver), so these are per-process and per-rank: each rank owns
// its index and its files; the files are written from run_tp's window-end writer, after the graphs have replayed.
namespace {
float* g_gdnval_full_h_[2] = {nullptr, nullptr};   // [rank]: MT * ZV floats (pinned, persistent)
float* g_gdnval_half_h_[2] = {nullptr, nullptr};   // [rank]: MT * ZVt floats (pinned, persistent)
int    g_gdnval_T_ = 0;
int    g_gdnval_ZV_ = 0;
int    g_gdnval_ZVt_ = 0;
int    g_gdnval_fired_[2] = {0, 0};   // set in-capture when a rank recorded its D2H; written once at window-end
}  // namespace

// ---- the TP scratch: this rank's half-width pipeline buffers, its local conv weights, and the experts' FFN
// split (see tp_moe_split.hpp).  Everything here is allocated on THIS rank's device and launched on THIS
// rank's stream: the coordinator may call it on the peer's verifier while another card is current.
bool Verifier::tp_setup_scratch(std::string& err) {
    if (!tp_on_) { err = "verify: tp_setup_scratch on a verifier that is not a TP rank"; return false; }
    if (qkv_tp_ != nullptr) return true;   // said once
    const OnDevice on_self(device_);
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    const int64_t HV = g.ssm_v_heads;                   // 48 global
    const int64_t ZV = g.ssm_value_dim;                 // 6144 global
    const int64_t NH = g.n_head, HD = g.head_dim;       // 24, 256
    const int64_t MT = max_t_;
    const int64_t nG = g.n_gdn_layers();
    const int64_t Ct = strata::core::kTpGdnConvChannels;   // 7168 local conv channels
    const int64_t HVt = HV / 2;                            // 24 local v-heads
    const int64_t ZVt = ZV / 2;                            // 3072 local value dim
    const int64_t NHt = NH / 2;                            // 12 local q heads
    if ((HV % 2) != 0 || (ZV % 2) != 0 || (NH % 2) != 0) {
        err = "verify: tp: the GDN v-heads, the value dim and the q heads must be even";
        return false;
    }
    // The local conv history is [channels][3]; the local recurrent state is [row][v_head][col] with the LOCAL
    // v-head count, so it is S*(HV/2)*S + 7168*3 floats per GDN layer.
    const int64_t S = g.ssm_state_size;
    gdn_state_tp_floats_ = S * HVt * S + Ct * (g.ssm_d_conv - 1);
    uint64_t bytes = 0;
    bytes += (uint64_t) nG * MT * Ct * 4 * 2;     // qkv_tp_ + h_tp_
    bytes += (uint64_t) nG * MT * HVt * 4 * 2;    // gate_tp_ + beta_tp_
    bytes += (uint64_t) MT * ZVt * 4 * 2;         // z_tp_ + y_tp_
    bytes += (uint64_t) MT * ZV * 4;              // y_full_ (rank 0's full-width GDN value buffer)
    bytes += (uint64_t) MT * (g.n_embd / 2) * 4;   // half_out_ (the out-projections' compact [T][N/2])
    bytes += (uint64_t) nG * Ct * g.ssm_d_conv * 4;                // conv_w_tp_ (4 taps)
    bytes += (uint64_t) nG * gdn_state_tp_floats_ * 4;             // gdn_state_tp_
    void* arena = nullptr;
    if (cudaMalloc(&arena, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "verify: tp: the local pipeline scratch could not be allocated (" + std::to_string(bytes >> 20) +
              " MiB on this card)";
        return false;
    }
    tp_arena_ = arena;
    uint8_t* p = (uint8_t*) arena;
    auto take = [&](uint64_t n, uint64_t esize) -> void* {
        void* r = p;
        p += (n * esize + 255) & ~255ull;
        return r;
    };
    qkv_tp_ = (float*) take((uint64_t) nG * MT * Ct, 4);
    h_tp_ = (float*) take((uint64_t) nG * MT * Ct, 4);
    gate_tp_ = (float*) take((uint64_t) nG * MT * HVt, 4);
    beta_tp_ = (float*) take((uint64_t) nG * MT * HVt, 4);
    z_tp_ = (float*) take((uint64_t) MT * ZVt, 4);
    y_tp_ = (float*) take((uint64_t) MT * ZVt, 4);
    y_full_ = (float*) take((uint64_t) MT * ZV, 4);
    half_out_ = (float*) take((uint64_t) MT * (g.n_embd / 2), 4);
    conv_w_tp_ = (float*) take((uint64_t) nG * Ct * g.ssm_d_conv, 4);   // 4 taps per channel
    gdn_state_tp_ = (float*) take((uint64_t) nG * gdn_state_tp_floats_, 4);
    const cudaError_t z = cudaMemsetAsync(arena, 0, bytes, cs_);
    if (z != cudaSuccess) {
        err = std::string("verify: tp: the local scratch could not be zeroed: ") + cudaGetErrorString(z);
        return false;
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "verify: tp: the local scratch's zeroing failed";
        return false;
    }
    // This rank's 7168 of the 10240 conv channels per GDN layer (q/k are replicated, v is this card's half).
    {
        int64_t gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) continue;
            const WeightRef* wc = need(LayerView(wt, l), "ssm_conv1d.weight", err);
            if (wc == nullptr) return false;
            strata::core::tp_local_conv_weights((const float*) wc->data, tp_rank_, conv_w_tp_ + (size_t) gi * Ct * 4,
                                                cs_);
            ++gi;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = "verify: tp: the local conv weights could not be extracted";
            return false;
        }
        if (gi != nG) {
            err = "verify: tp: the pack holds " + std::to_string(gi) + " GDN layers, the geometry " +
                  std::to_string(nG);
            return false;
        }
    }
    // The experts' FFN split for this rank.  The layouts are per (gu_type, d_type) pair, so the pack's format
    // is resolved here once and every layer must agree (one expert format per model).
    {
        const auto& lay = strata::kernels::cpu::expert_layout();
        std::string why;
        bool ok = lay.native;
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const auto& f = lay.fmt[(size_t) l];
            TpMoeSplit ms;
            if (!tp_moe_split(f.gu_type, f.d_type, f.n_embd, f.n_ff, tp_rank_, ms, why)) {
                ok = false;
                break;
            }
            if (l == 0) tp_moe_ = ms;
            // every layer's split must address the same blob geometry, or one layout cannot serve them all
            else if (ms.ptr_off != tp_moe_.ptr_off || ms.layout.n_ff != tp_moe_.layout.n_ff ||
                     ms.layout.gu_row != tp_moe_.layout.gu_row || ms.layout.d_row != tp_moe_.layout.d_row) {
                ok = false;
                why = "the layers' expert formats differ";
            }
        }
        if (!ok) {
            err = "verify: tp: the experts' FFN cannot be split across the two cards: " + why +
                  " (a tensor-parallel window needs both cards on every layer's experts)";
            return false;
        }
        tp_moe_scratch_ = nullptr;
        const int64_t capx = (int64_t) MT * ss_->k;
        // gate/up scratch (the local width) and the FULL-width hq the down reads, ONE each per group set
        // (the two share of a layer must not clobber the first before the exchange); the compact down output
        // is a single buffer - the two group sets' entries' dst rows are disjoint (an entry is in exactly one
        // tier), so the second down cannot clobber the first's.
        const size_t local_scratch = tp_moe_split_scratch_bytes(capx, tp_moe_);
        // The down's scratch is a FULL-width buffer: the down phase reads its hq at `scratch + 3*fa` (fa from
        // L.n_ff = the full width), so passing a full-width scratch puts that hq at hq_off(capx, n_ff_full) -
        // exactly where moe_hq_exchange assembles the two ranks' halves.  The gate|up|h triple at the front of
        // that buffer is unused by the down phase (its fa would differ), just padding.
        const int64_t n_ff_full = tp_moe_.layout.n_ff * 2;
        const size_t down_scratch = strata::kernels::native_expert_scratch_bytes(capx, n_ff_full);
        if (cudaMalloc(&tp_moe_scratch_, local_scratch) != cudaSuccess ||
            cudaMalloc(&tp_moe_scratch2_, local_scratch) != cudaSuccess ||
            cudaMalloc(&tp_moe_hq_, down_scratch) != cudaSuccess ||
            cudaMalloc(&tp_moe_hq2_, down_scratch) != cudaSuccess ||
            cudaMalloc(&tp_moe_hitc_, (size_t) capx * (size_t) (g.n_embd / 2) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &tp_ptr_, (size_t) capx * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMalloc((void**) &tp_ptr2_, (size_t) capx * sizeof(unsigned long long)) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: the expert scratch could not be allocated";
            return false;
        }
        tp_moe_ok_ = true;
        // This rank's byte windows of every layer's blob (its half of gate, of up and of down, plus the NVFP4
        // tail): the PCIe share's fetch copies only these, so each card's link carries half a blob instead of a
        // whole one (fetch_blobs_spans).  The windows come from core::tp_blob_ranges - the SAME definition the
        // prompt path's DMA uses, so the two moving paths and the kernels' own addressing cannot drift apart
        // (that header's down window is `down_ptr_off + down.down_off`: the rank's down rows live in the LAYOUT,
        // not in a pointer offset).  The layout is per layer (up_off/down_off/tail_off can differ), so every layer
        // is resolved; if any window is not 16-byte aligned the fetch falls back to the whole blob (the alignment
        // depends on the pack's row strides, which are format constants, so both ranks agree either way).
        {
            size_t windowed = 0;
            bool ok = g.n_layers > 0, cover = ok;
            tp_fetch_spans_.assign((size_t) g.n_layers, strata::kernels::FetchSpans{});
            for (int64_t l = 0; l < g.n_layers && ok; ++l) {
                const int64_t blob = (int64_t) lay.blob_bytes(l);
                const auto& f = lay.fmt[(size_t) l];
                // BOTH ranks' windows: this rank's go into tp_fetch_spans_ (its fetch), the pair's union decides
                // whether an ADMITTED blob (a cache slot both ranks later read) can be filled by both ranks at
                // once - see tp_slot_cover_.  Windows always copy source -> the same offsets, so overlaps are
                // harmless (both ranks read the same NVFP4 tail); only a HOLE would leave stale bytes in a slot.
                core::TpBlobRange both[8];
                int nb = 0;
                for (int r = 0; r < 2 && ok; ++r) {
                    TpMoeSplit ms;
                    std::string why;
                    core::TpBlobRange q[4];
                    if (!tp_moe_split(f.gu_type, f.d_type, f.n_embd, f.n_ff, r, ms, why)) {
                        ok = false;   // the geometry check above already passed: treat as "no windows", copy whole
                        break;
                    }
                    const int nq = core::tp_blob_ranges_for_layer(lay, l, ms, q);
                    if (nq <= 0) {
                        ok = false;
                        break;
                    }
                    for (int j = 0; j < nq; ++j) both[nb++] = q[j];
                    if (r != tp_rank_) continue;
                    strata::kernels::FetchSpans& s = tp_fetch_spans_[(size_t) l];
                    s.nr = nq;
                    for (int j = 0; j < nq; ++j) {
                        s.off[j] = (int64_t) q[j].off;
                        s.len[j] = (int64_t) q[j].bytes;
                        if (s.off[j] < 0 || s.len[j] <= 0 || (s.off[j] & 15) != 0 || (s.len[j] & 15) != 0 ||
                            s.off[j] + s.len[j] > blob) {
                            ok = false;
                            break;
                        }
                        windowed += (size_t) s.len[j];
                    }
                }
                if (!ok) break;
                for (int a = 1; a < nb; ++a) {   // insertion sort by offset (nb <= 8)
                    const core::TpBlobRange t = both[a];
                    int b = a - 1;
                    while (b >= 0 && both[b].off > t.off) { both[b + 1] = both[b]; --b; }
                    both[b + 1] = t;
                }
                int64_t reached = 0;
                for (int a = 0; a < nb; ++a) {
                    const int64_t o = (int64_t) both[a].off, e = o + (int64_t) both[a].bytes;
                    if (o > reached) { cover = false; break; }   // a hole: the slot would keep stale bytes
                    if (e > reached) reached = e;
                }
                if (reached != blob) cover = false;
            }
            tp_half_fetch_ = ok;
            tp_slot_cover_ = ok && cover;
            if (tp_slot_cover_) {   // measured slower (see the member's comment): opt-in only
                const char* e = std::getenv("STRATA_TP_TWO_RANK_SLOT");
                tp_two_rank_slot_ = e != nullptr && e[0] != '0';
            }
            if (tp_half_fetch_) {
                std::fprintf(stderr, "strata verify: tp: rank %d fetches %zu of %zu blob bytes per expert over PCIe"
                                     " (its half; the peer fetches the other half over its own link); the pair's"
                                     " windows %s the blob, so %s\n",
                             tp_rank_, windowed / (size_t) g.n_layers, (size_t) lay.blob_bytes(0),
                             tp_slot_cover_ ? "tile" : "do NOT tile",
                             tp_slot_cover_ ? (tp_two_rank_slot_ ? "an admitted cache slot is filled by both ranks"
                                                                 : "rank 0 fills an admitted slot whole and this"
                                                                   " rank skips it (STRATA_TP_TWO_RANK_SLOT=1"
                                                                   " splits the fill, measured slower)")
                                            : "an admitted cache slot is filled whole by rank 0");
            }
        }
        // DIAG (STRATA_MOE_HQ_DUMP=1): pinned staging for the capture-safe layer-0 hq dump.  The in-graph D2H
        // node (record_window_tp's dump_moe_hq_diag) fills the front with the ASSEMBLED set0 hq and the back
        // with this rank's LOCAL set0 gate/up hq; run_tp writes both ranks' buffers to disk after the join.
        // q8_1 block = 36 bytes (4B scale + 32B int8 codes); full hq = n_ff/32 blocks, local = n_ff/2/32.
        // Layout in the staging (per rank): [set0 hq][set0 lq][set2 hq][set2 lq][hit_out (this rank's pitched
        // MoE band, N floats per entry)].  hit_out is dumped AFTER both down_split calls (it holds the pitched
        // band and the CPU share that follows writes parts_, not hit_out_).
        const int64_t blocks_full = (int64_t) (tp_moe_.layout.n_ff * 2) / 32;   // full n_ff 640/32 = 20 blocks
        const int64_t blocks_local = (int64_t) tp_moe_.n_ff_local / 32;         // local 320/32 = 10 blocks
        moe_hq_diag_hq_bytes_ = (int64_t) capx * blocks_full * 36;             // per set hq: cap x 720 B
        const int64_t lq_bytes = (int64_t) capx * blocks_local * 36;           // per set lq: cap x 360 B
        moe_hq_dump_hq_sz_ = (size_t) moe_hq_diag_hq_bytes_;
        moe_hq_dump_hitc_sz_ = (size_t) capx * (size_t) g.n_embd * sizeof(float);   // hit_out: capx x N f32
        // 4 hq/lq regions (layer 0 only) + hit_out for EVERY layer (STRATA_MOE_HQ_DUMP=1: the per-layer MoE band,
        // so the split's first diverging layer is found against the full-width mofull band).
        const size_t total = 2 * (size_t) moe_hq_diag_hq_bytes_ + 2 * (size_t) lq_bytes +
                             (size_t) g.n_layers * moe_hq_dump_hitc_sz_;
        if (!mapped(total, (void**) &h_moe_hq_diag_, (void**) &m_moe_hq_diag_)) {
            err = "verify: tp: the MoE-hq dump staging could not be mapped";
            return false;
        }
    }
    // STRATA_GDN_VALDUMP staging: PERSISTENT pinned host buffers for the per-rank GDN value vectors (y_full_ and
    // y_tp_).  They must be allocated HERE, OUTSIDE the graph capture: a cudaHostAlloc is not a legal capture op, so
    // allocating it inside record_window_tp (as an earlier attempt did) poisons the capture and the window dies at
    // layer 0's allreduce with "operation failed due to a previous error during capture".  Allocated once for the
    // max window (MT); each window's T <= MT, so the buffers are large enough.  The in-capture D2H and the window-end
    // fwrite mirror the STRATA_MOE_HQ_DUMP / STRATA_TP_VEC pattern (pinned buffer, graph-node D2H, write outside).
    if (std::getenv("STRATA_GDN_VALDUMP") != nullptr && tp_rank_ >= 0 && tp_rank_ < 2 &&
        g_gdnval_full_h_[tp_rank_] == nullptr) {
        if (cudaHostAlloc(&g_gdnval_full_h_[tp_rank_], (size_t) MT * (size_t) ZV * sizeof(float), 0) != cudaSuccess ||
            cudaHostAlloc(&g_gdnval_half_h_[tp_rank_], (size_t) MT * (size_t) ZVt * sizeof(float), 0) != cudaSuccess) {
            (void) cudaGetLastError();
            g_gdnval_full_h_[tp_rank_] = nullptr;
            g_gdnval_half_h_[tp_rank_] = nullptr;
        }
    }
    std::fprintf(stderr,
                 "strata verify: tp: rank %d local scratch %llu MiB (qkv/h %lld conv channels, gate/beta %lld heads, "
                 "state %.1f MiB over %lld GDN layers, experts at n_ff %lld per card)\n",
                 tp_rank_, (unsigned long long) (bytes >> 20), (long long) Ct, (long long) HVt,
                 (double) ((uint64_t) nG * gdn_state_tp_floats_ * 4) / (1024.0 * 1024.0), (long long) nG,
                 (long long) tp_moe_.n_ff_local);
    (void) NHt;
    return true;
}

uint8_t* Verifier::tp_moe_local_hq(int set) const {
    // The peer's exchange adds the PER-WINDOW hq offset (hq_off(cap, n_ff_local), cap = the window's entry
    // count) to this base itself, so the pointer handed over P2P must be the raw gate/up scratch BASE - the
    // same `scratch` the local `exchange_hq` reads its own local hq from.
    return (uint8_t*) (set ? tp_moe_scratch2_ : tp_moe_scratch_);
}

bool Verifier::tp_zero_persistent(std::string& err) {
    if (!tp_on_) return true;
    if (gdn_state_tp_ == nullptr) { err = "verify: tp_zero_persistent before tp_setup_scratch"; return false; }
    const OnDevice on_self(device_);
    const int64_t nG = g_->n_gdn_layers();
    const uint64_t bytes = (uint64_t) nG * gdn_state_tp_floats_ * 4;
    if (bytes == 0) return true;
    const cudaError_t z = cudaMemsetAsync(gdn_state_tp_, 0, bytes, cs_);
    if (z != cudaSuccess) { err = "verify: tp: zeroing the local GDN state failed: " + std::string(cudaGetErrorString(z)); return false; }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify: tp: the local GDN state's zeroing did not complete"; return false; }
    return true;
}

bool Verifier::tp_setup_flags(std::string& err) {
    if (!tp_on_) { err = "verify: tp_setup_flags on a verifier that is not a TP rank (set_tp)"; return false; }
    if (tp_peer_ == nullptr || !tp_peer_->tp_on_) {
        err = "verify: tp_setup_flags needs the peer verifier (set_tp_peer) with set_tp on it";
        return false;
    }
    if (tp_flag_a_h_ != nullptr && tp_flag_b_h_ != nullptr) return true;   // said once
    // The pair: rank 0's allreduce kernel reads and writes BOTH buffers, so the two cards must see each other
    // (P2P over NVLink on this box).  open() is idempotent per device pair and reports a plain-PCIe fallback.
    // Rank 1 opens the pair in the MIRRORED order (dev_a = rank 0's card, dev_b = its own): its fused handshake
    // runs on dev_b (publish_wait) and the full-width GDN out-projection has it write its value half into rank 0's
    // buffer (broadcast_to_a, also dev_b).  Both need P2P enabled on this card.
    if (tp_rank_ == 0) {
        if (!tp_ar_.open(device_, tp_peer_dev_, err)) return false;
    } else if (!tp_ar_.open(tp_peer_dev_, device_, err)) {
        return false;
    }
    // the host words are allocated once; each card resolves its own device view of them below
    const OnDevice on_self(device_);
    uint32_t* ha = nullptr;
    uint32_t* hb = nullptr;
    void* da = nullptr;
    void* db = nullptr;
    if (!mapped(64, (void**) &ha, &da) || !mapped(64, (void**) &hb, &db)) {
        err = "verify: tp: the mapped handshake flags could not be allocated";
        return false;
    }
    uint32_t* hc = nullptr;
    void* dc = nullptr;
    if (!mapped(64, (void**) &hc, &dc)) {
        err = "verify: tp: the mapped MoE-broadcast flag could not be allocated";
        return false;
    }
    uint32_t* hd = nullptr;
    void* dd = nullptr;
    uint32_t* he = nullptr;
    void* de = nullptr;
    uint32_t* hf = nullptr;
    void* df = nullptr;
    uint32_t* hg = nullptr;
    void* dg = nullptr;
    uint32_t* hh = nullptr;
    void* dh = nullptr;
    uint32_t* hi = nullptr;
    void* di = nullptr;
    uint32_t* hj = nullptr;
    void* dj = nullptr;
    if (!mapped(64, (void**) &hd, &dd) || !mapped(64, (void**) &he, &de) || !mapped(64, (void**) &hf, &df) ||
        !mapped(64, (void**) &hg, &dg) || !mapped(64, (void**) &hh, &dh) || !mapped(64, (void**) &hi, &di) ||
        !mapped(64, (void**) &hj, &dj)) {
        err = "verify: tp: the mapped MoE-allreduce flags could not be allocated";
        return false;
    }
    tp_flag_a_h_ = ha; tp_flag_a_d_ = (uint32_t*) da;
    tp_flag_b_h_ = hb; tp_flag_b_d_ = (uint32_t*) db;
    tp_flag_c_h_ = hc; tp_flag_c_d_ = (uint32_t*) dc;
    tp_flag_d_h_ = hd; tp_flag_d_d_ = (uint32_t*) dd;
    tp_flag_e_h_ = he; tp_flag_e_d_ = (uint32_t*) de;
    tp_flag_f_h_ = hf; tp_flag_f_d_ = (uint32_t*) df;
    tp_flag_g_h_ = hg; tp_flag_g_d_ = (uint32_t*) dg;
    tp_flag_h_h_ = hh; tp_flag_h_d_ = (uint32_t*) dh;   // MoE hq exchange, rank 1 -> rank 0
    tp_flag_j_h_ = hj; tp_flag_j_d_ = (uint32_t*) dj;   // MoE band stitch, rank 1 -> rank 0
    tp_flag_i_h_ = hi; tp_flag_i_d_ = (uint32_t*) di;   // MoE hq exchange, rank 0 -> rank 1
    // The device pointer is per-device: cudaHostGetDevicePointer resolves the host range in the CALLING
    // device's UVA space, and a pointer resolved on one card is not valid on another.  The peer therefore
    // resolves the same host words again with its own device current.
    if (tp_peer_ != nullptr) {
        const OnDevice on_peer(tp_peer_->device_);
        void* pa = nullptr;
        void* pb = nullptr;
        if (cudaHostGetDevicePointer(&pa, ha, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&pb, hb, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the handshake flags";
            return false;
        }
        tp_peer_->tp_flag_a_h_ = ha;
        tp_peer_->tp_flag_a_d_ = (uint32_t*) pa;
        tp_peer_->tp_flag_b_h_ = hb;
        tp_peer_->tp_flag_b_d_ = (uint32_t*) pb;
        // the MoE-broadcast flag, mapped into the peer's UVA space the same way
        void* pc = nullptr;
        if (cudaHostGetDevicePointer(&pc, hc, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the broadcast flag";
            return false;
        }
        tp_peer_->tp_flag_c_h_ = hc;
        tp_peer_->tp_flag_c_d_ = (uint32_t*) pc;
        // the MoE combine's allreduce pair, mapped the same way
        void* pd = nullptr;
        void* pe = nullptr;
        if (cudaHostGetDevicePointer(&pd, hd, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&pe, he, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the MoE-allreduce flags";
            return false;
        }
        tp_peer_->tp_flag_d_h_ = hd;
        tp_peer_->tp_flag_d_d_ = (uint32_t*) pd;
        tp_peer_->tp_flag_e_h_ = he;
        tp_peer_->tp_flag_e_d_ = (uint32_t*) pe;
        void* pf = nullptr;
        if (cudaHostGetDevicePointer(&pf, hf, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the staging flag";
            return false;
        }
        tp_peer_->tp_flag_f_h_ = hf;
        tp_peer_->tp_flag_f_d_ = (uint32_t*) pf;
        void* pg = nullptr;
        if (cudaHostGetDevicePointer(&pg, hg, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the GDN flag";
            return false;
        }
        tp_peer_->tp_flag_g_h_ = hg;
        tp_peer_->tp_flag_g_d_ = (uint32_t*) pg;
        void* ph = nullptr;
        void* pi = nullptr;
        if (cudaHostGetDevicePointer(&ph, hh, 0) != cudaSuccess || cudaHostGetDevicePointer(&pi, hi, 0) !=
            cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the MoE-exchange flags";
            return false;
        }
        tp_peer_->tp_flag_h_h_ = hh;
        tp_peer_->tp_flag_h_d_ = (uint32_t*) ph;
        tp_peer_->tp_flag_i_h_ = hi;
        tp_peer_->tp_flag_i_d_ = (uint32_t*) pi;
        void* pj = nullptr;
        if (cudaHostGetDevicePointer(&pj, hj, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "verify: tp: CUDA" + std::to_string(tp_peer_->device_) + " could not map the stitch flag";
            return false;
        }
        tp_peer_->tp_flag_j_h_ = hj;
        tp_peer_->tp_flag_j_d_ = (uint32_t*) pj;
    }
    std::fprintf(stderr, "strata verify: tp: rank %d on CUDA%d (peer CUDA%d), handshake flags in mapped host memory\n",
                 tp_rank_, device_, tp_peer_dev_);
    return true;
}

bool Verifier::capture_tp(int T, std::string& err) {
    if (T < 1 || T > max_t_) { err = "verify: tp window size out of range"; return false; }
    if (tp_exec_[T] != nullptr) return true;   // captured once per window size
    const OnDevice on_self(device_);   // the coordinator may call this on the peer's verifier
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: tp: begin window capture failed";
        return false;
    }
    bool ok = true;
    try {
        ok = record_window_tp(T, cs_, err);
    } catch (const std::exception& e) {
        ok = false;
        err = std::string("verify: tp window: ") + e.what();
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph != nullptr) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        err = "verify: tp: end window capture: " + std::string(cudaGetErrorString(ce));
        return false;
    }
    const bool made = instantiate_evicting(tp_exec_[T], graph, {}, "verify: tp: instantiate: ", err);
    cudaGraphDestroy(graph);
    if (!made) return false;
    const cudaError_t ue = cudaGraphUpload(tp_exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: tp: captured the %d-token window on rank %d (upload %s, sync %s)\n", T,
                 tp_rank_, cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

// ===================================== the TP window =====================================
// The default decode path of record_window with the dense GDN and QSA work split across the two cards, a
// per-layer allreduce of the row-parallel out-projection's partial sums, and the experts' FFN cut in two.
//
//   * the hyper-connection reads/writes, the router, the shared expert and the head all read the FULL residual
//     (both cards hold it after the allreduce), so they run unchanged on both cards.
//   * the GDN mixer splits its v-heads in two.  q/k are replicated inside the conv (head pairing is h % h_k), so
//     each card's conv output is 7168 channels and its `ssm_out` holds the 3072 columns of its own v half.
//   * the QSA mixer replicates the query/key/value projections and the attention itself (the attention kernel's
//     KV-head group is a compile-time constant), and splits `attn_output`'s reduction axis over q-head halves.
//   * the experts' FFN is split by intermediate width (tp_moe_split.hpp): each card runs gate/up rows and down
//     columns [rank*n_ff/2, (rank+1)*n_ff/2) of every routed expert, and the two partial residuals are summed.
//
// Per layer the two cards hand off through mapped flags (tp_flag_a_/b_ for the out-projection's allreduce,
// d_/e_ for the combine's, f_ for the PCIe staging), captured INSIDE the graphs as spin-wait kernels - a CUDA
// graph is single-device, so no event can order them:
//
//   rank 0:  ... out-proj partial -> wait(b, l+1) -> allreduce(bo_, peer) -> raise(a, l+1) -> ... experts ->
//            wait(d, l+1) -> allreduce(bo_, peer) -> raise(e, l+1)
//   rank 1:  ... out-proj partial -> raise(b, l+1) -> wait(a, l+1) -> ... experts -> raise(d, l+1) -> wait(e, l+1)
//
// NOTE on the residual write: like the ordinary window, a non-final layer does NOT write the residual streams
// in `post`.  Its FFN write is folded into the NEXT layer's hyper-connection read (`gr_read_group`'s `apply`).
// Writing it here as well applied it twice - the single card's layer 0 `bo_` matched to 1e-8 while layer 1's
// residual was 3% off.
bool Verifier::record_window_tp(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes sh = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int64_t S = g.ssm_state_size;
    const int64_t Ct = strata::core::kTpGdnConvChannels;   // 7168 local conv channels
    const int64_t HVt = HV / 2;                            // 24 local v-heads
    const int64_t ZVt = ZV / 2;                            // 3072 local value dim
    const int head_off = (int) (tp_rank_ * HVt);           // this rank's first global v-head
    const int64_t TS = (sh.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = 1;   // one group: a TP window is never split (the two cards are the parallelism)
    const int te = T;
    const bool self_commit = false;   // the commit graph advances the state; a TP window never writes it
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    static const bool fuse_head_gr = [] {
        const char* e = std::getenv("STRATA_FUSE_HEAD_GR");
        return e && e[0] == '1';
    }() && (le_ == g.n_layers) && (head_ != nullptr && head_->loaded()) && !cvec().covers(g.n_layers - 1);
    auto stamp = [&](int64_t l, int i) {
        if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
        if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2), cs);
    };
    if (qkv_tp_ == nullptr || conv_w_tp_ == nullptr || gdn_state_tp_ == nullptr) {
        err = "verify: tp: record_window_tp before tp_setup_scratch";
        return false;
    }
    if (tp_flag_a_d_ == nullptr || tp_flag_b_d_ == nullptr) {
        err = "verify: tp: record_window_tp before tp_setup_flags";
        return false;
    }
    if (tp_rank_ == 0 && tp_bo_peer_ == nullptr) {
        // the coordinator needs the peer's bo_ as a P2P pointer: set_tp_bo_peer(peer->tp_bo()) exposed it
        err = "verify: tp: rank 0 has no peer bo_ buffer (set_tp_bo_peer) - the allreduce has nowhere to write";
        return false;
    }

    // ---- the window's inputs
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings (replicated: every card holds the full table)
    if (const NativeEmbed* ne = native_embed()) {
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l): up to the ring
    // STRATA_TP_FUSED=0: the earlier three-launch handshake form, kept for A/B against the fused one
    static const bool tp_fused = [] {
        const char* v = std::getenv("STRATA_TP_FUSED");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // STRATA_TP_ATTN_FULL=1 (EXPERIMENTAL, default off): rank 0 computes the whole attention out-projection.
    // It is bit-exact but gives up the parallel gain, and on its own it did not make the tokens match.
    static const bool attn_full = [] {
        const char* v = std::getenv("STRATA_TP_ATTN_FULL");
        return v != nullptr && std::atoi(v) != 0;   // opt-in: full-width out-projection (bit-exact, no parallel gain)
    }();
    // STRATA_TP_GDN_FULL=1 (EXPERIMENTAL, default off): rank 0 computes the whole GDN out-projection from
    // both cards' value halves.  NOT CORRECT YET: enabling it produced wrong tokens at 7.3 tok/s (the P2P
    // hand-off between the two value halves is not ordered correctly), so the row-parallel split stays default.
    static const bool gdn_full = [] {
        const char* v = std::getenv("STRATA_TP_GDN_FULL");
        return v != nullptr && std::atoi(v) != 0;   // opt-in: the full-width GDN path is not bit-exact yet
    }();
    auto pre = [&](int64_t l) -> bool {
        stamp(l, 0);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        bool pending = l > 0 && !cvec().covers(l - 1);
        // the PLE stage on layer 1 (replicated: the table and the history are the same on both cards)
        if (l == 1 && ple_on) {
            if (ar_on()) wait_flag_ge(m_flag_, 1, cs);
            copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            PleWeights pw = ss.ple.w;
            // The residual write, in the ordinary window's batch form: the fused `native_gr_post_multi` reduces
            // the hyper-connection's streams in its own order, so it is not bitwise the per-token gr_write.
            const bool multi_write = dec_batch && !g_no_multi_gr && T > 1;
            if (multi_write) gr_write_multi(Rt(0), bo_, inj2_, gs, Rt(0), (int) T, cs);
            for (int t = 0; t < te; ++t) {
                if (!multi_write) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = ss.ple.hist;
                try {
                    ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = "verify: tp PLE: " + std::string(e.what());
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * (int64_t) NG_HIST * NG_HC_DIM, hist,
                                 (int64_t) NG_HIST * NG_HC_DIM, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                a.q8_inject = (const uint8_t*) wi[half]->hc_q8;
                a.s_down = wd[half]->q8_scale; a.s_up = wu[half]->q8_scale;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
                if (qcnt_ != nullptr) {
                    a.q8_cnt = qcnt_;
                    if (half == 0) a.q8_mixed = xq_ + (size_t) t * (N / 32) * 36;
                    else if (strata::kernels::cpu::expert_layout().native) a.q8_mixed = nat_xq_ + (size_t) t * (N / 32) * 36;
                }
            }
            if (strata::kernels::fused_gr_read_persist(fa, T, cs)) return false;
            return fused_gr_read_multi(fa, T, xn_, cs, (prof_on_) ? prof_ : nullptr,
                                       (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        const bool q8_attn = gr_read_group(0, pending, inj2_, inj_);
        stamp(l, 1);
        float* xm = mixed_;
        if (m_tpdump_ != nullptr && l == 1)   // DIAG: this rank's mixer output (the MoE's input)
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 96), xm, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        bool il_ready = false;
        auto quant_in = [&](const float* x, int n_in) {
#if !defined(STRATA_USE_HIP)
            static const int qil_env = [] { const char* v = std::getenv("STRATA_QIL"); return v != nullptr ? std::atoi(v) : -1; }();
            const bool qil = qil_env >= 0 ? qil_env != 0 : device_is_sm75();
            if (qil && T >= 2 && T <= 4 && g_mmvq_il()) {
                strata::kernels::native_quantize_q8_1_il(x, xq_, xil_, n_in, T, cs);
                il_ready = true;
                return;
            }
#endif
            native_quantize_q8_1(x, xq_, n_in, T, cs);
            il_ready = false;
        };
        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {
            if (g_mmvq_il() && strata::kernels::native_mmvq_il_supported(w->native_type, T, n_out)) {
                if (!il_ready && T >= 2) {
                    strata::kernels::native_q8_1_interleave(xq_, xil_, n_in, T, cs);
                    il_ready = true;
                }
                strata::kernels::native_mmvq_il(w->native_type, w->native_data, xq_, xil_, out, n_in, n_out, T, cs);
            } else {
                native_mmvq(w->native_type, w->native_data, xq_, out, n_in, n_out, T, cs);
            }
        };
        // ---- the row-parallel out-projection's partial, then the per-layer handshake and allreduce
        auto ring = [&](int64_t l) -> bool {
            const uint32_t want = (uint32_t) (l + 1);   // global layer index + 1
            std::string aerr;
            if (!tp_fused) {
                if (tp_rank_ == 0) {
                    wait_flag_ge(tp_flag_b_d_, want, cs);
                    if (!tp_ar_.allreduce(bo_, tp_bo_peer_, (int64_t) T * N, cs, nullptr, aerr)) {
                        err = "verify: tp allreduce at layer " + std::to_string(l) + ": " + aerr;
                        return false;
                    }
                    tp_raise_flag(tp_flag_a_d_, want, cs);
                } else {
                    tp_raise_flag(tp_flag_b_d_, want, cs);
                    wait_flag_ge(tp_flag_a_d_, want, cs);
                }
                return true;
            }
            if (tp_rank_ == 0) {
                // wait for the peer's partial, sum both buffers, then tell the peer (one launch, P1)
                if (!tp_ar_.allreduce_wait_raise(bo_, tp_bo_peer_, (int64_t) T * N, cs, tp_flag_b_d_, want,
                                                 tp_flag_a_d_, want, aerr)) {
                    err = "verify: tp allreduce at layer " + std::to_string(l) + ": " + aerr;
                    return false;
                }
            } else {
                // publish this card's partial, then wait for the sum to land in bo_ over P2P (one launch, P1)
                if (!tp_ar_.publish_wait(tp_flag_b_d_, want, tp_flag_a_d_, want, cs, aerr)) {
                    err = "verify: tp publish at layer " + std::to_string(l) + ": " + aerr;
                    return false;
                }
            }
            return true;
        };
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN (split: HV/2 v-heads per card) =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = gdn_state_tp_ + (size_t) gi * gdn_state_tp_floats_;
                float* conv = state + (size_t) S * HVt * S;
                float* cwt = conv_w_tp_ + (size_t) gi * Ct * g.ssm_d_conv;   // 4 taps
                float* qkv = qkv_tp_ + (size_t) gi * MT * Ct;
                float* hb = h_tp_ + (size_t) gi * MT * Ct;
                float* gate = gate_tp_ + (size_t) gi * MT * HVt;
                float* beta = beta_tp_ + (size_t) gi * MT * HVt;
                if (!q8_attn) quant_in(xm, (int) N);
                // qkv: the local 7168 of 10240 rows (q 2048 + k 2048 + this card's v 3072)
                mm(wqkv, qkv, (int) N, (int) Ct);
                stamp(l, 2);
                if (m_tpdump_ != nullptr && l == 0) {   // DIAG: the GDN qkv rows and the conv output, vs the solo path
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 548), qkv, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 552), qkv + 2048, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 556), qkv + 4096, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                }
                gdn_conv_l2_multi(conv, qkv, cwt, hb, (int) Ct, (int) (2 * HK), EPS, T, cs, 0, self_commit);
                stamp(l, 3);
                if (m_tpdump_ != nullptr && l == 0)   // DIAG: the conv/L2 output (q and k rows, then this card's v)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 560), hb, 4 * sizeof(float),
                                    cudaMemcpyDeviceToHost, cs);
                // the gates over this card's HV/2 v-heads (the global head's weights/dt/ssm_a)
                gdn_ab_multi_tp(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                                (const float*) wsa->data, gate, beta, (int) N, (int) HVt, T, cs, head_off);
                stamp(l, 4);
                mm(wg, z_tp_, (int) N, (int) ZVt);
                stamp(l, 5);
                gdn_step_norm_multi_tp(state, hb, (int) Ct, gate, beta, z_tp_, (const float*) wnm->data, EPS, y_tp_,
                                       (int) HK, (int) HVt, T, nullptr, cs, 0, nullptr, head_off);
                stamp(l, 6);
                if (gdn_full) {
                    // rank 0 computes the WHOLE GDN out-projection: its own value heads in y_full_[0..ZVt),
                    // rank 1's half copied over P2P into y_full_[ZVt..ZV).  Rank 0's out-projection is then
                    // bit-exact with the single-card path; rank 1 contributes zero to the allreduce.
                    // PITCH: y_tp_ is [T][ZVt], y_full_ is [T][ZV]: per token, as in the column-parallel branch.
                    if (tp_rank_ == 0) {
                        for (int t = 0; t < T; ++t)
                            if (cudaMemcpyAsync(y_full_ + (size_t) t * ZV, y_tp_ + (size_t) t * ZVt,
                                                (size_t) ZVt * sizeof(float), cudaMemcpyDeviceToDevice,
                                                cs) != cudaSuccess) {
                                err = "verify: tp: the GDN value copy failed";
                                return false;
                            }
                        // rank 1's half must be in y_full_[ZVt..ZV) before the projection reads it
                        wait_flag_ge(tp_flag_g_d_, (uint32_t) (l + 1), cs);
                        native_quantize_q8_1(y_full_, xq_, (int) ZV, T, cs);
                        il_ready = false;
                        mm(wout, bo_, (int) ZV, (int) N);
                        // ONE flag per direction (see the column-parallel branch): rank 1 spins on c
                        tp_raise_flag(tp_flag_c_d_, (uint32_t) (l + 1), cs);
                    } else {
                        std::string aerr;
                        // rank 1's half into RANK 0's y_full_[ZVt..ZV) (the P2P pointer set at setup), then tell
                        // rank 0 it is there.  Rank 1 contributes zero to the allreduce here, so its own
                        // y_full_ is never read and needs no local copy.
                        for (int t = 0; t < T; ++t)
                            if (!tp_ar_.broadcast_to_a(y_tp_ + (size_t) t * ZVt, y_peer_ + (size_t) t * ZV + ZVt,
                                                       (int64_t) ZVt, cs, aerr)) {
                                err = "verify: tp GDN half at layer " + std::to_string(l) + ": " + aerr;
                                return false;
                            }
                        tp_raise_flag(tp_flag_g_d_, (uint32_t) (l + 1), cs);
                        cudaMemsetAsync(bo_, 0, (size_t) T * N * sizeof(float), cs);
                    }
                } else {
                    // COLUMN-parallel out-projection: each card holds half the OUTPUT ROWS and reduces the
                    // FULL ZV value vector, so every output element is bit-identical with the single card's
                    // (tests/core/tp_outproj_split_test.cu asserts exactly this on the real kernels).  The old
                    // row-parallel split summed half the reduction per card and let the allreduce add the two
                    // partials, which is not the single card's sum in float32.
                    //
                    // The full ZV vector needs the peer's value half, and the exchange is TWO-WAY: rank 1 sends
                    // its half up (rank 0's [ZVt, ZV)) and rank 0 sends its half down (rank 1's [0, ZVt)).
                    //
                    // ONE flag per direction, never one shared flag with two counting sequences.  raise_flag is
                    // a plain store (`*flag = value`), so two writers on one word overwrite each other: rank 0
                    // would publish l+1+n_layers while rank 1 published l+1, and whichever landed second ROLLED
                    // THE COUNTER BACK, leaving the other rank waiting on a value that never came (observed as
                    // "tp stall at layer 1: seq=1 want=2").  rank 1 -> rank 0 uses g, rank 0 -> rank 1 uses c.
                    //
                    // PITCH: y_tp_ is [T][ZVt] and y_full_ is [T][ZV].  A single flat copy of T*ZVt floats would
                    // put token t's half at t*ZVt instead of t*ZV, so token 0 alone would look right and every
                    // later token in the window would read its neighbour's values.  Both transfers are per token.
                    std::string aerr;
                    if (tp_rank_ == 0) {
                        for (int t = 0; t < T; ++t) {
                            if (cudaMemcpyAsync(y_full_ + (size_t) t * ZV, y_tp_ + (size_t) t * ZVt,
                                                (size_t) ZVt * sizeof(float), cudaMemcpyDeviceToDevice,
                                                cs) != cudaSuccess) {
                                err = "verify: tp: the GDN value copy failed";
                                return false;
                            }
                        }
                        wait_flag_ge(tp_flag_g_d_, (uint32_t) (l + 1), cs);   // rank 1's half is in y_full_[ZVt, ZV)
                        // rank 0's half into RANK 1's y_full_[0, ZVt); rank 1 spins on flag c
                        for (int t = 0; t < T; ++t)
                            if (!tp_ar_.broadcast(y_tp_ + (size_t) t * ZVt, y_peer_ + (size_t) t * ZV,
                                                  (int64_t) ZVt, cs, aerr)) {
                                err = "verify: tp GDN half at layer " + std::to_string(l) + ": " + aerr;
                                return false;
                            }
                        tp_raise_flag(tp_flag_c_d_, (uint32_t) (l + 1), cs);
                        quant_in(y_full_, (int) ZV);
                        // COLUMN-parallel output: the kernel writes a COMPACT [T][N/2] (y[j*n_out + row]), so
                        // it goes to the scratch and each token is pitched into bo_'s [T][N] layout below.
                        mm(wout, half_out_, (int) ZV, (int) (N / 2));
                        for (int t = 0; t < T; ++t) {
                            cudaMemcpyAsync(bo_ + (size_t) t * N, half_out_ + (size_t) t * (N / 2),
                                            (size_t) (N / 2) * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                            cudaMemsetAsync(bo_ + (size_t) t * N + N / 2, 0,
                                            (size_t) (N / 2) * sizeof(float), cs);
                        }
                    } else {
                        // rank 1's half into RANK 0's y_full_[ZVt, ZV) (the P2P pointer set at setup)
                        for (int t = 0; t < T; ++t)
                            if (!tp_ar_.broadcast_to_a(y_tp_ + (size_t) t * ZVt, y_peer_ + (size_t) t * ZV + ZVt,
                                                       (int64_t) ZVt, cs, aerr)) {
                                err = "verify: tp GDN half at layer " + std::to_string(l) + ": " + aerr;
                                return false;
                            }
                        // ...AND into our OWN y_full_[ZVt, ZV): the exchange fills rank 0's second half from
                        // here and OUR first half from rank 0, but nobody else can write our second half.
                        // Without this the quantizer below reads last layer's stale values for our own 3072
                        // v channels, so our 1280 output rows are wrong and the allreduce's sum is wrong in
                        // every element (observed: l0 mixer output differing on 2560/2560, first token 420).
                        for (int t = 0; t < T; ++t)
                            if (cudaMemcpyAsync(y_full_ + (size_t) t * ZV + ZVt, y_tp_ + (size_t) t * ZVt,
                                                (size_t) ZVt * sizeof(float), cudaMemcpyDeviceToDevice,
                                                cs) != cudaSuccess) {
                                err = "verify: tp: the GDN value copy failed";
                                return false;
                            }
                        tp_raise_flag(tp_flag_g_d_, (uint32_t) (l + 1), cs);
                        // rank 0's half arrives in OUR y_full_[0, ZVt) over P2P
                        wait_flag_ge(tp_flag_c_d_, (uint32_t) (l + 1), cs);
                        quant_in(y_full_, (int) ZV);
                        mm(wout, half_out_, (int) ZV, (int) (N / 2));
                        for (int t = 0; t < T; ++t) {
                            cudaMemsetAsync(bo_ + (size_t) t * N, 0, (size_t) (N / 2) * sizeof(float), cs);
                            cudaMemcpyAsync(bo_ + (size_t) t * N + N / 2, half_out_ + (size_t) t * (N / 2),
                                            (size_t) (N / 2) * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                        }
                    }
                }
                stamp(l, 7);
                // DIAG (STRATA_GDN_VALDUMP=1): the GDN VALUE VECTORS, per rank, at layer 0 (first window).  The
                // column-parallel out-projection is bit-exact ONLY if the assembled full y_full_ is byte-identical
                // on both ranks; that requires the two P2P copies to land clean:
                //   y_full_[0,ZVt)    = rank0's local value half   (rank1's via tp_ar_.broadcast,     rank0 -> rank1)
                //   y_full_[ZVt,ZV)   = rank1's local value half   (rank0's via tp_ar_.broadcast_to_a, rank1 -> rank0)
                // so rank0.y_full_ and rank1.y_full_ must match byte-for-byte:
                //   [0,ZVt)  byte-equal  => the rank0->rank1 broadcast is a clean copy
                //   [ZVt,ZV) byte-equal  => the rank1->rank0 broadcast_to_a is a clean copy AND rank1's local
                //                           gdn_step_norm_multi_tp(head_off=HVt) half equals rank0's.
                // rank1's own y_tp_ (its local half) is dumped too, to tell "local half wrong" from "P2P copy wrong".
                // STRATA_GDN_VALDUMP: the GDN value vectors, per rank, at layer 0 (first window).  y_full_/y_tp_ are
                // DEVICE pointers (the take() arena is cudaMalloc'd), and this runs INSIDE the graph capture, so the
                // only legal move is to RECORD a cudaMemcpyAsync (device->host) into the PERSISTENT pinned staging
                // allocated in tp_setup_scratch (a cudaHostAlloc is NOT a legal capture op - it aborts the capture
                // and poisons the stream).  The graph replays those copies and run_tp's window-end writer (outside
                // capture) then fwrites the landed buffers - the exact STRATA_MOE_HQ_DUMP / STRATA_TP_VEC pattern.
                if (std::getenv("STRATA_GDN_VALDUMP") != nullptr && l == 0 && tp_rank_ >= 0 && tp_rank_ < 2) {
                    static bool done_[2] = {false, false};
                    if (!done_[tp_rank_] && g_gdnval_full_h_[tp_rank_] != nullptr && g_gdnval_half_h_[tp_rank_] != nullptr) {
                        g_gdnval_T_ = T; g_gdnval_ZV_ = (int) ZV; g_gdnval_ZVt_ = (int) ZVt;
                        for (int t = 0; t < T; ++t) {
                            cudaMemcpyAsync(g_gdnval_full_h_[tp_rank_] + (size_t) t * ZV, y_full_ + (size_t) t * ZV,
                                            (size_t) ZV * sizeof(float), cudaMemcpyDeviceToHost, cs);
                            cudaMemcpyAsync(g_gdnval_half_h_[tp_rank_] + (size_t) t * ZVt, y_tp_ + (size_t) t * ZVt,
                                            (size_t) ZVt * sizeof(float), cudaMemcpyDeviceToHost, cs);
                        }
                        g_gdnval_fired_[tp_rank_] = 1;
                    }
                    done_[tp_rank_] = true;
                }
                // DIAG (STRATA_MOE_DN_D=1): the attention's output, per rank, AFTER the column-split out-projection
                // allreduce (full width on both ranks).  The L0 site1 delta = this + the MoE delta + shared, so the
                // [1280,2560) band's 1e-8 is split attention-vs-MoE by whether this dump differs there.
                if (std::getenv("STRATA_MOE_DN_D") != nullptr && l == 0) {
                    static bool done = false;
                    if (!done) {
                        const char* fn = std::getenv("STRATA_MOE_DN_D_FILE");
                        if (fn != nullptr) {
                            const std::string path = std::string(fn) + ".rank" + std::to_string(tp_rank_);
                            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                                for (int t = 0; t < 2; ++t)
                                    std::fwrite(bo_ + (size_t) t * N, sizeof(float), (size_t) N, f);
                                std::fclose(f);
                            }
                            done = true;
                        }
                    }
                }
                if (!ring(l)) return false;
            } else {
                // ======================= QSA (replicated attention, split out-projection) =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope_on = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) sh.n_rot, rope_scaling(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) sh.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                const bool fuse_nr = native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable((int) HD, (int) sh.n_rot);
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) sh.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) sh.n_rot, EPS,
                                                 rope_scaling(), pos, cs);
                        return;
                    }
                    norm_rope_on(data, norm, rows, cols, pos);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                static const bool qb_q4 = [] {
                    const char* e = std::getenv("STRATA_DF_QB_Q4");
                    return e == nullptr || std::atoi(e) != 0;
                }();
                const bool qb = dec_batch && T > 1 && native_qsa_enabled() && native_rope_enabled() && (!st.kv_q4 || qb_q4);
                if (!q8_attn) quant_in(xm, (int) N);
                // all of the mixer's projections: replicated (both cards hold full K/V and the full cache)
                if (qb) bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID, T, cs);
                else for (int t = 0; t < te; ++t) bf16_gemv_fp32_mmvf(xm + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                // k/v: the replicated full projections.  The grouped launch only runs under STRATA_MMVQ_V2, exactly
                // as the single-card path; without it the call returns false and the buffers stay unwritten, so the
                // plain per-matrix mmvqs (bitwise the grouped one's) must run instead
                bool q_done = false;
                if (native_mmvq_v2_enabled() && wk->native_type == 8 && wv->native_type == 8 && wq->native_type == 8)
                    q_done = native_q8_0_mmvq_group(3, std::array<const void*, 3>{wk->native_data, wv->native_data, wq->native_data}.data(),
                                       std::array<float*, 3>{kcur_, vcur_, qfull_}.data(),
                                       std::array<int, 3>{(int) (NKV * HD), (int) (NKV * HD), (int) (NH * 2 * HD)}.data(),
                                       xq_, (int) N, T, cs);
                if (!q_done) {
                    mm(wk, kcur_, (int) N, (int) (NKV * HD));
                    mm(wv, vcur_, (int) N, (int) (NKV * HD));
                }
                if (m_tpdump_ != nullptr && l == 3) {   // DIAG: the k/v weight heads and the raw (pre-norm) k/v
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 528), wk->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 532), wv->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 536), wq->native_data, 16, cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 540), kcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 544), vcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                }
                norm_rope(kcur_, wkn, (int) (T * NKV), (int) HD, pos_k);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp), exactly as the solo path
                    fwht256_inplace_cuda(kcur_, (int64_t) T * NKV, cs);
                    fwht256_inplace_cuda(vcur_, (int64_t) T * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_, (int64_t) T * NKV, cs);
                }
                if (m_tpdump_ != nullptr && l == 3) {   // DIAG: the l3 intermediates, vs the solo path
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 492), kcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 496), vcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                }
                {
                    const int32_t* step_b = step_ + kStepPos;
                    if (dec_batch && !g_no_batch_kv) {
                        if (st.kv_hybrid) {
                            const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                            const bool mirror = st.host.present();
                            kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b, kStepCount,
                                               kcur_, kcur_, (int) (NKV * HD), T, sh, cs, mirror ? &hk : nullptr);
                            kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, kStepCount, T, vcur_, vcur_, sh, cs,
                                               mirror ? &hv : nullptr);
                        } else if (st.kv_q4)
                            kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, kStepCount, T, kcur_, vcur_, sh, cs, &st.host);
                        else if (st.kv_int8)
                            kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b, kStepCount,
                                               kcur_, vcur_, (int) (NKV * HD), T, sh, cs, &st.host);
                        else
                            for (int t = 0; t < te; ++t)
                                kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount,
                                               kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, sh, cs, &st.host);
                    } else
                        for (int t = 0; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, sh, cs, &st.host);
                }
                {
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    if (dec_batch && !g_no_batch_kv) {
                        native_qsa_indexer_append_steps(idx_raw, step_ + kStepPos, kStepCount, T, 0, (const float*) wikn->data,
                                                        EPS, ib, sh, st.max_cells, rope_scaling(), cs);
                    } else {
                        for (int t = 0; t < te; ++t)
                            native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                      (const float*) wikn->data, EPS, ib, sh, st.max_cells, rope_scaling(), cs);
                    }
                }
                stamp(l, 9);
                // the queries: the q|gate projection, then the q/gate split with norm and RoPE
                mm(wq, qfull_, (int) N, (int) (NH * 2 * HD));
                if (qb) {
                    if (fuse_nr) {
                        native_qsa_rms_norm_rope(qfull_, (int) (2 * HD), (const float*) wqn->data, qcur_, (int) (T * NH), (int) HD,
                                                 (int) sh.n_rot, EPS, rope_scaling(), pos_, cs);
                    } else {
                        copy_rows_strided(qcur_, qfull_, (int64_t) T * NH, HD, 2 * HD, cs);
                        norm_rope(qcur_, wqn, (int) (T * NH), (int) HD, pos_);
                    }
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
                } else {
                    for (int t = 0; t < te; ++t) {
                        float* qc = qcur_ + t * NH * HD;
                        if (fuse_nr) {
                            native_qsa_rms_norm_rope(qfull_ + t * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data, qc,
                                                     (int) NH, (int) HD, (int) sh.n_rot, EPS, rope_scaling(), pos_ + t * NH, cs);
                        } else {
                            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                                  (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                                err = "verify: tp: the q/gate split failed";
                                return false;
                            }
                            norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                        }
                        if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);
                    }
                }
                if (m_tpdump_ != nullptr && l == 3)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 500), qcur_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                // the indexer's query, norm and RoPE, then the block selection
                if (qb) {
                    bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) wiq->data, qidx_, IQ * ID, N, IQ * ID, T, cs);
                    norm_rope(qidx_, wiqn, (int) (T * IQ), (int) ID, pos_i);
                } else
                    for (int t = 0; t < te; ++t) {
                        float* qx = qidx_ + t * IQ * ID;
                        bf16_gemv_fp32_mmvf(xm + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                        // the indexer's own position row: `pos_i` holds IQ entries per token (`pos_` is
                        // MT*(NH+NKV+IQ), the q heads' rows first).  `pos_ + t*NH` is the q-head row and is only
                        // right while NH == IQ by accident; with NH 24 and IQ 16 it read past the row.
                        norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_i + t * IQ);
                    }
                if (m_tpdump_ != nullptr && l == 3)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 504), qidx_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_, step_, T, max_blocks_, sh, scores_, cs);
                qsa_block_topk(scores_, step_, T, max_blocks_, cap_, sh, sel_, cs);
                if (m_tpdump_ != nullptr && l == 3) {
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 508), scores_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 512), sel_, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost, cs);
                }
                stamp(l, 11);
                qsa_kv_resolve(st, g, sel_, step_, T, cap_, cs);
                {
                    const QsaAttnPools px = qsa_attn_pools(st);
                    qsa_decode_attn_batch(qcur_, px, sel_, step_, cap_, sh, attn_scratch_, attn_, T, cs);
                }
                stamp(l, 13);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
                if (m_tpdump_ != nullptr && l == 3)
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 516), attn_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                if (qb) native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
                else
                    for (int t = 0; t < te; ++t) {
                        if (native_qsa_enabled()) native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD,
                                                                       attn32_ + t * NH * HD, (int) NH, (int) HD, cs);
                        else qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, sh, attn32_ + t * NH * HD, cs);
                    }
                stamp(l, 14);
                // The out-projection is COLUMN-parallel: each card holds half the OUTPUT ROWS and reduces the
                // full NH*HD attention vector.  attn32_ already holds all NH heads on BOTH cards (the attention
                // runs full width), so this needs no exchange at all - unlike the GDN side, whose value vector
                // is head-split.  Every output element is bit-identical with the single card's
                // (tests/core/tp_outproj_split_test.cu); the two cards' rows are disjoint.
                native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
                il_ready = false;
                if (attn_full) {
                    // STRATA_TP_ATTN_FULL=1 (diagnostic): rank 0 computes ALL N rows, rank 1 contributes zero.
                    // Full width means n_out == N, so the kernel's own pitch already matches bo_ - no staging.
                    if (tp_rank_ == 0) mm(wo, bo_, (int) (NH * HD), (int) N);
                    else cudaMemsetAsync(bo_, 0, (size_t) T * N * sizeof(float), cs);
                } else {
                    // this card's own half of the output rows; the other half stays zero for the allreduce.
                    // The kernel writes a COMPACT [T][N/2] (y[j*n_out + row]), so each token is pitched into
                    // bo_'s [T][N] layout - a single flat write would shift every token after the first.
                    mm(wo, half_out_, (int) (NH * HD), (int) (N / 2));
                    for (int t = 0; t < T; ++t) {
                        if (tp_rank_ == 0) {
                            cudaMemcpyAsync(bo_ + (size_t) t * N, half_out_ + (size_t) t * (N / 2),
                                            (size_t) (N / 2) * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                            cudaMemsetAsync(bo_ + (size_t) t * N + N / 2, 0, (size_t) (N / 2) * sizeof(float), cs);
                        } else {
                            cudaMemsetAsync(bo_ + (size_t) t * N, 0, (size_t) (N / 2) * sizeof(float), cs);
                            cudaMemcpyAsync(bo_ + (size_t) t * N + N / 2, half_out_ + (size_t) t * (N / 2),
                                            (size_t) (N / 2) * sizeof(float), cudaMemcpyDeviceToDevice, cs);
                        }
                    }
                }
                if (m_tpdump_ != nullptr && l == 3)   // DIAG: each rank's pre-allreduce l3 partial
                    cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 520), bo_, 4 * sizeof(float), cudaMemcpyDeviceToHost, cs);
                stamp(l, 15);
                if (!ring(l)) return false;
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16);
        const bool q8_ffn = gr_read_group(1, true, inj_, inj2_);
        stamp(l, 32);
        if (m_tpdump_ != nullptr && l < 2)   // DIAG: the REAL MoE input (half 1, after the FFN-side read)
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 100) + (size_t) l * 4, mixed_, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (m_tpdump_ != nullptr && l == 3)   // DIAG: the l3 MoE input, vs the solo path
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 524), mixed_, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        // DIAG (STRATA_TP_VEC=N): site 0, this layer's mixer output in full (rank 0's block; the two cards read
        // the same residual, so one copy is the pair's)
        if (const int64_t vat = tp_vec_at(l, 0); vat >= 0 && tp_rank_ == 0)
            cudaMemcpyAsync(m_tpdump_ + vat, mixed_, (size_t) tp_vec_len() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        // ---- the router (replicated: the full residual is identical on both cards)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        if (dec_batch && T > 1 && w_router != nullptr && native_router_enabled() && NE == 512 && K == 10) {
            try {
                bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) w_router->data, logits_, NE, N, NE, T, cs);
                native_router_top10_multi(logits_, ids_, w_, T, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
            for (int t = 0; t < te; ++t) {
                MoEBuffers mb = ss.moe;
                mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
                if (!moe_route(wt, g, l, K, mb, xm + t * N, cs, err, nullptr)) return false;
            }
        // ---- the doorbell.  ONE doorbell for the pair: rank 0's.  Rank 1 shares rank 0's mapped sequence and
        // plan words (tp_share_doorbell), so if BOTH cards published, the sequence would advance twice per layer.
        if (ar_on()) {
            resident_plan(ids_, (int) (T * K), (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob, plan_, (long long) max_t_ * K, nullptr, 0,
                          cs, m_plan_err_);
        } else if (tp_rank() == 0) {
            if (device_plan_)
                resident_plan(ids_, (int) (T * K), (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob, plan_, (long long) max_t_ * K, skip_,
                              (uint32_t) ((l - lb_) * G + 1), cs);
            if (remote_opt_ || always_publish_)
                doorbell_publish(xm, ids_, w_, (int64_t) T * N, (int64_t) T * K, m_x_, m_ids_, m_w_, m_seq_, cs);
            else {
                const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
                doorbell_publish_res(xm, ids_, layer_res, (int) g.n_expert, (int64_t) T * N, (int64_t) T * K, m_x_, m_ids_,
                                     m_seq_, cs);
            }
        }
        stamp(l, 17);
        if (m_tpdump_ != nullptr && l < 2) {   // DIAG: the routing decision, before anything else touches it
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 72) + (size_t) l * 8, ids_, 4 * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, cs);
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 72) + (size_t) l * 8 + 4, w_, 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        }
        // ---- the shared expert (replicated, full width)
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = xq_;
            try {
                if (dec_batch) f32_to_bf16_bulk(xm, sh_bf16_, (int64_t) T * N, cs);
                else for (int t = 0; t < te; ++t) f32_to_bf16_bulk(xm + t * N, sh_bf16_ + t * N, N, cs);
                shared_expert_multi((int) T, xm, sh_bf16_, nsw, (const uint16_t*) wgi->data, sh_gate_, sh_up_, sh_g_,
                                    shared_, N, g.n_ff, cs, nullptr, 0);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
        }
        if (strata::kernels::cpu::expert_layout().native) {
            if (!q8_ffn) quantize_q8_1_rows(xm, (int) T, (int) N, nat_xq_, cs);
            // STRATA_POOL_ORDER=1: the pool's own activation contract alongside the q8_1 image (see the solo
            // path's note).  The TP split's q8_0 calls read this one.
            if (g_pool_order()) quantize_q8_0(xm, hit_xq_, (int64_t) T * N, cs);
        } else
            quantize_q8_0_scaled(xm, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        stamp(l, 18);
        return true;
    };

    // ---------------------------------------------------------------- post(l): experts, combine
    auto post = [&](int64_t l) -> bool {
        const int64_t cap = (int64_t) T * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_;
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_;
        float* parts_out = parts_;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // ---- the experts.  BOTH cards compute a real half of every routed expert: the intermediate width is
        // cut in two (tp_moe_), so rank r runs gate/up rows [r*n_ff/2, (r+1)*n_ff/2) and down's columns of the
        // same range, and the two partial residuals add to the single-card one.  The residency table, the plan
        // and the expert arena are rank 0's (P2P makes them readable from rank 1); the CPU rows arrive in the
        // mapped block only on rank 0.
        if (!tp_moe_ok_) {
            err = "verify: tp: the experts' FFN split is not set up (tp_setup_scratch)";
            return false;
        }
        const NativeExpertLayout Lgu = tp_moe_.layout;
        const NativeExpertLayout Ldown = tp_moe_.down;
        const unsigned long long grp_shift = (unsigned long long) tp_moe_.ptr_off;
        const uint32_t want = (uint32_t) ((l - lb_) * G + 1);
        {
            const bool use_dev = tp_rank() == 0 && device_plan_;
            if (use_dev) {
                wait_flag_ge_or(m_flagA_, want, skip_, cs);
                copy_i32_from_mapped_unless(pl, m_plan_, plan_i32_, skip_, want, cs);
            } else {
                wait_flag_ge(m_flagA_, want, cs);
                copy_i32_from_mapped(pl, m_plan_, plan_i32_, cs);
            }
        }
        // This rank's group pointers: the plan's own arrays shifted into private copies (the plan block is
        // shared with the peer and rank 0 rewrites `p_ptr2` in place, so the shift cannot happen there).  The
        // GATE/UP call uses these shifted pointers (the row offset is a group-pointer shift); the DOWN call
        // uses the UNshifted plan pointers - its row offset lives in Ldown.down_off instead (half the output
        // rows), and s_down is read from the unshifted group pointer's blob tail.
        auto ensure_ptrs = [&]() -> bool {
            if (grp_shift == 0) return true;
            std::string serr;
            return tp_shift_group_ptrs(p_ptr, tp_ptr_, capx, grp_shift, cs, serr);
        };
        auto ensure_ptrs2 = [&]() -> bool {
            if (grp_shift == 0) return true;
            std::string serr;
            return tp_shift_group_ptrs(p_ptr2, tp_ptr2_, capx, grp_shift, cs, serr);
        };
        // The split experts run in TWO PHASES (the down needs the FULL intermediate the gate/up only half-
        // produced).  `gu_split` runs the gate/up rows at the local width into `scratch` (its hq region holds the
        // local hq the exchange reads); `down_split` runs the down rows (n_embd/2, full n_ff reduction) reading
        // the assembled full hq from the SAME scratch's hq region, into the compact [cap][n_embd/2] output that
        // `moe_hit_pitch` then pitches into hit_out's own half.  The phases are the native launcher's g_exp_phase
        // (host, read at launch decision): 1 = gate/up + SwiGLU + q8_1 only, 2 = down only.  Set on the host here
        // (before the launch) so the captured graph bakes in exactly those kernel sequences; record_window_tp
        // restores phase 0 after the layer loop so the solo / other grouped calls see the full pipeline.
        auto gu_split = [&](const unsigned long long* gp, const int32_t* gs2, const int32_t* gn, int64_t gy,
                            void* scratch) {
            if (lay.native) {
                if (g_pool_order()) {
                    // The pool-order split: gate/up at the LOCAL width (Lgu.n_ff = n_ff/2) from the q8_0
                    // activation, then the intermediate quantized into THIS scratch's hq region - which is what
                    // `exchange_hq` gathers and the down reads.  `native_expert_grouped_q8_0` does the quantize
                    // inside phase 1, exactly where the q8_1 path's `swiglu_q8_1_entries_kernel` sat.
                    native_expert_grouped_q8_0(Lgu, gp, gs2, gn, p_dst, p_tok, cap, cap, hit_xq_,
                                               (int) ((g.n_embd / 32) * 34), scratch, hit_out, cs, 1, gy);
                } else {
                    native_expert_set_mode(-1, 1);
                    native_expert_grouped(Lgu, gp, gs2, gn, p_dst, p_tok, cap, cap, nat_xq_, scratch, hit_out, cs, gy);
                }
            } else
                moe_grouped_s2(gp, gs2, gn, p_dst, p_tok, cap, cap, hit_xq_, hit_xs_, hit_scratch_, hit_out, cs);
        };
        auto down_split = [&](const unsigned long long* gpd, const int32_t* gs2, const int32_t* gn, int64_t gy,
                              void* scratch) {
            if (!lay.native) return;   // S2 has no split down; the default path is native on this card
            if (g_pool_order()) {
                // The pool-order down: half the output rows (Ldown), the FULL n_ff reduction over the ASSEMBLED
                // hq, one Q8_0 row per entry (the kernel maps entry e to activation row e itself).  The result is
                // the compact [cap][n_embd/2] band the pitch then places in hit_out's own half.
                native_expert_grouped_q8_0(Ldown, gpd, gs2, gn, p_dst, nullptr, cap, cap, nullptr, 0, scratch,
                                           tp_moe_hitc_, cs, 2, gy);
            } else {
                native_expert_set_mode(-1, 2);
                native_expert_grouped(Ldown, gpd, gs2, gn, p_dst, p_tok, cap, cap, nat_xq_, scratch, tp_moe_hitc_, cs, gy);
            }
            moe_hit_pitch(tp_moe_hitc_, hit_out, p_dst, cap, (int) g.n_embd, (int) (g.n_embd / 2),
                          (int) (tp_rank() * (g.n_embd / 2)), cs);
        };
        // The exchange (copy only): assemble the full hq (this rank's local hq in `scratch` + the peer's over
        // P2P) into `full_hq`.  `peer_base` is the peer's gate/up scratch BASE (fixed, set at setup); the
        // per-window offset (cap = T*K) is added here because the local hq region sits at hq_off(cap, n_ff_local)
        // inside that scratch - both ranks run the same window size, so both compute the same offset.  The
        // handshake (wait the peer's flag, raise own) is done in the rank sections: ONE flag per direction,
        // raised after BOTH group sets' gu, so set0's and set1's exchanges both sit after a single barrier.
        auto exchange_hq = [&](void* scratch, void* full_hq, const uint8_t* peer_base) {
            const uint8_t* local_hq = (const uint8_t*) scratch +
                                      strata::kernels::native_expert_hq_off(cap, tp_moe_.n_ff_local);
            const uint8_t* full_off = (const uint8_t*) full_hq +
                                      strata::kernels::native_expert_hq_off(cap, tp_moe_.layout.n_ff * 2);
            const uint8_t* peer_hq = peer_base + strata::kernels::native_expert_hq_off(cap, tp_moe_.n_ff_local);
            const int half_blocks = (int) (tp_moe_.n_ff_local / 32);
            const int own = (int) (tp_rank() * half_blocks);
            // pool order: the intermediate is `block_q8_0` (34 bytes), so the exchange must move 34-byte blocks -
            // the kernel is byte-granular for exactly this (see moe_hq_exchange_kernel).
            moe_hq_exchange(local_hq, peer_hq, (uint8_t*) full_off, cap, half_blocks, own, cs,
                            g_pool_order() ? 34 : 36);
        };
        static const bool moe_full = [] {   // DIAG: rank 0 computes the experts at FULL width, rank 1 contributes
            // nothing: the allreduce then carries rank 0's single-card-math result, isolating the residency
            // partition from the split itself.
            const char* e = std::getenv("STRATA_TP_MOE_FULL");
            return e != nullptr && std::atoi(e) != 0;
        }();
        // STRATA_MOE_HQ_DUMP=1: capture-safe, one-shot (per process, per rank) layer-0 set0 hq dump.  The two
        // D2H copies are GRAPH NODES into this rank's pinned staging (no host sync, no fopen here - record_window_tp
        // runs inside cudaStreamBeginCapture, where a host sync aborts the capture and a host fwrite would write
        // the unexecuted async copy).  The front is the ASSEMBLED full hq (this rank's view, read back through
        // the same P2P path the down uses), the back this rank's LOCAL gate/up hq; taken right after the two
        // exchange_hq calls so the assembled image is fully written and before the down reads it.  run_tp writes
        // both ranks' buffers to disk after the window join (the buffers outlive the layer - they are pinned,
        // not scratch the next layer reuses).
        auto moe_hq_dump_on = [&]() -> bool {
            if (m_moe_hq_diag_ == nullptr) return false;
            const char* ge = std::getenv("STRATA_MOE_HQ_DUMP");
            if (ge == nullptr || std::atoi(ge) == 0) return false;
            if (moe_hq_dump_cap_ < 0) moe_hq_dump_cap_ = cap;   // the window's entry count (run_tp sizes the files)
            return true;
        };
        // set 0 = tp_moe_hq_/tp_moe_scratch_ (the VRAM share), set 2 = tp_moe_hq2_/tp_moe_scratch2_ (the PCIe share).
        // Staging is laid out at the capx (max window) size; the runtime copies only `cap` bytes into each slot, so
        // the SET OFFSETS must be the capx region sizes (moe_hq_diag_hq_bytes_ and the capx local-hq size) - using
        // the runtime `cap` there would misplace set2 and hit_out.  Layout: [set0 hq][set0 lq][set2 hq][set2 lq][hit_out].
        const int64_t moe_blocks_full = (int64_t) (tp_moe_.layout.n_ff * 2) / 32;    // 20
        const int64_t moe_blocks_local = (int64_t) tp_moe_.n_ff_local / 32;         // 10
        const size_t moe_lq_region = (size_t) capx * (size_t) moe_blocks_local * 36;   // capx-sized per-set local hq
        const size_t moe_set_region = (size_t) moe_hq_diag_hq_bytes_ + moe_lq_region;  // one set = hq slot + lq slot
        auto dump_moe_hq_diag = [&](int set) {
            if (!moe_hq_dump_on()) return;
            const size_t hq_sz = (size_t) cap * (size_t) moe_blocks_full * 36;   // runtime: only `cap` entries
            const size_t lq_sz = (size_t) cap * (size_t) moe_blocks_local * 36;
            const void* hq_src = (set == 0 ? tp_moe_hq_ : tp_moe_hq2_);
            const void* lq_src = (set == 0 ? tp_moe_scratch_ : tp_moe_scratch2_);
            uint8_t* dst = h_moe_hq_diag_ + (size_t) (set / 2) * moe_set_region;  // set 0 -> 0, set 2 -> 1*region
            if (cudaMemcpyAsync(dst, (const uint8_t*) hq_src +
                                         strata::kernels::native_expert_hq_off(cap, (int64_t) tp_moe_.layout.n_ff * 2),
                            hq_sz, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata verify: tp: the hq dump's set%d assembled D2H failed\n", set);
                return;
            }
            if (cudaMemcpyAsync(dst + (size_t) moe_hq_diag_hq_bytes_, (const uint8_t*) lq_src +
                                          strata::kernels::native_expert_hq_off(cap, tp_moe_.n_ff_local),
                            lq_sz, cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata verify: tp: the hq dump's set%d local D2H failed\n", set);
            }
        };
        // After BOTH down_split calls: the pitched MoE band this rank owns (rank 0: cols [0, N/2), rank 1: the rest).
        auto dump_moe_hitout = [&]() {
            if (!moe_hq_dump_on()) return;
            const size_t off = 2 * moe_set_region + (size_t) l * moe_hq_dump_hitc_sz_;   // per-layer slot
            if (cudaMemcpyAsync(h_moe_hq_diag_ + off, hit_out_, (size_t) cap * (size_t) g.n_embd * sizeof(float),
                            cudaMemcpyDeviceToHost, cs) != cudaSuccess) {
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata verify: tp: the hq dump's hit_out D2H failed\n");
            }
        };
        auto grouped_full = [&](const unsigned long long* gp, const int32_t* gs2, const int32_t* gn, int64_t gy) {
            const auto& f = lay.fmt[(size_t) l];
            const NativeExpertLayout Lfull = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            native_expert_grouped(Lfull, gp, gs2, gn, p_dst, p_tok, cap, cap, nat_xq_, hit_scratch_, hit_out, cs, gy);
        };
        stamp(l, 19);
        if (m_tpdump_ != nullptr && l < 2 && tp_rank() == 0)   // DIAG: the plan's group counts (hits, CPU, PCIe)
            cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 88) + (size_t) l * 4, p_counts, 3 * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, cs);
        if (tp_rank() == 0) {
            if (moe_full) {
                grouped_full(p_ptr, p_start, p_counts, 0);
                stamp(l, 20);
                if (device_plan_) wait_flag_ge_or(m_flagB_, want, skip_, cs);
                else wait_flag_ge(m_flagB_, want, cs);
                if (sink_.pcie_mode == 2) {
                    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                    uint8_t* stage = staging_ + (size_t) (0 * per) * lay.max_blob;
                    const unsigned long long* p_dst2 = (const unsigned long long*) (pl + dst2_off_);
                    fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs, p_dst2);
                    rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs, p_dst2);
                }
                tp_raise_flag(tp_flag_f_d_, want, cs);
                stamp(l, 21);
                grouped_full(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows);
                stamp(l, 22);
                dump_moe_hitout();   // DIAG: rank 0's FULL-width MoE band (bit-exact reference for the split)
            } else {
                // SPLIT (two phases): gate/up for BOTH group sets, exchange the two local hq halves over P2P,
                // then the down for both sets (n_embd/2 rows, full n_ff reduction) reading the assembled hq.
                gu_split(p_ptr, p_start, p_counts, 0, tp_moe_scratch_);
                stamp(l, 20);
                if (device_plan_) wait_flag_ge_or(m_flagB_, want, skip_, cs);
                else wait_flag_ge(m_flagB_, want, cs);
                // the PCIe share: stage it with a copy kernel, then point the plan's pointers at the staging area
                if (sink_.pcie_mode == 2) {
                    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                    uint8_t* stage = staging_ + (size_t) (0 * per) * lay.max_blob;
                    const unsigned long long* p_dst2 = (const unsigned long long*) (pl + dst2_off_);
                    // Half-fetch: this rank copies only the byte windows its own kernels read (gate/up rows
                    // [0,320) and down output rows [0,1280) plus the tail), so half the blob crosses this
                    // card's x8 link while rank 1 copies the other half over its own.  Admitted cache slots take
                    // mode 0 (the whole blob on THIS link) unless STRATA_TP_TWO_RANK_SLOT=1, which splits the fill
                    // instead: measured 3% slower per window, because rank 1's half arrives as 16-byte peer stores.
                    if (tp_half_fetch_)
                        fetch_blobs_spans(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l),
                                          tp_fetch_spans_[(size_t) l], (int) per, cs, p_dst2,
                                          (tp_slot_cover_ && tp_two_rank_slot_) ? 2 : 0);
                    else
                        fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs, p_dst2);
                    rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs, p_dst2);
                }
                tp_raise_flag(tp_flag_f_d_, want, cs);   // the staging area's pointers are final: rank 1 may read them
                stamp(l, 21);
                gu_split(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, tp_moe_scratch2_);
                // both ranks' gate/up halves are now written: rank 1's local hq (both sets) is ready once it has
                // raised flag_h; then assemble the full hq for both sets and tell rank 1 ours is ready (flag_i).
                wait_flag_ge(tp_flag_h_d_, want, cs);
                exchange_hq(tp_moe_scratch_, tp_moe_hq_, tp_moe_hq_peer_);
                exchange_hq(tp_moe_scratch2_, tp_moe_hq2_, tp_moe_hq2_peer_);
                dump_moe_hq_diag(0);  // DIAG: set0 assembled + local hq, after the exchange, before the down
                dump_moe_hq_diag(2);  // DIAG: set2 (PCIe) assembled + local hq
                tp_raise_flag(tp_flag_i_d_, want, cs);
                // the down: half the output rows, full n_ff reduction, over the unshifted plan pointers.
                down_split(p_ptr, p_start, p_counts, 0, tp_moe_hq_);
                down_split(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, tp_moe_hq2_);
                dump_moe_hitout();    // DIAG: this rank's pitched MoE band (set0 + set2), after both downs
                // Stitch rank 1's band half into ours BEFORE the combine.  The single card reduces all K
                // experts in one chain and adds the shared expert last; summing two per-rank chains instead
                // gives a different fp32 association wherever the residency table put experts on the CPU -
                // measured as a 1-ULP (7.45e-09 at 0.068) drift in rank 1's columns only, which the GR gate's
                // q8_1 then amplified into a different token at t=1.  With the peer's half in place our
                // `parts_` is byte-identical to the single card's input, so the combine reproduces it exactly.
                // Rank 1 keeps its own half and publishes ZERO from its combine, so the allreduce below is a
                // plain copy of this result to its `bo_` (`x + 0.0f` is bitwise `x`).
                wait_flag_ge(tp_flag_j_d_, want, cs);
                moe_hit_stitch(tp_moe_hitc_peer_, hit_out, p_dst, p_counts + 1, (int) cap, (int) g.n_embd,
                               (int) (g.n_embd / 2), cs);
                stamp(l, 22);
            }
            // the CPU share, then the hit rows: exactly the single-card path's order
            if (device_plan_) {
                wait_flag_ge_or(m_flag_, want, skip_, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_, (long long) T * K * N, skip_, want, cs);
            } else {
                const bool fuse_rows = fuse_rows_on() && !prof_on_ && !remote_opt_ && dec_batch && m_bal_ == nullptr;
                if (!fuse_rows) {
                    if (m_bal_ != nullptr)
                        wait_flag_ge_stamped(m_flag_, want, nullptr, m_bal_ + ((l - lb_) * 2) * 4 + 2, cs);
                    else wait_flag_ge(m_flag_, want, cs);
                }
                stamp(l, 23);
                if (remote_opt_)
                    remote_opt_->copy_rows(parts_out, m_ymiss_, 0, T, p_dst, p_counts + 1, cs);
                else if (fuse_rows)
                    copy_rows_from_mapped_after(m_flag_, want, parts_out, m_ymiss_, (int64_t) T * K, (int) N, p_dst,
                                                p_counts + 1, cs);
                else if (dec_batch)
                    copy_rows_from_mapped(parts_out, m_ymiss_, (int64_t) T * K, (int) N, p_dst, p_counts + 1, cs);
                else copy_from_mapped(parts_out, m_ymiss_, (int64_t) T * K * N, cs);
            }
            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, (int) N, cs);
            stamp(l, 24);
            // rank 0 adds the shared expert; rank 1 does not, so the allreduce's sum carries it exactly once
            native_moe_combine_multi(parts_, w_, shared_, bo_, (int) N, (int) K, (int) T, cs);
            if (m_tpdump_ != nullptr && l < 2)   // DIAG: rank 0's own combine (it carries `shared_`)
                cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 56) + (size_t) l * 4, bo_, 4 * sizeof(float),
                                cudaMemcpyDeviceToHost, cs);
            // DIAG (STRATA_TP_VEC=N): site 3, rank 0's OWN half-width partial, BEFORE the allreduce.  Rank 1
            // never runs `run()` (only the coordinator does), so its half cannot be dumped by the same writer;
            // `site1 - site3` is what rank 1 contributed, which is exactly the split's other half.
            if (const int64_t vat = tp_vec_at(l, 3); vat >= 0)
                cudaMemcpyAsync(m_tpdump_ + vat, bo_, (size_t) tp_vec_len() * sizeof(float),
                                cudaMemcpyDeviceToHost, cs);
            // wait for rank 1's partial, sum both bo_, tell rank 1 - one launch (P1); STRATA_TP_FUSED=0: three
            {
                std::string aerr;
                if (!tp_fused) {
                    wait_flag_ge(tp_flag_d_d_, want, cs);
                    if (!tp_ar_.allreduce(bo_, tp_bo_peer_, (int64_t) T * N, cs, nullptr, aerr)) {
                        err = "verify: tp MoE allreduce at layer " + std::to_string(l) + ": " + aerr;
                        return false;
                    }
                    tp_raise_flag(tp_flag_e_d_, want, cs);
                } else
                if (!tp_ar_.allreduce_wait_raise(bo_, tp_bo_peer_, (int64_t) T * N, cs, tp_flag_d_d_, want,
                                                 tp_flag_e_d_, want, aerr)) {
                    err = "verify: tp MoE allreduce at layer " + std::to_string(l) + ": " + aerr;
                    return false;
                }
            }
        } else {
            if (moe_full) {   // DIAG: rank 1 contributes zero; the allreduce then carries rank 0's full result
                cudaMemsetAsync(bo_, 0, (size_t) T * N * sizeof(float), cs);
                std::string aerr;
                if (!tp_fused) {
                    tp_raise_flag(tp_flag_d_d_, want, cs);
                    wait_flag_ge(tp_flag_e_d_, want, cs);
                } else
                if (!tp_ar_.publish_wait(tp_flag_d_d_, want, tp_flag_e_d_, want, cs, aerr)) {
                    err = "verify: tp MoE publish at layer " + std::to_string(l) + ": " + aerr;
                    return false;
                }
            } else {
            // rank 1: the plan block is rank 0's mapped copy, the staging area rank 0's device memory (P2P).
            // Its own `parts_` holds only this card's half, so the CPU rows' slots must read as zero: clearing
            // the buffer is what makes the weighted sum below leave them out (rank 0 owns those rows).
            cudaMemsetAsync(parts_, 0, (size_t) T * K * N * sizeof(float), cs);
            if (!ensure_ptrs()) {
                err = "verify: tp: the expert pointers could not be shifted";
                return false;
            }
            // SPLIT (two phases): gate/up for BOTH group sets (rank 1's half), then - once rank 0's halves are
            // ready (flag_i) - assemble the full hq from rank 0's local hq (over P2P) and our own, and raise
            // flag_h so rank 0's exchange may read OUR local hq.  THEN the down for both sets (n_embd/2 rows,
            // full n_ff reduction) over the unshifted plan pointers.
            gu_split(tp_ptr_, p_start, p_counts, 0, tp_moe_scratch_);
            stamp(l, 20);
            if (device_plan_) wait_flag_ge_or(m_flagB_, want, skip_, cs);
            else wait_flag_ge(m_flagB_, want, cs);
            // The PCIe share.  Half-fetch (the default): THIS rank copies its own byte windows of the miss blobs
            // into its OWN staging area over its OWN x8 link, so both links carry the fetch in parallel; the plan's
            // pointers are rebased to this rank's own address by the same deterministic kernel.  An admitted cache
            // slot is skipped here (mode 1): rank 0 fills it whole and this card reads its half over NVLink.  With
            // STRATA_TP_TWO_RANK_SLOT=1 this rank instead writes its windows INTO rank 0's slot (mode 2; measured
            // slower, see the member's comment).  Fallback (no spans computed): rank 0 stages the whole blobs and
            // this card reads that area over P2P, rebasing to rank 0's address (`tp_stage0_`, tp_share_doorbell).
            if (sink_.pcie_mode == 2) {
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (0 * per) * lay.max_blob;
                const unsigned long long* p_dst2 = (const unsigned long long*) (pl + dst2_off_);
                if (tp_half_fetch_)
                    fetch_blobs_spans(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l),
                                      tp_fetch_spans_[(size_t) l], (int) per, cs, p_dst2,
                                      (tp_slot_cover_ && tp_two_rank_slot_) ? 2 : 1);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, tp_half_fetch_ ? stage : tp_stage0_,
                            (int64_t) lay.blob_bytes(l), cs, p_dst2);
            }
            // In the host-DMA modes (pcie_mode 0/1) the plan's pointers still address RANK 0's staging area (the
            // pool thread wrote them into the mapped plan), so this rank reads that area over P2P and must wait
            // for rank 0 to have seen its DMA land.  With the two-rank slot fill this rank never reads that area;
            // otherwise (mode 1 skips it) it does not either, but the wait keeps the pair's phases aligned.
            if (!tp_two_rank_slot_ || !tp_half_fetch_ || sink_.pcie_mode != 2) wait_flag_ge(tp_flag_f_d_, want, cs);
            if (!ensure_ptrs2()) {
                err = "verify: tp: the PCIe expert pointers could not be shifted";
                return false;
            }
            stamp(l, 21);
            gu_split(tp_ptr2_, p_start2, p_counts + 2, kPcieGroupRows, tp_moe_scratch2_);
            // rank 1's halves are written: raise flag_h so rank 0's exchange may read our local hq, then wait
            // for rank 0's halves (flag_i) and assemble both sets' full hq (rank 0's local hq over P2P).
            tp_raise_flag(tp_flag_h_d_, want, cs);
            wait_flag_ge(tp_flag_i_d_, want, cs);
            exchange_hq(tp_moe_scratch_, tp_moe_hq_, tp_moe_hq_peer_);
            exchange_hq(tp_moe_scratch2_, tp_moe_hq2_, tp_moe_hq2_peer_);
            dump_moe_hq_diag(0);  // DIAG: set0 assembled + local hq, after the exchange, before the down
            dump_moe_hq_diag(2);  // DIAG: set2 (PCIe) assembled + local hq
            down_split(p_ptr, p_start, p_counts, 0, tp_moe_hq_);
            down_split(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, tp_moe_hq2_);
            dump_moe_hitout();    // DIAG: this rank's pitched MoE band (set0 + set2), after both downs
            // Our compact band half is final: tell rank 0 it may stitch it into its full-width band.
            tp_raise_flag(tp_flag_j_d_, want, cs);
            stamp(l, 22);
            stamp(l, 23);
            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, (int) N, cs);
            stamp(l, 24);
            native_moe_combine_multi(parts_, w_, nullptr, bo_, (int) N, (int) K, (int) T, cs);
            // Rank 0 now combines at FULL width (it stitched our half in above), so it already holds the
            // single card's result.  Publish zero: the allreduce below becomes a plain copy of rank 0's
            // `bo_` into ours (`x + 0.0f` is bitwise `x`), instead of summing two per-rank chains - which is
            // a different fp32 association from the single card's one-chain-then-shared order.
            cudaMemsetAsync(bo_, 0, (size_t) T * N * sizeof(float), cs);
            if (m_tpdump_ != nullptr && l < 2)   // DIAG: rank 1's own combine (it carries no `shared_`)
                cudaMemcpyAsync(m_tpdump_ + (size_t) (g.n_layers * 8 + 64) + (size_t) l * 4, bo_, 4 * sizeof(float),
                                cudaMemcpyDeviceToHost, cs);
            // publish this card's partial, then wait for rank 0's sum - one launch (P1); STRATA_TP_FUSED=0: two
            {
                std::string aerr;
                if (!tp_fused) {
                    tp_raise_flag(tp_flag_d_d_, want, cs);
                    wait_flag_ge(tp_flag_e_d_, want, cs);
                } else
                if (!tp_ar_.publish_wait(tp_flag_d_d_, want, tp_flag_e_d_, want, cs, aerr)) {
                    err = "verify: tp MoE publish at layer " + std::to_string(l) + ": " + aerr;
                    return false;
                }
            }
            }
        }
        // STRATA_TP_DUMP=1: the residual's head, per layer (see h_tpdump_)
        if (m_tpdump_ != nullptr)
            cudaMemcpyAsync(m_tpdump_ + (size_t) l * 8, bo_, 8 * sizeof(float), cudaMemcpyDeviceToHost, cs);
        // DIAG (STRATA_TP_VEC=N): site 1, this layer's combined residual in full - the allreduce's sum, which is
        // what the next layer reads
        if (const int64_t vat = tp_vec_at(l, 1); vat >= 0)
            cudaMemcpyAsync(m_tpdump_ + vat, bo_, (size_t) tp_vec_len() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                if (dec_batch && !g_no_multi_gr) gr_write_multi(Rt(0), bo_, inj2_, gs, Rt(0), (int) T, cs);
                else for (int t = 0; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(0), l, T, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(0), l, T, HC * N, bo_, (int) N, inj2_, (int) HC, true, cs);
        }
        // NOTHING else writes the residual streams here.  The ordinary window folds this layer's FFN write
        // into the NEXT layer's hyper-connection read (`gr_read_group` with `apply`); doing it here as well
        // applied it twice (layer 1's residual ended up 3% off while layer 0's bo_ matched to 1e-8).
        return true;
    };

    // ---------------------------------------------------------------- the layer loop
    if (!pre(lb_)) return false;
    for (int64_t l = lb_; l < le_; ++l) {
        if (!post(l)) return false;
        if (l + 1 < le_ && !pre(l + 1)) return false;
    }
    // The split experts left g_exp_phase at the last call's value (2 for the native split down).  Restore the
    // full pipeline so any non-TP `native_expert_grouped` call this process makes afterward (the bench, a later
    // capture) is not left down-only.
    native_expert_set_mode(-1, 0);

    // ---- the head (replicated: the full residual is the same on both cards)
    if (le_ < g.n_layers) {
        err = "verify: tp: a layer split with tensor parallelism is not supported";
        return false;
    }
    {
        // The final hyper-connection mixer, exactly the solo path's: `head_mixed_` is the MIXED n_embd vector, not a
        // copy of stream 0.  This block used to look for `output_hc_mix.weight` / `output_hc_inject.weight`, which no
        // manifest carries (family_layout.hpp declares only output_hc_{norm,down,up}), so it always fell through to
        // `copy_from_mapped(head_mixed_, Rt(0), T*N)` - stream 0 of the 4 hyper-connection streams, with the mixer
        // never run.  That fed the head a tensor ~1/hc the right size: measured |A|=146.8 (solo) vs |B|=22.8 (TP),
        // relL2 0.91, which is why every TP logit was ~4x too small.
        const WeightRef *won = wt.find("output_hc_norm.weight"), *wod = wt.find("output_hc_down.weight");
        const WeightRef *wou = wt.find("output_hc_up.weight");
        if (won == nullptr || wod == nullptr || wou == nullptr) {
            err = "verify: tp: an output_hc_* weight is missing";
            return false;
        }
        {
            // The last layer's residual write was done at the end of the layer loop above (its `l == n_layers - 1`
            // branch), so this read never folds a pending write - unless STRATA_FUSE_HEAD_GR=1 skipped that write,
            // in which case the read applies it, exactly as solo's fuse_head_gr branch does.
            if (head_mix_multi_enabled() && head_ != nullptr && head_->loaded()) {
                if (won->kind != WeightKind::F32 || wod->kind != WeightKind::Bf16InF32 ||
                    wou->kind != WeightKind::Bf16InF32) {
                    err = "verify: tp: the output_hc_* weights have the wrong engine forms";
                    return false;
                }
                FusedGrArgs fa[kFusedGrMaxT];
                for (int t = 0; t < T; ++t) {
                    FusedGrArgs& a = fa[t];
                    a.R = Rt(t); a.R_out = Rt(t); a.apply = fuse_head_gr;
                    a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                    a.w_norm = (const float*) won->data; a.w_down = (const uint16_t*) wod->data;
                    a.w_up = (const uint16_t*) wou->data; a.eps = EPS;
                    a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC; a.mixed = head_mixed_ + t * N;
                }
                fused_gr_read_multi(fa, T, xn_, cs);
            } else {
                if (fuse_head_gr) {
                    for (int t = 0; t < T; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                }
                for (int t = 0; t < T; ++t) {
                    BlockBuffers bb = ss.block;
                    bb.R = Rt(t);
                    bb.mixed = head_mixed_ + t * N;
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                }
            }
        }
        if (head_ == nullptr || !head_->loaded()) { err = "verify: tp: the head is not loaded"; return false; }
        // DIAG (STRATA_TP_VEC=N): site 2 of the pseudo-layer g.n_layers, the head's input in full
        if (const int64_t vat = tp_vec_at(g.n_layers, 2); vat >= 0)
            cudaMemcpyAsync(m_tpdump_ + vat, head_mixed_, (size_t) tp_vec_len() * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
        native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
        // The greedy pick, by the SAME rule the solo window uses.  This used to call `argmax_rows` unconditionally,
        // while solo calls it only when `argmax_rows_wanted()` - which is false on this hardware (it needs compute
        // capability 8.x; these are 2080 Tis, sm_75), so solo takes `sample_tokens`' greedy branch and the two paths
        // were picking with different kernels.  Both are greedy with the same tie rule (`arg_take`:
        // `ov > bv || (ov == bv && oi < bi)`), but a pick is the one place a difference is not a rounding error:
        // it is the token.  Same branch, same call, so the two paths cannot drift apart here.
        if (argmax_rows_wanted()) {   // sm_80 to sm_89: the pick over many blocks a row (bitwise the one-block kernel's)
            argmax_rows(head_logits_, T, (int) n_vocab_, arg_scratch_, m_out_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        }
        if (nan_guard_on()) strata::kernels::logits_nonfinite_rows(head_logits_, T, (int) n_vocab_, m_nf_, cs);
    }
    return true;
}

// ---- the TP window: one coordinator call drives both cards
bool Verifier::run_tp(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                      std::string& err) {
    using namespace strata::kernels;
    if (!tp_on_) { err = "verify: run_tp on a verifier that is not a TP rank"; return false; }
    if (tp_rank_ != 0) { err = "verify: run_tp must be called on rank 0 (the coordinator)"; return false; }
    if (tp_peer_ == nullptr || !tp_peer_->tp_on_) { err = "verify: run_tp needs the peer verifier (set_tp_peer)"; return false; }
    if (T < 1 || T > max_t_ || T > tp_peer_->max_t_) { err = "verify: tp window size out of range"; return false; }
    if (released_.load() || tp_peer_->released_.load()) {
        err = "verify: an earlier window never finished on the GPU (#267); restart the engine";
        return false;
    }
    if (pos0 + T > ss_->qsa_states[ss_->qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (tp_peer_->ss_->qsa_states[tp_peer_->ss_->qsa_primary()].max_cells < pos0 + T) {
        err = "verify: the peer's window runs past the context";
        return false;
    }
    // Self-check (once per rank): can THIS rank read the PEER's device memory?  The experts' FFN split has
    // rank 1 reading rank 0's expert arena and PCIe staging area over P2P, a direction no earlier TP path used
    // (the allreduce and the broadcast are rank 0's kernels touching rank 1's buffers).  With peer access off
    // the fault lands inside the window graph and surfaces only as "layer N never rang".
    if (tp_bo_peer_ != nullptr && tp_exec_[T] == nullptr) {
        float got = -12345.0f;
        const cudaError_t ce = cudaMemcpy(&got, tp_bo_peer_, sizeof(float), cudaMemcpyDeviceToHost);
        (void) cudaGetLastError();
        if (ce != cudaSuccess) {
            err = std::string("verify: tp: rank ") + std::to_string(tp_rank_) +
                  " cannot read the peer's device memory (P2P): " + cudaGetErrorString(ce);
            return false;
        }
    }
    // The TP window is captured with the HOST-DOORBELL expert plan: rank 0's pool publishes each layer's plan into
    // the mapped block, and the post (4357) spins on m_flagA_ / m_flagB_ / m_flag_ for it.  There is NO zero-doorbell
    // variant: record_window_tp's post unconditionally waits on those flags, and run_tp's host loop that would raise
    // them (the doorbell loop below) is skipped entirely when ar_on().  So the all-resident graph solo takes when
    // every expert is in VRAM would hang a TP window forever - the post spins on flags no host thread ever raises.
    // Force the doorbell graph: all_resident_ = false makes ar_on() false on both ranks, selecting the path that is
    // actually implemented.  On 2x 22GiB the 125B expert set is never 100% resident (init prints no
    // "zero-doorbell" line), so this changes nothing here; it only closes the hang on a machine where it would be.
    if (all_resident_ || tp_peer_->all_resident_) {
        all_resident_ = false;
        tp_peer_->all_resident_ = false;
        std::fprintf(stderr, "strata verify: tp: the window runs the host-doorbell expert plan (the zero-doorbell "
                             "all-resident graph is not implemented for TP and would hang: its post waits on doorbell "
                             "flags no host loop raises)\n");
    }
    // both cards' windows are captured once (this is a graph of both the dense half and the replicated MoE)
    if (!capture_tp(T, err)) return false;
    // Rank 1 shares rank 0's doorbell (the experts are replicated, so the plan is the same) - install it BEFORE
    // rank 1's graph is captured, because the plan block and the flag words are baked into that graph.
    tp_peer_->tp_share_doorbell(*this);
    // The dump block stays this rank's own: it is mapped pinned HOST memory, and a device pointer resolved on
    // one card is not writable from the other (unlike the P2P-visible device buffers it is not portable).
    if (!tp_peer_->capture_tp(T, err)) return false;
    if (!capture_commit_tp(err)) return false;
    if (!tp_peer_->capture_commit_tp(err)) return false;
    // the layer loop's handshake flags start at 0 for this window.  This runs on EVERY request, not just the
    // first: capture_tp() above returns early once the graph exists, but these words are the graphs' runtime
    // mailbox and carry last request's high-water mark (flag g ends at 2*n_layers = 96).  A flag left raised
    // makes every `wait_flag_ge` in the next request return at once, so a rank reads its peer's buffer before
    // the P2P write has landed.  Flag g is the GDN value-half handshake of the default column-parallel
    // out-projection (not just the FULL diagnostic), so it must be reset with the others.  Both ranks' pointers
    // name the SAME host words (tp_setup_flags hands the peer this rank's allocation): one store clears the pair.
    *(volatile uint32_t*) tp_flag_a_h_ = 0;
    *(volatile uint32_t*) tp_flag_b_h_ = 0;
    // flag c is the OTHER direction of that same value-half exchange: rank 0 raises it after its half has
    // landed in rank 1's y_full_[0, ZVt), and rank 1 spins on it.  Leaving it out here cost a whole window:
    // the word still held 2*n_layers from the previous request, so rank 1's `wait_flag_ge(c, l + 1)` returned
    // at once and it quantized `y_full_` before rank 0's broadcast landed - its 1280 rows then came from the
    // previous layer's values, and the per-layer allreduce was wrong in every element.
    *(volatile uint32_t*) tp_flag_c_h_ = 0;
    *(volatile uint32_t*) tp_flag_d_h_ = 0;
    *(volatile uint32_t*) tp_flag_e_h_ = 0;
    *(volatile uint32_t*) tp_flag_f_h_ = 0;
    *(volatile uint32_t*) tp_flag_g_h_ = 0;
    // flags h / i are the MoE hq exchange handshake (the split down's two-phase gate/up -> down).  Like the
    // others they are `want`-valued high-water marks a raise is a plain store into, so a leftover value from the
    // previous request would make wait_flag_ge return at once; both ranks name the same host words.
    *(volatile uint32_t*) tp_flag_h_h_ = 0;
    *(volatile uint32_t*) tp_flag_i_h_ = 0;
    // flag j is the MoE band stitch handshake (rank 1 -> rank 0: "my pitched band half is final, read it").  It
    // carries `want` like the others, so a word still holding 48 from the previous request makes rank 0's very
    // first `wait_flag_ge(j, 1)` return at once: it stitches rank 1's PREVIOUS-request band half into this
    // window's first layer, which shows up as a whole-band error (0.29 at window 1's L0, rank 1's columns only -
    // the exact shape the stitch reads) rather than an ULP.  Every other flag is listed here; this one must be too.
    *(volatile uint32_t*) tp_flag_j_h_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    // both cards stage the same tokens/positions: their dense halves read the same residual
    stage_inputs(T, tokens, pos0);
    {
        const OnDevice on_peer(tp_peer_->device_);
        tp_peer_->stage_inputs(T, tokens, pos0);
    }
    // The PLE rows: layer 1's block reads `m_ple_`, and in the ordinary window run()'s host thread gathers them
    // (ngram_rows from the session's last two tokens, then the table read).  run_tp has no host doorbell - the
    // graphs run to completion - so the gather happens here, before either graph is launched.  Both ranks read
    // their own mapped row block (`init` mapped one per verifier) and the rows do not differ between the cards
    // (they hash the token window), so each rank's block is filled from the same `ple_rows`.
    if (ple_stage() && ss_->ple.ready()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
        int32_t prev[2] = {ss_->ple_prev[0], ss_->ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss_->ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
        if (!ss_->ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
        if (tp_peer_->ple_stage() && tp_peer_->ss_->ple.ready() && tp_peer_->h_ple_ != nullptr &&
            !tp_peer_->ss_->ple.table->gather_batch(ple_rows, (size_t) T, tp_peer_->h_ple_, err))
            return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
    }
    // ARRANGE: rank 1's window must be launched BEFORE rank 0's, so that rank 1's first layer's partial is
    // already on its way when rank 0's first wait spins.  (The spin kernels make the order safe either way -
    // whichever graph starts first just waits - but starting the consumer last keeps the critical path short.)
    {
        const OnDevice on_peer(tp_peer_->device_);
        const cudaError_t le = cudaGraphLaunch(tp_peer_->tp_exec_[T], tp_peer_->cs_);
        if (le != cudaSuccess) { err = std::string("verify: tp peer launch: ") + cudaGetErrorString(le); return false; }
    }
    {
        const OnDevice on_self(device_);
        const cudaError_t le = cudaGraphLaunch(tp_exec_[T], cs_);
        if (le != cudaSuccess) {
            err = std::string("verify: tp launch: ") + cudaGetErrorString(le);
            (void) tp_peer_->release_gpu_waits(5000);   // no spin kernel may outlive this call
            return false;
        }
    }
    // the window runs both graphs to completion.  Rank 1 finishes on its own after its last layer; rank 0's
    // graph ends with the head.  Neither graph has a host-side doorbell (the experts are VRAM resident), so the
    // coordinator only hands the CPU experts over if the caller supplied a pool.
    if (ar_on()) {
        // the all-resident graph serves no layer from the host: both graphs run to completion
    } else {
        // the doorbell path: rank 0's graph publishes each layer's plan as it reaches it
        const int G = 1;
        const int64_t steps = (le_ - lb_) * G;
        volatile uint32_t* const seq = h_seq_;
        volatile uint32_t* const flag = h_flag_;
        for (int64_t k = 0; k < steps; ++k) {
            const int64_t l = lb_ + k / G;
            const uint32_t want = (uint32_t) (k + 1);
            const Clock::time_point a = Clock::now();
            uint32_t spins = 0;
            progress_at("verify tp window: waiting for the GPU to reach layer", l);
            while (*seq < want) {
                _mm_pause();
                if ((++spins & 1023u) != 0) continue;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: tp: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    (void) tp_peer_->release_gpu_waits(5000);
                    return false;
                }
                if (Clock::now() - a > std::chrono::seconds(6)) {
                    // the GPU is stuck: the flags say WHERE (a card waiting on the other's handshake, or a
                    // layer that never reached its doorbell)
                    std::fprintf(stderr,
                                 "strata verify: tp stall at layer %lld: seq=%u want=%u flag_a=%u flag_b=%u "
                                 "(peer flag_a=%u flag_b=%u)\n",
                                 (long long) l, (unsigned) *seq, (unsigned) want,
                                 (unsigned) *(volatile uint32_t*) tp_flag_a_h_, (unsigned) *(volatile uint32_t*) tp_flag_b_h_,
                                 (unsigned) *(volatile uint32_t*) tp_peer_->tp_flag_a_h_,
                                 (unsigned) *(volatile uint32_t*) tp_peer_->tp_flag_b_h_);
                    err = "verify: tp: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                    (void) tp_peer_->release_gpu_waits(5000);
                    return false;
                }
            }
            cur_layer_ = want - 1;
            set_plan_slot(k % G);
            const int tb = 0, n = T;
            if (pool != nullptr)
                pool(user, h_x_ + (size_t) tb * g_->n_embd, h_ids_ + (size_t) tb * ss_->k, n, ss_->k,
                     h_ymiss_ + (size_t) tb * ss_->k * g_->n_embd, l);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            if (*(volatile uint32_t*) h_flagA_ != want) {   // the pool published no plan: an empty one
                sink_.counts[0] = 0;
                sink_.counts[1] = 0;
                sink_.counts[2] = 0;
                sink_.start[0] = 0;
                sink_.start2[0] = 0;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *(volatile uint32_t*) h_flagA_ = want;
                raise_flag(h_flagB_, want);
            }
            *flag = want;
        }
    }
    // both graphs are still in flight; join them
    {
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) {
            err = std::string("verify: tp: ") + cudaGetErrorString(se);
            (void) tp_peer_->release_gpu_waits(5000);
            return false;
        }
        const OnDevice on_peer(tp_peer_->device_);
        const cudaError_t pe = cudaStreamSynchronize(tp_peer_->cs_);
        if (pe != cudaSuccess) {
            err = std::string("verify: tp peer: ") + cudaGetErrorString(pe);
            return false;
        }
    }
    // #871, the same check solo makes at both of its exits (verify.cpp:2911 and 2935).  `refresh_ar` leaves this
    // word set when the residency table changed mid-window and a layer then met an expert that is not in VRAM -
    // the graph still runs, serving that layer from whatever the plan held, so the window's tokens are wrong.
    // TP passed `m_plan_err_` into `resident_plan` (4282) and then never read it: a window that hit the race
    // returned tokens instead of an error.  Cleared on read exactly as solo clears it, so the next window starts
    // from a known state.
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
    }
    if (copy_used_) {
        cudaStreamSynchronize(copy_);
        copy_used_ = false;
    }
    // STRATA_MOE_HQ_DUMP=1: both ranks' graphs are joined (their D2H nodes complete), so write the one-shot
    // layer-0 set0 hq to disk - the ASSEMBLED full hq (each rank's own view, read back through the same P2P
    // path the down uses) and the rank's LOCAL gate/up hq.  Both ranks write their own buffer; the two .hq
    // files must be byte-identical if the P2P hq exchange is clean (rank 0's down is bit-exact, so the
    // divergence - if any - is in the hq image rank 1 assembles from rank 0's P2P half).
    if (h_moe_hq_diag_ != nullptr && !moe_hq_dump_written_ && moe_hq_dump_cap_ > 0) {
        const char* ge = std::getenv("STRATA_MOE_HQ_DUMP");
        if (ge != nullptr && std::atoi(ge) != 0) {
            const int64_t capd = moe_hq_dump_cap_;
            const size_t hq_sz = (size_t) capd * (size_t) ((tp_moe_.layout.n_ff * 2) / 32) * 36;
            const size_t lq_sz = (size_t) capd * (size_t) (tp_moe_.n_ff_local / 32) * 36;
            const size_t hit_sz = (size_t) capd * (size_t) g_->n_embd * sizeof(float);
            const size_t lq_region = (size_t) ((int64_t) max_t_ * (int64_t) ss_->k) *
                                     (size_t) (tp_moe_.n_ff_local / 32) * 36;   // the capx-sized local-hq slot
            const size_t set_region = (size_t) moe_hq_diag_hq_bytes_ + lq_region;  // one set = hq slot + lq slot
            const char* base = std::getenv("STRATA_MOE_HQ_DUMP_FILE");
            if (base == nullptr) base = "/tmp/strata_moe_hq_dump";
            auto w2 = [&](const char* tag, const Verifier* self, const Verifier* peer) {
                if (self == nullptr) return;
                const uint8_t* s0 = self->h_moe_hq_diag_;                        // set0 hq slot
                const uint8_t* s0l = self->h_moe_hq_diag_ + (size_t) moe_hq_diag_hq_bytes_;
                const uint8_t* s2 = self->h_moe_hq_diag_ + set_region;           // set2 hq slot
                const uint8_t* s2l = self->h_moe_hq_diag_ + set_region + (size_t) moe_hq_diag_hq_bytes_;
                const std::string r = std::to_string((int) self->tp_rank_);
                const std::string pref = std::string(base) + tag;
                const std::vector<std::string> names = { pref + "0.hq" + r, pref + "0.lq" + r, pref + "2.hq" + r,
                                                         pref + "2.lq" + r };
                const uint8_t* ptrs[] = { s0, s0l, s2, s2l };
                const size_t nsz[] = { hq_sz, lq_sz, hq_sz, lq_sz };
                for (int i = 0; i < 4; ++i) {
                    if (FILE* f = std::fopen(names[i].c_str(), "wb")) {
                        std::fwrite(ptrs[i], 1, nsz[i], f);
                        std::fclose(f);
                    }
                }
                // the per-layer MoE band (ho_<l>.<rank>): find the first diverging layer vs the mofull band.
                for (int64_t ll = 0; ll < g_->n_layers; ++ll) {
                    const std::string hn = pref + "ho_" + std::to_string((int) ll) + "." + r;
                    const uint8_t* ho = self->h_moe_hq_diag_ + 2 * set_region + (size_t) ll * hit_sz;
                    if (FILE* f = std::fopen(hn.c_str(), "wb")) {
                        std::fwrite(ho, 1, hit_sz, f);
                        std::fclose(f);
                    }
                }
            };
            w2(".hq", this, nullptr);
            if (tp_peer_ != nullptr) w2(".hq", tp_peer_, nullptr);
            moe_hq_dump_written_ = true;
            std::fprintf(stderr,
                         "strata verify: tp: wrote the layer-0 set0+set2 hq dump (cap %lld, hq %zu B, lq %zu B, hit %zu B "
                         "per rank)\n",
                         (long long) capd, hq_sz, lq_sz, hit_sz);
        }
    }
    // STRATA_GDN_VALDUMP: the per-rank GDN value vectors staged by record_window_tp's IN-capture D2H (pinned, graph
    // nodes) now hold window-0 data - both window graphs have launched and the streams synced above, so the copies
    // have landed in pinned memory.  run_tp runs on RANK 0 (it launches the peer's graph too), so write BOTH ranks'
    // files from here: each rank's capture filled its own file-scope index (g_gdnval_*_[r], shared across the process).
    if (std::getenv("STRATA_GDN_VALDUMP") != nullptr) {
        static const char* vfn = std::getenv("STRATA_GDN_VALDUMP_FILE");
        static bool vwr = false;
        if (vfn != nullptr && !vwr && g_gdnval_T_ > 0) {
            for (int r = 0; r < 2; ++r) {
                if (!g_gdnval_fired_[r] || g_gdnval_full_h_[r] == nullptr || g_gdnval_half_h_[r] == nullptr) continue;
                if (FILE* f = std::fopen((std::string(vfn) + ".full.rank" + std::to_string(r)).c_str(), "wb")) {
                    std::fwrite(g_gdnval_full_h_[r], sizeof(float), (size_t) g_gdnval_T_ * (size_t) g_gdnval_ZV_, f);
                    std::fclose(f);
                }
                if (FILE* f = std::fopen((std::string(vfn) + ".half.rank" + std::to_string(r)).c_str(), "wb")) {
                    std::fwrite(g_gdnval_half_h_[r], sizeof(float), (size_t) g_gdnval_T_ * (size_t) g_gdnval_ZVt_, f);
                    std::fclose(f);
                }
                std::fprintf(stderr, "strata verify: tp: GDN value dump rank %d (T=%d ZV=%d ZVt=%d)\n",
                             r, g_gdnval_T_, g_gdnval_ZV_, g_gdnval_ZVt_);
            }
            vwr = true;
        }
    }
    if (prof_on_) collect_profile();
    if (!sample_window(T, tokens, pos0, err)) return false;
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    if (h_tpdump_ != nullptr) {
        {
            // rank 1's probes live in ITS mapped block (the dump buffer is not shared across devices)
            const float* h1 = tp_peer_ != nullptr ? tp_peer_->h_tpdump_ : nullptr;
            const size_t base0 = (size_t) g_->n_layers * 8;
            {
                const float* x0 = h_tpdump_ + base0 + 96;
                const float* x1 = h1 ? h1 + base0 + 96 : nullptr;
                std::fprintf(stderr, "TPDUMP tp XM1 r0=%.9e,%.9e r1=%.9e,%.9e\n",
                             (double) x0[0], (double) x0[1],
                             x1 ? (double) x1[0] : 0.0, x1 ? (double) x1[1] : 0.0);
            }
            {
                const float* m0 = h_tpdump_ + base0 + 100;
                const float* m1 = h1 ? h1 + base0 + 100 : nullptr;
                std::fprintf(stderr,
                             "TPDUMP tp MOEIN l0 r0=%.9e,%.9e r1=%.9e,%.9e | l1 r0=%.9e,%.9e r1=%.9e,%.9e\n",
                             (double) m0[0], (double) m0[1], m1 ? (double) m1[0] : 0.0, m1 ? (double) m1[1] : 0.0,
                             (double) m0[4], (double) m0[5], m1 ? (double) m1[4] : 0.0, m1 ? (double) m1[5] : 0.0);
            }
            for (int64_t ll = 0; ll < 2; ++ll) {
                const double a = (double) h_tpdump_[base0 + 56 + (size_t) ll * 4];
                const double b = (double) (h1 ? h1[base0 + 64 + (size_t) ll * 4] : 0.0f);
                std::fprintf(stderr, "TPDUMP tp CMB l%lld r0=%.9e r1=%.9e sum=%.9e\n", (long long) ll, a, b, a + b);
            }
            for (int64_t ll = 0; ll < 2; ++ll) {
                const int32_t* id = (const int32_t*) (h_tpdump_ + (size_t) g_->n_layers * 8 + 72 + (size_t) ll * 8);
                const float* ww = h_tpdump_ + (size_t) g_->n_layers * 8 + 72 + (size_t) ll * 8 + 4;
                const int32_t* id1 = h1 ? (const int32_t*) (h1 + (size_t) g_->n_layers * 8 + 72 + (size_t) ll * 8) : nullptr;
                std::fprintf(stderr, "TPDUMP tp ROUTE l%lld r0=%d,%d,%d,%d w=%.6e,%.6e,%.6e,%.6e r1=%d,%d,%d,%d\n",
                             (long long) ll, id[0], id[1], id[2], id[3], (double) ww[0], (double) ww[1],
                             (double) ww[2], (double) ww[3], id1 ? id1[0] : -1, id1 ? id1[1] : -1,
                             id1 ? id1[2] : -1, id1 ? id1[3] : -1);
            }
        }
        for (int64_t ll = 0; ll < 2; ++ll) {
            const int32_t* c = (const int32_t*) (h_tpdump_ + (size_t) g_->n_layers * 8 + 88 + (size_t) ll * 4);
            std::fprintf(stderr, "TPDUMP tp PLAN l%lld hits=%d cpu=%d pcie=%d\n", (long long) ll, c[0], c[1], c[2]);
        }
        {
            const float* h1 = tp_peer_ != nullptr ? tp_peer_->h_tpdump_ : nullptr;
            const float* q3 = h_tpdump_ + (size_t) g_->n_layers * 8 + 492;
            const float* q1 = h1 ? h1 + (size_t) g_->n_layers * 8 + 492 : nullptr;
            const int32_t* sel3 = (const int32_t*) (h_tpdump_ + (size_t) g_->n_layers * 8 + 512);
            const int32_t* sel1 = h1 ? (const int32_t*) (h1 + (size_t) g_->n_layers * 8 + 512) : nullptr;
            std::fprintf(stderr,
                         "TPDUMP tp QSA3 k=%.9e,%.9e v=%.9e,%.9e q=%.9e,%.9e qi=%.9e,%.9e s=%.6e,%.6e sel=%d,%d,%d,%d attn=%.9e,%.9e\n",
                         (double) q3[0], (double) q3[1], (double) q3[4], (double) q3[5],
                         (double) q3[8], (double) q3[9], (double) q3[12], (double) q3[13],
                         (double) q3[16], (double) q3[17],
                         sel3[0], sel3[1], sel3[2], sel3[3], (double) q3[24], (double) q3[25]);
            std::fprintf(stderr,
                         "TPDUMP tp QSA3r1 k=%.9e,%.9e v=%.9e,%.9e q=%.9e,%.9e qi=%.9e,%.9e s=%.6e,%.6e sel=%d,%d,%d,%d attn=%.9e,%.9e\n",
                         q1 ? (double) q1[0] : 0.0, q1 ? (double) q1[1] : 0.0, q1 ? (double) q1[4] : 0.0, q1 ? (double) q1[5] : 0.0,
                         q1 ? (double) q1[8] : 0.0, q1 ? (double) q1[9] : 0.0, q1 ? (double) q1[12] : 0.0, q1 ? (double) q1[13] : 0.0,
                         q1 ? (double) q1[16] : 0.0, q1 ? (double) q1[17] : 0.0,
                         sel1 ? sel1[0] : -1, sel1 ? sel1[1] : -1, sel1 ? sel1[2] : -1, sel1 ? sel1[3] : -1,
                         q1 ? (double) q1[24] : 0.0, q1 ? (double) q1[25] : 0.0);
            const double p0 = (double) h_tpdump_[(size_t) g_->n_layers * 8 + 520];
            const double p1 = (double) (h1 ? h1[(size_t) g_->n_layers * 8 + 520] : 0.0f);
            std::fprintf(stderr, "TPDUMP tp L3P r0=%.9e r1=%.9e sum=%.9e\n", p0, p1, p0 + p1);
            const double m0 = (double) h_tpdump_[(size_t) g_->n_layers * 8 + 524];
            const double m1 = (double) (h1 ? h1[(size_t) g_->n_layers * 8 + 524] : 0.0f);
            std::fprintf(stderr, "TPDUMP tp M3 r0=%.9e,%.9e r1=%.9e,%.9e\n", m0,
                         (double) h_tpdump_[(size_t) g_->n_layers * 8 + 525], m1,
                         h1 ? (double) h1[(size_t) g_->n_layers * 8 + 525] : 0.0);
            const float* w0 = h_tpdump_ + (size_t) g_->n_layers * 8 + 528;
            const float* w1 = h1 ? h1 + (size_t) g_->n_layers * 8 + 528 : nullptr;
            std::fprintf(stderr,
                         "TPDUMP tp W3 r0 wk=%.9e,%.9e wv=%.9e,%.9e wq=%.9e,%.9e rk=%.9e,%.9e rv=%.9e,%.9e | r1 wk=%.9e,%.9e wv=%.9e,%.9e\n",
                         (double) w0[0], (double) w0[1], (double) w0[4], (double) w0[5], (double) w0[8], (double) w0[9],
                         (double) w0[12], (double) w0[13], (double) w0[16], (double) w0[17],
                         w1 ? (double) w1[0] : 0.0, w1 ? (double) w1[1] : 0.0, w1 ? (double) w1[4] : 0.0,
                         w1 ? (double) w1[5] : 0.0);
            const float* g0 = h_tpdump_ + (size_t) g_->n_layers * 8 + 548;
            const float* g1 = h1 ? h1 + (size_t) g_->n_layers * 8 + 548 : nullptr;
            std::fprintf(stderr,
                         "TPDUMP tp GDN0 r0 q=%.9e,%.9e k=%.9e,%.9e v=%.9e,%.9e h=%.9e,%.9e | r1 q=%.9e,%.9e h=%.9e,%.9e\n",
                         (double) g0[0], (double) g0[1], (double) g0[4], (double) g0[5], (double) g0[8],
                         (double) g0[9], (double) g0[12], (double) g0[13],
                         g1 ? (double) g1[0] : 0.0, g1 ? (double) g1[1] : 0.0,
                         g1 ? (double) g1[12] : 0.0, g1 ? (double) g1[13] : 0.0);
        }
        std::fprintf(stderr, "TPDUMP tp T=%d pos0=%lld", T, (long long) pos0);
        for (int64_t l = 0; l < g_->n_layers; ++l)
            std::fprintf(stderr, " %.6e", (double) h_tpdump_[(size_t) l * 8]);
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "TPDUMP tp xm l0=%.6e,%.6e l1=%.6e,%.6e R l0=%.6e,%.6e l1=%.6e,%.6e\n",
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 10], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 11],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 13], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 14],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 16], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 17],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 19], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 20]);
        std::fprintf(stderr, "TPDUMP tp Rin l0=%0.6e,%0.6e l1=%0.6e,%0.6e\n",
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 28], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 29],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 31], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 32]);
        std::fprintf(stderr, "TPDUMP tp bo8 l0=%.9e,%.9e,%.9e,%.9e Rafter l0=%0.6e l1=%0.6e\n",
                     (double) h_tpdump_[0], (double) h_tpdump_[1], (double) h_tpdump_[2], (double) h_tpdump_[3],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 46], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 49]);
        std::fprintf(stderr, "TPDUMP tp Rpost l0=%0.6e l1=%0.6e\n",
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 40], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 43]);
        std::fprintf(stderr, "TPDUMP tp inj l0=%.6e,%.6e l1=%.6e,%.6e\n",
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 22], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 23],
                     (double) h_tpdump_[(size_t) g_->n_layers * 8 + 25], (double) h_tpdump_[(size_t) g_->n_layers * 8 + 26]);
        // STRATA_TP_VEC: the vector dumps as raw fp32, same layout as the solo path's
        if (const int64_t vat = tp_vec_at(0, 0); vat >= 0) {
            const int nv = tp_vec_len();
            static const char* vf = std::getenv("STRATA_TP_VEC_FILE");
            static int vn = 0;
            if (vf != nullptr) {
                // the rank is part of the name: both cards run this block, and site 3 is each card's OWN
                // pre-allreduce partial, so a shared name would leave whichever rank finished last
                const std::string path = std::string(vf) + ".tp" + std::to_string(tp_rank_) + "." +
                                         std::to_string(vn++);
                if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                    for (int64_t l = 0; l <= g_->n_layers; ++l)
                        for (int site = 0; site < kTpVecSites; ++site)
                            std::fwrite(h_tpdump_ + tp_vec_at(l, site), sizeof(float), (size_t) nv, f);
                    std::fclose(f);
                }
            }
        }
    }
    ++windows;
    progress_at("decode");
    progress_beat();
    last_t_ = T;
    last_pos0_ = pos0;
    // the state advance is this rank's (and the peer's) commit graph
    // The state advance is NOT done here: the caller commits the tokens it kept (commit -> tp_state_advance),
    // exactly as the single-card path does.  Advancing here would run the state twice per window.
    return true;
}


// ---- the pair's single doorbell: rank 1 reads rank 0's plan and flag words.
//
// Both cards route the same ids (the router is replicated over the same residual), so one published plan serves
// both: rank 1 would otherwise need a second pool thread and a second mapped block, and the two could disagree.
// Rank 0's pool publishes into its mapped plan/sequence/flag words; installing those same host pointers on
// rank 1 makes its `post` wait on the flags rank 0 raises and read the plan rank 0's pool wrote.
//
// The plan's staging pointers address rank 0's mapped staging area, which rank 1 reaches over P2P (enabled by
// tp_setup_flags).  Rank 1 never writes them.
void Verifier::tp_share_doorbell(Verifier& src) {
    h_seq_ = src.h_seq_;
    m_seq_ = src.m_seq_;
    h_flagA_ = src.h_flagA_;
    m_flagA_ = src.m_flagA_;
    h_flagB_ = src.h_flagB_;
    m_flagB_ = src.m_flagB_;
    h_plan_ = src.h_plan_;
    m_plan_ = src.m_plan_;
    plan_i32_ = src.plan_i32_;
    dst2_off_ = src.dst2_off_;
    // the sink's own view (publish_plan is never used on this rank, but `pcie_mode` and the counts are read)
    sink_.counts = src.sink_.counts;
    sink_.start = src.sink_.start;
    sink_.dst = src.sink_.dst;
    sink_.tok = src.sink_.tok;
    sink_.ptr = src.sink_.ptr;
    sink_.ptr2 = src.sink_.ptr2;
    sink_.start2 = src.sink_.start2;
    sink_.dst2 = src.sink_.dst2;
    sink_.cap = src.sink_.cap;
    sink_.pcie_mode = src.sink_.pcie_mode;
    // The plan is read from the mapped host block; its staging addresses are rank 0's device memory.  Keep that
    // address as `tp_stage0_` and NOT as this rank's `staging_`: with pcie_mode 2 (`auto`, the decode default) each
    // rank stages its OWN half of the miss blobs into its OWN `staging_` over its OWN PCIe link, so the two x8
    // links carry the fetch in parallel (a whole-blob fetch on one link was 0.42 ms/expert; half a blob on each is
    // ~0.21).  `tp_stage0_` is only what the host-DMA modes (pcie_mode 0/1) point this rank's plan at, since there
    // rank 0 stages for the pair and this card reads that area over P2P.  NVLink keeps its role unchanged: the
    // resident experts live in rank 0's arena and this rank reads its half of them over P2P.
    tp_stage0_ = src.staging_;
}

// ---- The TP commit graph: this rank's GDN conv history + recurrence advance by the window's kept tokens, plus
// the (replicated) QSA tail restore and indexer append.  The ordinary commit graph advances whole 48-head
// layers, which is exactly what a TP rank must not do: its GDN state holds the LOCAL v-heads only.
bool Verifier::capture_commit_tp(std::string& err) {
    if (tp_commit_exec_ != nullptr) return true;
    const OnDevice on_self(device_);   // the coordinator may call this on the peer's verifier
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t Ct = strata::core::kTpGdnConvChannels;
    const int64_t HVt = g.ssm_v_heads / 2;
    const int64_t MT = max_t_;
    const int64_t S = g.ssm_state_size;
    const int64_t TS = (s.idx_block - 1) * g.idx_key_dim;
    const int64_t ID = g.idx_key_dim;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t conv_off = S * HVt * S;   // the local conv history starts after the local recurrent state
    if (gdn_state_tp_ == nullptr) { err = "verify: tp: capture_commit_tp before tp_setup_scratch"; return false; }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: tp: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        // the GDN layers' local state, one launch for the conv histories and one for the recurrences
        int64_t gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) if (!is_qsa_layer(g, l)) ++gdn_index;
        int64_t n_gdn = 0;
        bool norms_ok = true;
        for (int64_t l = lb_; l < le_ && norms_ok; ++l)
            if (!is_qsa_layer(g, l)) {
                ++n_gdn;
                norms_ok = need(LayerView(*wt_, l), "ssm_norm.weight", err) != nullptr;
            }
        if (!norms_ok) ok = false;
        else if (n_gdn > 0) {
            // the batched form only takes the split-free kernels; otherwise the per-layer pair below
            const bool batched = gdn_commit_layers(
                gdn_state_tp_ + (size_t) gdn_index * gdn_state_tp_floats_, gdn_state_tp_floats_, conv_off,
                qkv_tp_ + (size_t) gdn_index * MT * Ct, MT * Ct, h_tp_ + (size_t) gdn_index * MT * Ct, MT * Ct,
                gate_tp_ + (size_t) gdn_index * MT * HVt, beta_tp_ + (size_t) gdn_index * MT * HVt, MT * HVt, (int) Ct,
                (int) g.ssm_k_heads, (int) HVt, (int) MT, (int) n_gdn, commit_, cs_);
            if (!batched) {
                int64_t gi = gdn_index;
                for (int64_t l = lb_; l < le_ && ok; ++l) {
                    if (is_qsa_layer(g, l)) continue;
                    const WeightRef* wnm = need(LayerView(*wt_, l), "ssm_norm.weight", err);
                    if (!wnm) { ok = false; break; }
                    float* state = gdn_state_tp_ + (size_t) gi * gdn_state_tp_floats_;
                    float* conv = state + conv_off;
                    gdn_conv_commit(conv, qkv_tp_ + (size_t) gi * MT * Ct, (int) Ct, commit_, cs_);
                    // the TP recurrence's commit: the local heads, the global head offset
                    gdn_step_norm_multi_tp(state, h_tp_ + (size_t) gi * MT * Ct, (int) Ct,
                                           gate_tp_ + (size_t) gi * MT * HVt, beta_tp_ + (size_t) gi * MT * HVt, z_tp_,
                                           (const float*) wnm->data, EPS, y_dummy_, (int) g.ssm_k_heads, (int) HVt,
                                           (int) MT, commit_, cs_, (int) MT, nullptr, (int) (tp_rank_ * HVt));
                    ++gi;
                }
            }
        }
        // the QSA layers' tail restore and indexer append (replicated: every card keeps its own full cache)
        int64_t qsa_index = 0;
        for (int64_t l = 0; l < lb_; ++l) if (is_qsa_layer(g, l)) ++qsa_index;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            if (!is_qsa_layer(g, l)) continue;
            const QsaState& st = ss.qsa_states[qsa_index];
            const WeightRef* wikn = need(LayerView(*wt_, l), "indexer.k_norm.weight", err);
            if (!wikn) { ok = false; break; }
            copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
            const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
            if (!g_no_batch_kv) {
                native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT, 0,
                                                (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
            } else {
                for (int64_t t = 0; t < MT; ++t)
                    native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                              (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
            }
            ++qsa_index;
        }
        // The PLE history rewind.  The layer-1 block above advances `ss.ple.hist` once per token in the window
        // (`ple_history_advance`, verify.cpp:3773), but the commit keeps only `commit_ + 1` of those rows - so
        // without this the NEXT window convolves its 4 taps over a history that runs T - (commit_ + 1) rows too
        // far ahead.  The first window is right only because the prefill's own commit already left the history
        // at the kept row; every window after it is wrong.  Solo does exactly this at verify.cpp:3300, from the
        // same snapshot (`hist_snap_`, written per row at 3778).  Restoring before that guard was in place fed
        // speculative rows back in and changed the tokens, which is why this line was once removed - the
        // snapshot is now only reachable for the rows the commit actually keeps.
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        ok = false;
        err = std::string("verify: tp commit: ") + e.what();
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        err = "verify: tp: end commit capture: " + std::string(cudaGetErrorString(ce));
        return false;
    }
    const bool made = instantiate_evicting(tp_commit_exec_, graph, {}, "verify: tp: commit instantiate: ", err);
    cudaGraphDestroy(graph);
    if (!made) return false;
    if (cudaGraphUpload(tp_commit_exec_, cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "verify: tp: the commit graph could not be uploaded";
        return false;
    }
    return true;
}

// ---- the state advance after a TP window: both cards' GDN state and the replicated QSA caches
bool Verifier::tp_state_advance(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: tp commit count out of range"; return false; }
    if (tp_commit_exec_ == nullptr) { err = "verify: tp commit graph was never captured"; return false; }
    const OnDevice on_self(device_);
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (tp_peer_ == nullptr) { err = "verify: tp_state_advance without a peer"; return false; }
    tp_peer_->h_commit_[0] = h_commit_[0];
    tp_peer_->h_commit_[1] = h_commit_[1];
    for (int t = 0; t < tp_peer_->max_t_; ++t) tp_peer_->h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const cudaError_t le = cudaGraphLaunch(tp_commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: tp commit launch: ") + cudaGetErrorString(le); return false; }
    {
        const OnDevice on_peer(tp_peer_->device_);
        const cudaError_t pe = cudaGraphLaunch(tp_peer_->tp_commit_exec_, tp_peer_->cs_);
        if (pe != cudaSuccess) { err = std::string("verify: tp peer commit launch: ") + cudaGetErrorString(pe); return false; }
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify: tp commit failed"; return false; }
    {
        const OnDevice on_peer(tp_peer_->device_);
        if (cudaStreamSynchronize(tp_peer_->cs_) != cudaSuccess) { err = "verify: tp peer commit failed"; return false; }
    }
    // The PLE history's last two tokens: `commit` forwards here and returns, so the advance its own tail does
    // (see the `ple_stage()` block there) has to happen on this path too - otherwise every window hashes its
    // n-gram rows from a stale pair and layer 1 reads the wrong PLE rows.  Both ranks hash their own window, so
    // both histories advance.
    if (ple_stage()) {
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
            if (tp_peer_->ple_stage()) {
                tp_peer_->ss_->ple_prev[0] = tp_peer_->ss_->ple_prev[1];
                tp_peer_->ss_->ple_prev[1] = last_tokens_[t];
            }
        }
    }
    return true;
}

namespace {
// set_commit_async (--serve): commit() launches the commit graph and returns without waiting for it; wait_commit
// polls the event it recorded.  Off by default (one commit at a time, synchronously).
bool g_commit_async = false;
}  // namespace

void Verifier::set_commit_async(bool on) { g_commit_async = on; }

bool Verifier::commit(int n_keep, std::string& err) {
#if defined(STRATA_GPU_METAL)
    return strata::metal_engine::window_commit(metal_, n_keep, err);
#endif
    // --tp: this rank's state is split (the GDN half lives in the tp arena), so the ordinary commit graph -
    // which advances whole 48-head layers - would be wrong.  The pair's commit graphs advance both halves.
    // (A rank whose TP window path never ran has no captured pair graph; STRATA_TP_SOLO=1 keeps this verifier on
    // the ordinary `run`, so it must not be routed here - see the tp_commit_exec_ check below.)
    if (tp_on() && tp_rank() == 0 && tp_peer_ != nullptr && tp_commit_exec_ != nullptr)
        return tp_state_advance(n_keep, err);
    if (tp_on() && tp_commit_exec_ != nullptr) {
        err = "verify: commit on a TP rank other than the coordinator";
        return false;
    }
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ == 1 && !always_publish_ && one_token_self_commit()) {
        // a one-token window has advanced the state itself (record_window): no commit graph
    } else {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // STRATA_COMMIT_TIMING=1 (measurement only): the commit graph's GPU time, from an event pair around it (read at the
        // next commit, when the pair has long completed); printed every 300 commits
        static struct CommitTiming {
            bool on = false, pending = false;
            cudaEvent_t b = nullptr, e = nullptr;
            double ms = 0;
            int64_t n = 0;
            CommitTiming() { const char* v = std::getenv("STRATA_COMMIT_TIMING"); on = v != nullptr && v[0] == '1'; }
        } ct;
        if (ct.on && ct.b == nullptr &&
            (cudaEventCreate(&ct.b) != cudaSuccess || cudaEventCreate(&ct.e) != cudaSuccess)) { ct.on = false; (void) cudaGetLastError(); }
        if (ct.on && ct.pending) {
            float ms = 0;
            if (cudaEventSynchronize(ct.e) == cudaSuccess && cudaEventElapsedTime(&ms, ct.b, ct.e) == cudaSuccess) { ct.ms += ms; ++ct.n; }
            ct.pending = false;
            if (ct.n > 0 && ct.n % 300 == 0)
                std::fprintf(stderr, "strata verify: commit graph %.3f ms GPU on average over %lld commits\n", ct.ms / (double) ct.n,
                             (long long) ct.n);
        }
        if (ct.on) cudaEventRecord(ct.b, cs_);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
        if (ct.on) { cudaEventRecord(ct.e, cs_); ct.pending = true; }
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        // set_commit_async: no wait here - the next window runs on the same stream after it, and the drafter (its own
        // stream) reads only this window's final rows and its own K/V. h_commit_ is next written after the next window's
        // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit).
        if (!g_commit_async || next_ != nullptr) {
            const cudaError_t se = cudaStreamSynchronize(cs_);
            if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        } else {
            const cudaError_t re = cudaEventRecord(commit_done_, cs_);
            if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
            (void) cudaStreamQuery(cs_);
            commit_pending_ = true;
        }
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}

bool Verifier::wait_commit(std::string& err) {
#if defined(STRATA_GPU_METAL)
    (void) err;
    return true;
#endif
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    return next_ == nullptr || next_->wait_commit(err);
}


// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty()) {
        err = "verify: init_slots needs at least one session";
        return false;
    }
    for (SessionState* x : slots) {
        // (this fork, --batch-max-context: a slot may hold fewer cells than the stage's own session - the window's
        // buffers are sized by the stage's, every position is checked against the slot's own max_cells below)
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells > ss_->max_cells || x->max_cells <= 0 || x->gdn_ord0 != ss_->gdn_ord0 ||
            x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, at most its context)";
            return false;
        }
    }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (cudaMalloc(&d, a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    std::fprintf(stderr, "strata verify: batch windows of up to %lld sequences (layers [%lld, %lld))%s\n", (long long) S,
                 (long long) lb_, (long long) le_, g_batch_self_commit ? ", one-row windows keep their rows" : "");
    return true;
}

bool Verifier::capture_batch(const int* rows, int S, int hbase, std::string& err) {
    cudaGraphExec_t& ex = exec_bm_[bkey(rows, S, hbase)];
    if (ex != nullptr) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    const bool ok = record_window(S, cs_, rerr);
    batch_rec_ = false;
    row_base_ = 0;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok || ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = !ok ? rerr : std::string("verify: end batch capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const bool made = instantiate_evicting(ex, graph, bkey(rows, S, hbase), "verify: batch instantiate: ", err);
    cudaGraphDestroy(graph);
    if (!made) return false;
    if (cudaGraphUpload(ex, cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess)
        (void) cudaGetLastError();   // only a warm-up: the first launch uploads it (see capture)
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "strata verify: captured the batch window over slots %s\n", list.c_str());
    return true;
}

bool Verifier::capture_commit_batch(const int* rows, int S, int hbase, std::string& err) {
    if (batch_self_commits(rows, S)) return true;   // the window keeps its rows: no commit graph
    cudaGraphExec_t& cex = commit_bm_[bkey(rows, S, hbase)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            if (t == 0 || rows[t] != rows[t - 1])
                copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                // STRATA_BATCH_GDN_GROUPS=1: one-row groups (no --batch-mtp proposals) commit their recurrence
                // in one launch for all slots; the conv commits stay per slot
                static const bool gdn_groups_on = [] {
                    const char* v = std::getenv("STRATA_BATCH_GDN_GROUPS");
                    return v != nullptr && std::atoi(v) != 0;
                }();
                bool one_row = S <= 8;
                for (int t = 1; one_row && t < S; ++t) one_row = rows[t] != rows[t - 1];
                strata::kernels::GdnGroups gg;
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    SessionState& sx = *slots_[(size_t) rows[first]];
                    const int32_t* keep = commitb_ + (size_t) rows[first] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) first * C,
                                    (int) C, keep, cs_);
                    if (gdn_groups_on && one_row) {
                        gg.state[gg.n] = state;
                        gg.keep[gg.n] = keep;
                        gg.first[gg.n] = first;
                        gg.ntok[gg.n] = 1;
                        ++gg.n;
                        continue;
                    }
                    gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C,
                                        gate_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        z_ + (size_t) first * ZV, (const float*) wnm->data, EPS,
                                        y_dummy_ + (size_t) first * ZV, (int) g.ssm_k_heads,
                                        (int) HV, t - first, keep, cs_,
                                        t - first > 1 ? t - first : 0);   // a 1-row group keeps the 0.1.39 kernel
                }
                if (gg.n > 0)
                    gdn_step_norm_groups(gg, h_L_ + (size_t) gdn_index * MT * C, (int) C,
                                         gate_L_ + (size_t) gdn_index * MT * HV, beta_L_ + (size_t) gdn_index * MT * HV,
                                         z_, (const float*) wnm->data, EPS, y_dummy_, (int) g.ssm_k_heads, (int) HV, true,
                                         cs_);
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS,
                                     TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int u = first; u < t; ++u)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + u) * ID,
                                                  commitb_ + (size_t) rows[first] * CB + 2 + (u - first),
                                                  0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S;) {
                const int first = t;
                while (t < S && rows[t] == rows[first]) ++t;
                copy_indexed(slots_[(size_t) rows[first]]->ple_hist, hist_snap_ + (size_t) first * HS,
                             HS, commitb_ + (size_t) rows[first] * CB + 1, HS, cs_);
            }
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: batch commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const bool made = instantiate_evicting(cex, graph, bkey(rows, S, hbase), "verify: batch commit instantiate: ", err);
    cudaGraphDestroy(graph);
    return made;
}

bool Verifier::evict_batch_graph(const std::vector<int>& keep, bool& evicted, std::string& err) {
    evicted = false;
    auto old = exec_bm_.end();
    uint64_t oldest = UINT64_MAX;
    for (auto it = exec_bm_.begin(); it != exec_bm_.end(); ++it) {
        if (it->first == keep) continue;
        const auto u = bm_used_.find(it->first);
        const uint64_t t = u == bm_used_.end() ? 0 : u->second;
        if (t < oldest) { oldest = t; old = it; }
    }
    if (old == exec_bm_.end()) return true;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "verify: synchronizing before batch graph eviction failed";
        return false;
    }
    const auto old_key = old->first;
    if (old->second) cudaGraphExecDestroy(old->second);
    exec_bm_.erase(old);
    bm_used_.erase(old_key);
    auto commit_old = commit_bm_.find(old_key);
    if (commit_old != commit_bm_.end()) {
        if (commit_old->second) cudaGraphExecDestroy(commit_old->second);
        commit_bm_.erase(commit_old);
    }
    evicted = true;
    return true;
}

int Verifier::drop_solo_graphs() {
    // the caller may stand on the other card's device (§8.3-3d switches the kernel between requests): enter ours
    const OnDevice on_self(device_);
    // --pipeline-windows prepares its graphs once (capture_all) and never captures them again: those stay
    if (ev_done_ != nullptr || cudaStreamSynchronize(cs_) != cudaSuccess) return 0;
    int n = 0;
    for (int t = 0; t < 9; ++t) {
        if (exec_[t]) { cudaGraphExecDestroy(exec_[t]); exec_[t] = nullptr; ++n; }
        if (exec_nr_[t]) { cudaGraphExecDestroy(exec_nr_[t]); exec_nr_[t] = nullptr; ++n; }
    }
    if (commit_exec_) { cudaGraphExecDestroy(commit_exec_); commit_exec_ = nullptr; ++n; }
    return n;
}

// STRATA_GRAPH_PRIO: the capture-time notes.  A fork note keeps cs's current tail (the node the branch's first kernel
// will depend on); a tail note keeps a priority stream's current tail and the priority its nodes should run at
// (1 = the device's highest, -1 = its lowest).  No-ops with the switch off, outside a capture, and on HIP / Metal.
void Verifier::note_prio_fork(cudaStream_t from) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    if (!graph_prio_on()) return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t* deps = nullptr;
    size_t nd = 0;
    if (prio_capture_deps(from, &st, &deps, &nd) != cudaSuccess || st != cudaStreamCaptureStatusActive) {
        (void) cudaGetLastError();
        return;
    }
    for (size_t i = 0; i < nd; ++i) prio_forks_.push_back(deps[i]);
#else
    (void) from;
#endif
}

void Verifier::note_prio_tail(cudaStream_t on, int prio) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    if (!graph_prio_on()) return;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t* deps = nullptr;
    size_t nd = 0;
    if (prio_capture_deps(on, &st, &deps, &nd) != cudaSuccess || st != cudaStreamCaptureStatusActive) {
        (void) cudaGetLastError();
        return;
    }
    for (size_t i = 0; i < nd; ++i) prio_tails_.push_back({deps[i], prio});
#else
    (void) on; (void) prio;
#endif
}

// STRATA_GRAPH_PRIO: what priorities the captured kernel nodes carry (a histogram, logged once per window graph).  If the
// capture copied none from the streams (some drivers read back 0 everywhere), each priority stream's nodes are found by
// walking its noted tails back to the noted forks and the attribute is set per node.
void Verifier::apply_node_priorities(cudaGraph_t graph, int T) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    int lo = 0, hi = 0;
    if (cudaDeviceGetStreamPriorityRange(&lo, &hi) != cudaSuccess) { (void) cudaGetLastError(); return; }
    auto histogram = [&](std::map<int, int>& h) {
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        std::vector<cudaGraphNode_t> nodes(nn);
        cudaGraphGetNodes(graph, nodes.data(), &nn);
        h.clear();
        for (cudaGraphNode_t nd : nodes) {
            cudaGraphNodeType ty;
            if (cudaGraphNodeGetType(nd, &ty) != cudaSuccess || ty != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeAttrValue v{};
            if (cudaGraphKernelNodeGetAttribute(nd, cudaKernelNodeAttributePriority, &v) == cudaSuccess) ++h[v.priority];
            else { (void) cudaGetLastError(); ++h[999]; }
        }
    };
    std::map<int, int> h;
    histogram(h);
    bool captured = false;
    for (auto& [p, c] : h) captured = captured || (p != 0 && p != 999 && c > 0);
    int repaired = 0, walked_forks = (int) prio_forks_.size(), walked_tails = (int) prio_tails_.size();
    if (!captured && !prio_tails_.empty()) {   // the fallback: per node from the capture-time notes
        std::set<cudaGraphNode_t> stop(prio_forks_.begin(), prio_forks_.end()), seen;
        for (auto& [tail, pr] : prio_tails_) {
            std::vector<cudaGraphNode_t> stack{tail};
            while (!stack.empty()) {
                cudaGraphNode_t nd = stack.back();
                stack.pop_back();
                if (stop.count(nd) || !seen.insert(nd).second) continue;
                cudaGraphNodeType ty;
                if (cudaGraphNodeGetType(nd, &ty) == cudaSuccess && ty == cudaGraphNodeTypeKernel) {
                    cudaKernelNodeAttrValue v{};
                    v.priority = pr > 0 ? hi : lo;
                    if (cudaGraphKernelNodeSetAttribute(nd, cudaKernelNodeAttributePriority, &v) == cudaSuccess) ++repaired;
                    else (void) cudaGetLastError();
                }
                size_t nd_n = 0;
                prio_node_deps(nd, nullptr, &nd_n);
                std::vector<cudaGraphNode_t> deps(nd_n);
                if (nd_n > 0 && prio_node_deps(nd, deps.data(), &nd_n) == cudaSuccess)
                    for (size_t i = 0; i < nd_n; ++i) stack.push_back(deps[i]);
                else (void) cudaGetLastError();
            }
        }
        histogram(h);
    }
    std::string s;
    for (auto& [p, c] : h) s += " " + std::to_string(c) + " x prio " + (p == 999 ? std::string("?") : std::to_string(p)) + ";";
    std::fprintf(stderr, "strata verify: the %d-token window's kernel node priorities (range %d..%d):%s %s (%d forks, %d tails noted%s)\n",
                 T, lo, hi, s.c_str(), captured ? "captured from the streams" : "NOT captured from the streams",
                 walked_forks, walked_tails, repaired > 0 ? (", " + std::to_string(repaired) + " nodes set from the notes").c_str() : "");
#else
    (void) graph; (void) T;
#endif
}

bool Verifier::evict_window_graph(int keep_t, bool& evicted, std::string& err) {
    evicted = false;
    if (no_window_eviction_ || keep_t < 1) return true;
    bool have[9] = {};
    for (int t = 1; t <= max_t_ && t < 9; ++t) have[t] = (ar_off_ ? exec_nr_[t] : exec_[t]) != nullptr;
    const int victim = strata::core::pick_window_victim(have, std::min(max_t_, 8), keep_t, prelaunched_);
    if (victim == 0) return true;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "verify: synchronizing before window graph eviction failed";
        return false;
    }
    cudaGraphExec_t& e = ar_off_ ? exec_nr_[victim] : exec_[victim];
    cudaGraphExecDestroy(e);
    e = nullptr;   // capture() re-captures it on that size's next use
    evicted = true;
    return true;
}

bool Verifier::instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t graph, const std::vector<int>& key,
                                   const char* what, std::string& err, int keep_t, unsigned long long flags) {
    // #997: every layout of active slots keeps its own graph pair (20-30 MiB each on an L40S), and with an expert cache
    // sized to the reserve the next window graph - a batch layout or a one-request window captured on first use - can
    // find no VRAM left: free the least recently used layouts until it fits (they are captured again when needed)
    // batch-oom (4090, parallel 4): the last layout's commit graph still did not fit and the engine exited.  Then the
    // solo window graphs (a batch graph only: a solo pair is being made otherwise), then the expert cache's slots.
    static const int test_oom = [] {   // test hook: the first N instantiations after the first batch graph fail
        const char* v = std::getenv("STRATA_TEST_GRAPH_OOM");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    static int test_left = test_oom;
    auto inst = [&]() -> cudaError_t {
        if (test_left > 0 && !exec_bm_.empty() && !key.empty()) { --test_left; return cudaErrorMemoryAllocation; }
        const cudaError_t e = cudaGraphInstantiate(&ex, graph, flags);
        if (e != cudaSuccess) ex = nullptr;
        return e;
    };
    cudaError_t ie = inst();
    int freed = 0, solo = 0, win = 0;
    size_t relief = 0;
    for (bool evicted = true; ie == cudaErrorMemoryAllocation && evicted;) {
        (void) cudaGetLastError();
        if (!evict_batch_graph(key, evicted, err)) return false;
        if (evicted) {
            ++freed;
            ie = inst();
        }
    }
    if (ie == cudaErrorMemoryAllocation && !key.empty()) {
        (void) cudaGetLastError();
        if ((solo = drop_solo_graphs()) > 0) ie = inst();
    }
    // #3702: a single-request window's graph that still does not fit frees the graphs of the other window sizes - the
    // larger ones first: they cost the most VRAM and are asked for least often, and each is captured again when its size
    // is next used.  Cheaper than taking the expert cache's slots below (that costs the hit rate).
    for (int w = 0; ie == cudaErrorMemoryAllocation && w < max_t_; ++w) {
        bool evicted = false;
        (void) cudaGetLastError();
        if (!evict_window_graph(keep_t, evicted, err)) return false;
        if (!evicted) break;
        ++win;
        ie = inst();
    }
    for (int r = 0; ie == cudaErrorMemoryAllocation && vram_relief_ && r < 8; ++r) {
        (void) cudaGetLastError();
        if (cudaStreamSynchronize(cs_) != cudaSuccess) break;
        const size_t got = vram_relief_((size_t) 64 << 20);
        if (got == 0) break;
        relief += got;
        ie = inst();
    }
    if (freed > 0 && solo == 0 && win == 0 && relief == 0)
        std::fprintf(stderr, "strata verify: no VRAM for a new window graph: freed the graphs of %d older batch slot "
                             "layouts, %zu kept (a larger --vram-reserve-mib keeps more)\n", freed, exec_bm_.size());
    else if ((solo > 0 || win > 0 || relief > 0) && !graph_oom_said_) {
        graph_oom_said_ = true;   // once: later ones are the same story
        std::fprintf(stderr, "strata verify: no VRAM for a new window graph: freed %d batch layouts, %d solo window "
                             "graphs, %d window sizes and %zu MiB of the expert cache%s (said once; a larger "
                             "--vram-reserve-mib avoids it)\n",
                     freed, solo, win, relief >> 20, ie == cudaSuccess ? "" : " - still not enough");
    }
    if (ie != cudaSuccess) {
        err = std::string(what) + cudaGetErrorString(ie);
        return false;
    }
    return true;
}

bool Verifier::stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err) {
    using namespace strata::kernels;
    // the row layout: a slot may own a contiguous group of rows (--batch-mtp), also across the stages of a layer split,
    // whose hand-off buffers hold kVerifyMaxT rows, or set_handoff_rows' (a stage that hands rows on or takes them)
    if (const char* why = batch_rows_error(rows, S, pos, (int) slots_.size(), max_t_, hbase,
                                           next_ != nullptr || lb_ > 0, handoff_rows_ > 0 ? handoff_rows_ : kVerifyMaxT)) {
        err = why;
        return false;
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t)
        if (pos[t] < 0 || pos[t] + 1 > slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    refresh_ar();
    // --batch-mtp only (limit 0 = 0.1.39: no eviction): slot rotation creates new layouts; bound the captured graph
    // pairs, evicting the least recently used layout.  Without a limit the same eviction frees VRAM for a capture.
    const auto key = bkey(rows, S, hbase);
    bool evicted = false;
    if (batch_graph_limit_ > 0 && exec_bm_.find(key) == exec_bm_.end() && exec_bm_.size() >= batch_graph_limit_ &&
        !evict_batch_graph(key, evicted, err))
        return false;
    bm_used_[key] = ++bm_tick_;
    if (!capture_batch(rows, S, hbase, err) || !capture_commit_batch(rows, S, hbase, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < S; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos[t], s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos[t];
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos[t];
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos[t];
    }
    if (ss_->ple.ready() && ple_stage()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
        for (int t = 0; t < S; ++t) {
            const SessionState& sx = *slots_[(size_t) rows[t]];
            int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
            if (t > 0 && rows[t] == rows[t - 1]) {
                prev[0] = t > 1 && rows[t - 2] == rows[t] ? tokens[t - 2] : sx.ple_prev[1];
                prev[1] = tokens[t - 1];
            }
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
        }
        if (!ss_->ple.table->gather_batch(ple_rows, (size_t) S, h_ple_, err)) return false;
    }
    // Each slot owns a contiguous group. The default commit keeps all its rows; speculative
    // decoding may change the prefix length after comparing the draft with these picks.
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        int32_t* c = h_commitb_ + (size_t) rows[first] * CB;
        c[0] = t - first;
        c[1] = t - first - 1;
        for (int64_t j = 0; j < max_t_; ++j)
            c[2 + j] = j < t - first ? (int32_t) pos[first + j] : -1;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = S;
    for (int t = 0; t < S; ++t) last_rows_[t] = rows[t];
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    for (int t = 0; t < S; ++t) { last_tokens_[t] = tokens[t]; last_pos_b_[t] = pos[t]; }
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (pre_launch_ && !pre_launch_(cs_, err)) return false;
    if (!stage_batch(rows, S, 0, tokens, pos, err)) return false;
    const cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(rows, S, 0)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    // #646: a stage whose every expert is resident plans on the device and raises no host doorbells, so the
    // per-layer spin below has nothing to wait for: without the PLE flag the graph finishes with the ring silent
    // ("verify batch: layer K never rang (graph finished)" - K is that stage's first layer), and with it the graph
    // sits on the PLE wait until the 20 s timeout.  As run() does: raise the PLE flag the graph's first wait reads,
    // let the graph run to the end, and skip the host's per-layer service (there is nothing to serve).
    // #646 + #871: the skip is per-window (ar_on), not per-process (all_resident_): a stage that was 100%
    // resident at init takes the doorbell graph while a prompt loan/shrink/swap is in flight (ar_off_), and that
    // graph waits on host doorbells the skipped service never raises (GPU at 100%, host in the sync below).
    if (ar_on()) {
        if (ss_->ple.ready() && ple_stage()) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
        }
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
        cudaStreamSynchronize(copy_);
        if (prof_on_) collect_profile();
        ++windows;
        if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_rows(rows, S, tokens, pos, pool, next_user_, out, err);
        if (!sample_rows(S, err)) return false;
        for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
        progress_at("decode");
        progress_beat();
        return true;
    }
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        progress_at("verify batch: the CPU experts of layer", l);
        // The recorded graph consumes the helper mask and weighted sum just as
        // the solo graph does. Reset both and supply this layer's routing weights.
        if (remote_opt_) remote_opt_->begin(h_w_, 0, S);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l);
        if (remote_opt_) remote_opt_->end();
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
    cudaStreamSynchronize(copy_);
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_rows(rows, S, tokens, pos, pool, next_user_, out, err);
    if (!sample_rows(S, err)) return false;
    for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(S, last_pos_b_);
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) {
    std::vector<int> keep(slots_.size(), 0);
    for (int t = 0; t < last_t_; ++t) ++keep[(size_t) last_rows_[t]];
    return commit_slot_prefixes(keep.data(), err);
}

bool Verifier::commit_slot_prefixes(const int* keep, std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    const int S = last_t_;
    const Clock::time_point t0 = Clock::now();
    const int64_t CB = 2 + max_t_;
    // One prefix per slot: rejected draft rows must not enter recurrent state.
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && last_rows_[t] == last_rows_[first]) ++t;
        const int n = keep[last_rows_[first]];
        if (n < 1 || n > t - first) {
            err = "verify: accepted prefix is outside its slot group";
            return false;
        }
        int32_t* c = h_commitb_ + (size_t) last_rows_[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int j = 0; j < max_t_; ++j)
            c[2 + j] = j < n ? (int32_t) last_pos_b_[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!batch_self_commits(last_rows_, S)) {   // (else the window has kept its rows)
        const cudaError_t le = cudaGraphLaunch(commit_bm_[bkey(last_rows_, S, row_base_)], cs_);
        if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    }
    if (ple_stage())
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && last_rows_[t] == last_rows_[first]) ++t;
            SessionState& sx = *slots_[(size_t) last_rows_[first]];
            for (int u = first; u < first + keep[last_rows_[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[u];
            }
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slot_prefixes(keep, err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        strata::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        strata::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (pre_launch_ && !pre_launch_(cs_, err)) return false;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(rows, S, base)], cs_);
    if (le == cudaSuccess && !batch_self_commits(rows, S))   // (else the window has kept its rows itself)
        le = cudaGraphLaunch(commit_bm_[bkey(rows, S, base)], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    ring_now();
    b_running_.store(true, std::memory_order_release);   // last: an external server reads the rest after it
    return true;
}

int Verifier::batch_serve(PoolMultiFn pool, void* user) {
    if (!b_running_.load(std::memory_order_acquire)) return 1;
    volatile uint32_t* const seq = h_seq_;
    // #646: an all-resident stage's graph raises no host doorbells (see run_slot_rows): nothing to serve per layer,
    // except the PLE flag, which the graph's first wait reads and only the host can raise (the same raise
    // run_slot_rows makes).  Per-window (ar_on): with a loan in flight the doorbell graph needs the per-layer service.
    if (ar_on() && ss_->ple.ready() && ple_stage()) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *h_flag_ = 1;
    }
    for (;;) {
        // the window's row count is read after the served-layer count: with external serving the window this call
        // began on can have finished and the next one been launched meanwhile, and batch_launch writes its rows
        // (stage_batch) before it resets b_k_, so a count that is the new window's comes with the new rows
        const int64_t k = b_k_.load(std::memory_order_acquire);
        if (ar_on() || k >= b_steps_) break;
        const uint32_t want = (uint32_t) (k + 1);
        if (*seq < want) return 0;
        const int S = last_t_;
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        if (remote_opt_) remote_opt_->begin(h_w_, 0, S);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + k);
        if (remote_opt_) remote_opt_->end();
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        ring_now();
        b_k_.fetch_add(1, std::memory_order_release);
    }
    return 1;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    // set_external_serve: another thread serves the doorbells (batch_serve) and this one only checks for a window that
    // will not finish, as below: its graph ended (or failed) with a layer still to ring, or no layer rang for 20 s.
    // The stream query can wait for the driver's lock while a read's call holds it; the doorbells go on meanwhile.
    // The served count is read before the query and the ring count after it: a layer served in between rang first.
    if (ext_serve_.load(std::memory_order_acquire) && !ar_on()) {
        const int64_t k = b_k_.load(std::memory_order_acquire);
        if (k < b_steps_) {
            const auto now = Clock::now();
            if (now - last_ring() <= std::chrono::milliseconds(2) || now - ext_checked_ < std::chrono::milliseconds(1))
                return 0;
            ext_checked_ = now;   // (at most one query a millisecond: pump calls this in a loop)
            const OnDevice on_device(device_);
            const cudaError_t q = cudaStreamQuery(cs_);
            if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < (uint32_t) (k + 1)) {
                err = "verify batch: layer " + std::to_string(lb_ + k) + " never rang (" +
                      (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                b_running_ = false;
                return -1;
            }
            if (now - last_ring() > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(lb_ + k) +
                      released_note(release_gpu_waits(5000));
                b_running_ = false;
                return -1;
            }
            return 0;
        }
    }
    const OnDevice on_device(device_);
    const int S = last_t_;
    // the layers that have rung so far (batch_serve); with one still to ring: did the graph finish without it, or
    // has it waited too long?
    if (!batch_serve(pool, user)) {
        const auto now = Clock::now();
        if (now - last_ring() > std::chrono::milliseconds(2)) {
            const cudaError_t q = cudaStreamQuery(cs_);
            if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < (uint32_t) (b_k_ + 1)) {
                err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                      (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                b_running_ = false;
                return -1;
            }
            if (now - last_ring() > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) +
                      released_note(release_gpu_waits(5000));
                b_running_ = false;
                return -1;
            }
        }
        return 0;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    const cudaError_t qc = cudaStreamQuery(copy_);   // no host function of this window may raise flag B in the next
    if (qc == cudaErrorNotReady) return 0;
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
        note_nonfinite(S, last_pos_b_);
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

// #879: STRATA_NAN_GUARD=0 turns the guard off (no kernel is recorded; the graphs are captured once, so it is read once)
bool Verifier::nan_guard_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_NAN_GUARD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

namespace {
// STRATA_NAN_INJECT=<n>[,<n>...] (debug): the windows (counted from 1 over the process) the guard reports as non-finite
const std::vector<long long>& nan_inject_list() {
    static const std::vector<long long> list = [] {
        std::vector<long long> v;
        if (const char* e = std::getenv("STRATA_NAN_INJECT"))
            for (const char* q = e; *q != 0;) {
                char* end = nullptr;
                const long long n = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (n > 0) v.push_back(n);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return list;
}
std::atomic<long long> g_nf_windows{0};
}  // namespace

void Verifier::note_nonfinite(int T, const int64_t* pos_b) {
    nf_row_ = -1;
    nf_pos_ = -1;
    nf_mask_ = 0;
    if (!nan_guard_on() || h_nf_ == nullptr || !last_stage()) return;   // (a stage without the head checks nothing)
    const volatile uint32_t* f = h_nf_;
    for (int t = 0; t < T && t < strata::kernels::kVerifyMaxT; ++t)
        for (int b = 0; b < strata::kernels::kNonfiniteBlocks; ++b)
            if (f[t * strata::kernels::kNonfiniteBlocks + b] != 0) { nf_mask_ |= 1u << t; break; }
    const std::vector<long long>& inject = nan_inject_list();
    if (!inject.empty()) {
        const long long k = g_nf_windows.fetch_add(1) + 1;
        if (nf_mask_ == 0 && std::find(inject.begin(), inject.end(), k) != inject.end()) {
            nf_mask_ = 1u;
            std::fprintf(stderr, "strata verify: STRATA_NAN_INJECT: window %lld reported as non-finite\n", k);
        }
    }
    for (int t = 0; t < T && nf_row_ < 0; ++t)
        if ((nf_mask_ >> t) & 1u) nf_row_ = t;
    if (nf_row_ >= 0) nf_pos_ = pos_b != nullptr ? pos_b[nf_row_] : last_pos0_ + nf_row_;
}

bool Verifier::copy_logits(int t, float* host) const {
#if defined(STRATA_GPU_METAL)
    return strata::metal_engine::window_copy_logits(metal_, t, host);
#endif
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}


// ================================ PIPELINED WINDOWS (see verify.hpp) ================================
//
// The window is run()'s, step for step: the same graph, the same staging, the same per-layer service (the PLE rows
// gathered while layer 0 is served, the zero-doorbell graph's one flag), the same commit graph.  Only the waits are
// split up: the host polls instead of spinning, so it can serve the other stage's window in between.

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    auto ev = [](cudaEvent_t e) {
        if (e == nullptr) return "none";
        const cudaError_t q = cudaEventQuery(e);
        return q == cudaSuccess ? "done" : q == cudaErrorNotReady ? "PENDING" : "error";
    };
    std::fprintf(f, "  %s: %s T=%d pos %lld served %lld/%lld; GPU rang %u, flags served %u A %u B %u; window event %s, "
                    "commit event %s (commit launched %d)\n", name, fl_active_ ? "IN FLIGHT" : "idle", last_t_,
                 (long long) last_pos0_, (long long) fl_k_, (long long) fl_total_, rd(h_seq_), rd(h_flag_), rd(h_flagA_),
                 rd(h_flagB_), ev(ev_done_), ev(ev_commit_), (int) commit_live_);
}

bool Verifier::capture_all(std::string& err) {
    const OnDevice on_device(device_);
    // every size 1..max_t is captured here and all of them are needed: making room for one by freeing another's graph
    // would only move the failure one size further (and leave it uncaptured for the pipelined path)
    struct NoEvict {
        bool& flag;
        bool prev;
        explicit NoEvict(bool& f) : flag(f), prev(f) { flag = true; }
        ~NoEvict() { flag = prev; }
    } no_evict(no_window_eviction_);
    if (g_ == nullptr) { err = "verify: capture_all before init"; return false; }
    if (remote_opt_ != nullptr) { err = "verify: pipelined windows do not serve --remote-expert-opt"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (!capture_commit(err)) return false;
    if ((ev_done_ == nullptr && cudaEventCreateWithFlags(&ev_done_, cudaEventDisableTiming) != cudaSuccess) ||
        (ev_commit_ == nullptr && cudaEventCreateWithFlags(&ev_commit_, cudaEventDisableTiming) != cudaSuccess)) {
        err = "verify: event create failed";
        return false;
    }
    if (prof_on_ && prof_pin_ == nullptr &&
        cudaHostAlloc((void**) &prof_pin_, prof_h_.size() * 8, cudaHostAllocDefault) != cudaSuccess) {
        prof_pin_ = nullptr;   // the pipelined windows go unprofiled
        cudaGetLastError();
    }
    pl_ple_rows_.assign((size_t) strata::kernels::kVerifyMaxT * strata::kernels::PLE_N_HEADS, 0u);
    return true;
}

// The host staging of a pipelined window: run()'s (stage_inputs), and its PLE rows from `ple_prev` with their pages
// prefetched (they are gathered into the mapped rows when layer 0 is served, as run() does).
void Verifier::pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]) {
    using namespace strata::kernels;
    stage_inputs(T, tokens, pos0);
    staged_ = false;   // a later run() stages its own window
    SessionState& ss = *ss_;
    fl_ple_ = ss.ple.ready() && ple_stage();
    if (fl_ple_) {
        int32_t prev[2] = {ple_prev[0], ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, pl_ple_rows_.data() + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(pl_ple_rows_.data() + t * PLE_N_HEADS);
        }
    }
    pl_prev_[0] = ple_prev[0];
    pl_prev_[1] = ple_prev[1];
}

bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (pl_ple_rows_.empty()) { err = "verify: pipelined window not prepared (capture_all)"; return false; }
    const Clock::time_point t0 = Clock::now();
    pl_stage(T, tokens, pos0, ple_prev);
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    const OnDevice on_device(device_);
    if (fl_active_) { err = "verify: a window is already in flight on this verifier"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (exec_[T] == nullptr || commit_exec_ == nullptr || ev_done_ == nullptr || pl_ple_rows_.empty()) {
        err = "verify: pipelined window not prepared (capture_all)";
        return false;
    }
    if (pre_launch_ && !pre_launch_(cs_, err)) return false;
    const Clock::time_point t0 = Clock::now();
    last_batch_ = false;
    bool staged = pl_prestaged_ && last_t_ == T && last_pos0_ == pos0;
    for (int t = 0; staged && t < T; ++t) staged = last_tokens_[t] == tokens[t];
    if (staged && ss.ple.ready() && ple_stage())
        staged = pl_prev_[0] == ss.ple_prev[0] && pl_prev_[1] == ss.ple_prev[1];
    pl_prestaged_ = false;
    if (!staged) pl_stage(T, tokens, pos0, ss.ple_prev);
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    fl_T_ = T;
    fl_k_ = 0;
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    if (fl_prof_) cudaMemcpyAsync(prof_pin_, prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost, cs_);
    if (cudaEventRecord(ev_done_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
    (void) cudaStreamQuery(cs_);   // WDDM: submit now
    fl_active_ = true;
    pcie_drained_ = false;   // the replayed graph restamps every pair: this window's read-back is still to come
    fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
    return true;
}

int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
    if (!fl_active_) return 1;
    if (fl_k_ >= fl_total_) return 1;
    // Nothing rang yet and neither the WDDM flush nor the timeout below is due: return before any CUDA call.  The
    // pipelined loop asks both stages' windows this millions of times a second while they wait on their doorbells,
    // and the device switch (OnDevice: cudaGetDevice and two cudaSetDevice) made that a few driver calls each time:
    // 919 million passes before the host stalled inside one of those cudaSetDevice calls with both stages waiting
    // on it (0.1.41, Windows, an RTX 3060 + RTX 5070 Ti split).  The same answer as the loop below gives.
    if (!ar_on() && *(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
        const double now = now_ms();
        if (now - fl_flush_ms_ <= 2.0 && now - fl_since_ms_ <= 20000.0) return 0;
    }
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = fl_T_;
    // the PLE rows into the mapped staging (layer 1's pre() copies them once flag 1 is up), as run() does at k == 0
    auto gather_ple = [&]() -> bool {
        if (!fl_ple_) return true;
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(pl_ple_rows_.data(), (size_t) T, h_ple_, err)) return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        ms_host += ms_since(tp);
        return true;
    };
    if (ar_on()) {   // the zero-doorbell graph never rings: it waits only for flag 1 (stage 0's PLE rows).
        // Per-window: with a loan in flight (ar_off_) the doorbell graph needs the per-layer service below.
        if (!gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = 1;
        fl_k_ = fl_total_;
        progress_tick();
        return 1;
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    while (fl_k_ < fl_total_) {
        const int64_t l = lb_ + fl_k_ / G;
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (*(volatile uint32_t*) h_seq_ < want) {
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                fl_flush_ms_ = now;
                const cudaError_t q = cudaEventQuery(ev_done_);
                if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < want) {
                    trace_ev("NEVER-RANG", fl_k_, l, (int64_t) q);
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return -1;
                }
                (void) cudaStreamQuery(cs_);
            }
            if (now - fl_since_ms_ > 20000.0) {   // #267: no spin kernel may outlive the engine
                trace_ev("TIMEOUT", fl_k_, l, 0);
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return -1;
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        progress_at("verify window (pipelined): the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", fl_k_, l,
                              (int64_t) ms_since(b));
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        if (fl_k_ == 0 && !gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = want;
        ++fl_k_;
        ms_pool += ms_since(b);
        fl_since_ms_ = fl_flush_ms_ = now_ms();
    }
    return 1;
}

bool Verifier::done(std::string& err) {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    const cudaError_t q = cudaEventQuery(ev_done_);
    if (q == cudaErrorNotReady) {
        const double now = now_ms();
        if (now - fl_flush_ms_ > 2.0) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
        if (now - fl_since_ms_ > 20000.0) {   // every layer served, and the graph still runs: #267 as above
            err = "verify: the window never finished" + released_note(release_gpu_waits(5000));
        }
        return false;
    }
    if (q != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(q); return false; }
    if (copy_used_) {   // no host function of this window may raise flag B in the next one
        if (cudaStreamQuery(copy_) == cudaErrorNotReady) return false;
        copy_used_ = false;
    }
    if (!pcie_drained_) {   // ev_done_ says the window's graph finished: its H2D stamps are final (STRATA_PCIE_TIMING)
        drain_pcie_timing();
        pcie_drained_ = true;
    }
    return true;
}

bool Verifier::pl_finish(int32_t* out, std::string& err) {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    trace_ev("DONE (pipelined)", -1, -1, (int64_t) (now_ms() - fl_launch_ms_));
    if (le_ < g.n_layers) return true;   // an earlier stage: the hand-off is written
    const int T = fl_T_;
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    note_nonfinite(T, nullptr);
    progress_beat();
    return true;
}

bool Verifier::pl_commit_async(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    if (commit_exec_ == nullptr || ev_commit_ == nullptr) { err = "verify: pipelined commit not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (commit_live_) {   // its graph reads the words below when it starts: the previous one has (a window ago)
        cudaError_t q;
        while ((q = cudaEventQuery(ev_commit_)) == cudaErrorNotReady) _mm_pause();
        if (q != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(q); return false; }
    }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    if (cudaEventRecord(ev_commit_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
    (void) cudaStreamQuery(cs_);
    commit_live_ = true;
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return true;
}

void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    ms_pcie += o.ms_pcie;   // the report sums both parities until this absorbs one into the other: keep it exact
    windows += o.windows;
    o.ms_wait = o.ms_pool = o.ms_host = o.ms_commit = 0;
    o.ms_pcie = 0;
    o.windows = 0;
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < kProfPer; ++i) { prof_sum_[k][i] += o.prof_sum_[k][i]; o.prof_sum_[k][i] = 0; }
    prof_windows_ += o.prof_windows_;
    o.prof_windows_ = 0;
}

}  // namespace strata::core
