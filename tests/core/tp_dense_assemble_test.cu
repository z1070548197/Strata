// tests/core/tp_dense_assemble_test.cu - `--tp` long-context prefill: the prompt path must read the FULL dense
// matrix, but under `--tp` each card uploaded only ITS half of every split Q8_0 projection (NativeDense::load with
// tp_rank).  Reading the full n_in/n_out out of a half matrix walks off the end of the allocation - that is the
// illegal memory access that took the two-card prompt path down at 8000 tokens.
//
// The fix assembles the full matrix in the GEMM scratch: rows [0, n_own) from this card's half, rows
// [n_own, N) from the peer's, both dequantized in row order, then ONE GEMM over the full N - the call the single
// card makes.  This test proves both halves of that claim on the REAL pack tensors:
//
//   A. the row mapping (tp_peer_rows_for) reassembles the original full tensor byte for byte - the two ranges
//      cover the full matrix exactly, in the full matrix's own row order;
//   B. Gemm::native over the assembled halves is bit-identical to Gemm::native over the whole matrix - through
//      the whole-matrix path AND through the row-sliced path (a scratch too small for N x K), so both straddle
//      the two halves and both must produce every output element identically.
//
// Row mode is refused loudly (no row range exists: the halves are partial sums), and that refusal is asserted.
#include "strata/core/tp_split_plan.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fails = 0;
void req(bool ok, const char* what, const std::string& extra = {}) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s%s%s\n", what, extra.empty() ? "" : ": ", extra.c_str());
        ++fails;
    }
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        ++fails;
    }
}

std::vector<uint8_t> load_q8(const char* path, int n_in, int n_out) {
    std::FILE* f = std::fopen(path, "rb");
    req(f != nullptr, "open", path);
    if (f == nullptr) return {};
    const size_t want = (size_t) (n_in / 32) * 34u * (size_t) n_out;
    std::vector<uint8_t> v(want);
    const size_t got = std::fread(v.data(), 1, want, f);
    std::fclose(f);
    req(got == want, "short read", path);
    return v;
}

// deterministic FP16 activations
std::vector<uint16_t> make_x(int64_t T, int64_t K) {
    std::vector<uint16_t> x((size_t) (T * K));
    unsigned s = 12345u;
    for (auto& v : x) {
        s = s * 1664525u + 1013904223u;
        const float f = ((float) ((s >> 9) & 0x3ff) / 1023.0f) - 0.5f;
        // fp16 bits of a small magnitude float
        uint32_t b = 0;
        std::memcpy(&b, &f, 4);
        const uint32_t sign = (b >> 16) & 0x8000u;
        int32_t e = (int32_t) ((b >> 23) & 0xff) - 127 + 15;
        if (e < 1) {
            v = (uint16_t) sign;
            continue;
        }
        if (e > 30) e = 30;
        v = (uint16_t) (sign | ((uint32_t) e << 10) | ((b >> 13) & 0x3ffu));
    }
    return x;
}

struct Case {
    const char* name;        // canonical tensor name (tp_split_for's key)
    const char* path;        // fixture
    int n_in;
    int n_out;               // FULL n_out
    const char* kind;        // Qkv / Col
};

// Part A: the mapping reassembles the original bytes exactly.
void check_mapping(const Case& c, const std::vector<uint8_t>& full) {
    strata::core::TpPeerRows pr;
    std::string why;
    req(strata::core::tp_peer_rows_for(c.name, c.n_out, pr, &why), "tp_peer_rows_for",
        std::string(c.name) + ": " + why);
    req(pr.n_own + pr.rows == c.n_out, "the two ranges do not cover the full matrix", c.name);
    if (pr.n_own + pr.rows != c.n_out) return;

    const size_t row_bytes = (size_t) (c.n_in / 32) * 34u;
    // rank 0's own half, then the peer's rows [src_row0, src_row0 + rows)
    std::vector<uint8_t> own(strata::core::tp_q8_0_bytes(c.name, 0));
    std::vector<uint8_t> peer(strata::core::tp_q8_0_bytes(c.name, 1));
    const strata::core::TpSplit s0 = strata::core::tp_split_for(c.name, 0);
    req(strata::core::tp_split_q8_0_host(full.data(), s0.mode, c.n_in, c.n_out, 0, own.data()) == own.size(),
        "rank 0 host split", c.name);
    req(strata::core::tp_split_q8_0_host(full.data(), s0.mode, c.n_in, c.n_out, 1, peer.data()) == peer.size(),
        "rank 1 host split", c.name);

    std::vector<uint8_t> rebuilt((size_t) c.n_out * row_bytes);
    std::memcpy(rebuilt.data(), own.data(), (size_t) pr.n_own * row_bytes);
    std::memcpy(rebuilt.data() + (size_t) pr.n_own * row_bytes, peer.data() + (size_t) pr.src_row0 * row_bytes,
                (size_t) pr.rows * row_bytes);
    req(rebuilt.size() == full.size() && std::memcmp(rebuilt.data(), full.data(), full.size()) == 0,
        "the mapping does not reassemble the original tensor", c.name);
    std::printf("  A %-28s %s: rows [0,%lld) own + [%lld,%lld) peer(%lld..) == the full tensor, byte for byte\n",
                c.name, c.kind, (long long) pr.n_own, (long long) pr.n_own, (long long) c.n_out,
                (long long) pr.src_row0);
}

// Part B: the GEMM over the assembled halves == the GEMM over the whole matrix, bit for bit.
void check_gemm(const Case& c, const std::vector<uint8_t>& full, int dev0, int dev1, bool p2p, int64_t T,
                int64_t scratch_elems, const char* path_desc) {
    const size_t row_bytes = (size_t) (c.n_in / 32) * 34u;
    const strata::core::TpSplit s0 = strata::core::tp_split_for(c.name, 0);
    std::vector<uint8_t> own(strata::core::tp_q8_0_bytes(c.name, 0));
    std::vector<uint8_t> peer(strata::core::tp_q8_0_bytes(c.name, 1));
    strata::core::tp_split_q8_0_host(full.data(), s0.mode, c.n_in, c.n_out, 0, own.data());
    strata::core::tp_split_q8_0_host(full.data(), s0.mode, c.n_in, c.n_out, 1, peer.data());

    std::vector<uint16_t> x = make_x(T, c.n_in);

    void* d_full = nullptr;
    void* d_own = nullptr;
    void* d_peer = nullptr;
    void* d_x = nullptr;
    float* d_yref = nullptr;
    float* d_yasm = nullptr;

    ck(cudaSetDevice(dev0), "set dev0");
    ck(cudaMalloc(&d_full, full.size()), "malloc full");
    ck(cudaMalloc(&d_own, own.size()), "malloc own");
    ck(cudaMalloc(&d_x, x.size() * 2), "malloc x");
    ck(cudaMalloc(&d_yref, (size_t) (T * c.n_out) * 4), "malloc yref");
    ck(cudaMalloc(&d_yasm, (size_t) (T * c.n_out) * 4), "malloc yasm");
    ck(cudaMemcpy(d_full, full.data(), full.size(), cudaMemcpyHostToDevice), "h2d full");
    ck(cudaMemcpy(d_own, own.data(), own.size(), cudaMemcpyHostToDevice), "h2d own");
    ck(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "h2d x");
    if (p2p) {
        // the peer's half on the OTHER card, read over P2P exactly as the engine does
        ck(cudaSetDevice(dev1), "set dev1");
        ck(cudaMalloc(&d_peer, peer.size()), "malloc peer");
        ck(cudaMemcpy(d_peer, peer.data(), peer.size(), cudaMemcpyHostToDevice), "h2d peer");
        ck(cudaSetDevice(dev0), "back to dev0");
    } else {
        ck(cudaMalloc(&d_peer, peer.size()), "malloc peer same-dev");
        ck(cudaMemcpy(d_peer, peer.data(), peer.size(), cudaMemcpyHostToDevice), "h2d peer same-dev");
    }

    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    strata::prefill::Gemm gm;
    std::string err;
    req(gm.init(st, scratch_elems, err), "Gemm::init", err);

    // reference: the whole matrix, the single card's call
    gm.native((const uint16_t*) d_x, 8, d_full, d_yref, T, c.n_out, c.n_in, 0, 0.0f, 0, nullptr);
    // under test: this card's half + the peer's, assembled in the scratch
    strata::core::TpPeerRows rows;
    req(strata::core::tp_peer_rows_for(c.name, c.n_out, rows, &err), "tp_peer_rows_for", err);
    strata::prefill::Gemm::PeerRows pr;
    pr.blocks = d_peer;
    pr.src_row0 = rows.src_row0;
    pr.n_own = rows.n_own;
    gm.native((const uint16_t*) d_x, 8, d_own, d_yasm, T, c.n_out, c.n_in, 0, 0.0f, 0, &pr);
    ck(cudaStreamSynchronize(st), "sync");
    const cudaError_t le = cudaGetLastError();
    ck(le, "the assembled GEMM launched");

    std::vector<float> yref((size_t) (T * c.n_out)), yasm((size_t) (T * c.n_out));
    ck(cudaMemcpy(yref.data(), d_yref, yref.size() * 4, cudaMemcpyDeviceToHost), "d2h yref");
    ck(cudaMemcpy(yasm.data(), d_yasm, yasm.size() * 4, cudaMemcpyDeviceToHost), "d2h yasm");

    int bad = 0;
    int64_t first = -1;
    for (size_t i = 0; i < yref.size(); ++i) {
        if (std::memcmp(&yref[i], &yasm[i], 4) != 0) {
            if (bad < 3) {
                std::fprintf(stderr, "  %s %s idx %zu: whole %g assembled %g\n", c.name, path_desc, i, yref[i],
                             yasm[i]);
            }
            if (first < 0) first = (int64_t) i;
            ++bad;
        }
    }
    req(bad == 0, "the assembled GEMM is not bit-identical to the whole-matrix GEMM",
        std::string(c.name) + " " + path_desc);

    cudaFree(d_full);
    cudaFree(d_own);
    cudaFree(d_peer);
    cudaFree(d_x);
    cudaFree(d_yref);
    cudaFree(d_yasm);
    cudaStreamDestroy(st);
    std::printf("  B %-28s T=%-5lld scratch %-6lld %-22s %s (%d/%zu differ)\n", c.name, (long long) T,
                (long long) scratch_elems, path_desc, bad == 0 ? "bit-identical" : "MISMATCH", bad, yref.size());
    (void) row_bytes;
    (void) first;
}

}  // namespace

int main() {
    const Case cases[] = {
        {"blk.0.attn_qkv.weight", "tests/core/tp_data/blk_0_attn_qkv_weight.q8_0", 2560, 10240, "Qkv"},
        {"blk.0.ssm_out.weight", "tests/core/tp_data/blk_0_ssm_out_weight.q8_0", 6144, 2560, "Col"},
    };

    std::printf("=== --tp dense assembly: the prompt path's half x2 -> full ===\n");

    // Row mode has no row range to assemble - asserted both ways round.
    {
        strata::core::TpPeerRows pr;
        std::string why;
        // attn_output is 6144 x 2560: n_in 6144, n_out 2560, a plain column split
        const bool ok = strata::core::tp_peer_rows_for("blk.0.attn_output.weight", 2560, pr, &why);
        const strata::core::TpSplit s = strata::core::tp_split_for("blk.0.attn_output.weight", 0);
        if (s.mode == strata::core::TpMode::Row) {
            req(!ok, "a Row-mode tensor must not produce a peer row range");
        } else {
            req(ok && pr.n_own + pr.rows == 2560, "attn_output is a plain column split", why);
        }
        req(!strata::core::tp_peer_rows_for("blk.0.ffn_gate_shexp.weight", 640, pr, &why),
            "a replicated tensor must not produce a peer row range");
        req(!strata::core::tp_peer_rows_for("blk.0.attn_qkv.weight", 4096, pr, &why),
            "a wrong n_out must be refused (it is not the full matrix's)");
        // A Row-mode tensor, if the plan ever makes one of these Row, must be refused rather than read as rows.
        for (const char* n : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight"}) {
            const strata::core::TpSplit sp = strata::core::tp_split_for(n, 0);
            if (sp.mode != strata::core::TpMode::Row) continue;
            req(!strata::core::tp_peer_rows_for(n, sp.full_n_out, pr, &why),
                "a Row-mode tensor must not produce a peer row range", n);
        }
    }

    std::vector<std::vector<uint8_t>> full;
    for (const Case& c : cases) full.push_back(load_q8(c.path, c.n_in, c.n_out));
    if (fails) { std::fprintf(stderr, "FAIL: fixtures\n"); return 1; }

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) check_mapping(cases[i], full[i]);

    int n_dev = 0;
    ck(cudaGetDeviceCount(&n_dev), "device count");
    if (n_dev < 1) {
        std::fprintf(stderr, "FAIL: no CUDA device\n");
        return 1;
    }
    int dev0 = 0, dev1 = 0;
    bool p2p = false;
    if (n_dev >= 2) {
        dev1 = 1;
        int can = 0;
        ck(cudaSetDevice(dev0), "set dev0");
        if (cudaDeviceCanAccessPeer(&can, dev0, dev1) == cudaSuccess && can) {
            cudaError_t e = cudaDeviceEnablePeerAccess(dev1, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) e = cudaSuccess;
            ck(e, "enable peer access");
            p2p = (e == cudaSuccess);
        }
        std::printf("  peer tier: %s\n", p2p ? "the peer's half on CUDA1, read over P2P" : "no P2P - both halves on CUDA0");
    } else {
        std::printf("  one device: both halves on CUDA0 (the row assembly is what is under test)\n");
    }

    constexpr int64_t T = 64;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const Case& c = cases[i];
        const int64_t nk = (int64_t) c.n_in * c.n_out;
        // the whole-matrix path (the scratch holds all of N x K - the engine's GEMM_SCRATCH case)
        check_gemm(c, full[i], dev0, dev1, p2p, T, nk, "whole matrix");
        // the row-sliced path: a scratch that holds only 1/3 of the rows, so every slice but the last is a
        // pure-half range and the straddle lands mid-loop
        check_gemm(c, full[i], dev0, dev1, p2p, T, (nk / 3 / 64 + 1) * 64, "row slices");
    }

    if (fails) {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", fails);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
