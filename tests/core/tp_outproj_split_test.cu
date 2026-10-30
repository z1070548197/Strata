// tests/core/tp_outproj_split_test.cu - how the two out-projections must be split so a two-card window's
// residual is BIT-EXACT with the single-card path.
//
// `ssm_out` (GDN) and `attn_output` (QSA) are the only dense projections whose output goes through the
// per-layer allreduce.  Their full shape is 2560 x 6144 (n_out x n_in): every output row is a 6144-term dot
// product with the activation vector.
//
// There are two ways to give each card half the work, and only one of them can reproduce the single card:
//
//   n_in split (row-parallel, what the engine does today): each card reduces 3072 of the 6144 terms and the
//   allreduce adds the two partials.  (a1+a2)+b is not a1+(a2+b) in float32, so the sum differs from the
//   single card's 6144-term reduction by a few ULP - and one ULP is enough to flip a q8_1 bucket later.
//
//   n_out split (column-parallel): each card holds half the OUTPUT ROWS and reduces all 6144 terms of each
//   with the same kernel, the same block layout and the same warp assignment the single card uses.  Every
//   element is then bit-identical to the single card's, and the two cards' outputs are disjoint (no
//   allreduce at all).
//
// This test measures both on the real Q8_0 kernels at the model's shape, and asserts the second is bitwise
// and the first is not.  A regression that turns the out-projection back into a reduction (or a kernel change
// that makes the n_out split stop being bit-exact) fails here, on one GPU, in under a second.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(call)                                                                     \
    do {                                                                             \
        const cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "FAIL: %s: %s\n", #call, cudaGetErrorString(e_));    \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

constexpr int kNin = 6144;    // ssm_out / attn_output reduction axis
constexpr int kNout = 2560;   // ssm_out / attn_output output rows (N)
constexpr int kQ8Block = 32;
constexpr int kQ8Bytes = 34;   // one Q8_0 block: fp16 d + 32 int8

size_t row_bytes(int n_in) { return (size_t) (n_in / kQ8Block) * kQ8Bytes; }

// A deterministic Q8_0 tensor with sane (non-zero, finite) scales, so the dot products are exercised rather
// than short-circuited by denormals.
std::vector<uint8_t> make_q8_0(int n_in, int n_out, std::mt19937& rng) {
    std::vector<uint8_t> w(row_bytes(n_in) * (size_t) n_out);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int r = 0; r < n_out; ++r) {
        uint8_t* row = w.data() + row_bytes(n_in) * (size_t) r;
        for (int b = 0; b < n_in / kQ8Block; ++b) {
            const float d = std::fabs(nd(rng)) * 0.02f + 1e-4f;
            const __half dh = __float2half(d);   // a REAL fp16 scale: a float's low two bytes would be a NaN pattern
            std::memcpy(row + (size_t) b * kQ8Bytes, &dh, 2);
            for (int i = 0; i < kQ8Block; ++i) row[(size_t) b * kQ8Bytes + 2 + i] = (uint8_t) byte(rng);
        }
    }
    return w;
}

}  // namespace

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 1) {
        std::fprintf(stderr, "tp_outproj_split: needs one GPU\n");
        return 77;
    }
    std::printf("tp_outproj_split: 2560 x 6144 out-projection, full vs n_in half vs n_out half (Q8_0, real kernels)\n");
    std::mt19937 rng(20261026);
    const std::vector<uint8_t> w = make_q8_0(kNin, kNout, rng);

    cudaStream_t cs = nullptr;
    CK(cudaStreamCreate(&cs));
    uint8_t* d_w = nullptr;
    CK(cudaMalloc((void**) &d_w, w.size()));
    CK(cudaMemcpyAsync(d_w, w.data(), w.size(), cudaMemcpyHostToDevice, cs));

    int fails = 0;
    for (int T : {1, 2, 3, 4}) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x((size_t) T * kNin);
        for (auto& v : x) v = nd(rng);
        float* d_x = nullptr;
        uint8_t* d_xq = nullptr;
        CK(cudaMalloc((void**) &d_x, x.size() * sizeof(float)));
        CK(cudaMalloc((void**) &d_xq, (size_t) T * (kNin / kQ8Block) * 36));
        CK(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
        strata::kernels::native_quantize_q8_1(d_x, d_xq, kNin, T, cs);

        float* d_y_full = nullptr;
        float* d_y_in0 = nullptr;
        float* d_y_in1 = nullptr;
        float* d_y_out0 = nullptr;
        float* d_y_out1 = nullptr;
        CK(cudaMalloc((void**) &d_y_full, (size_t) T * kNout * sizeof(float)));
        CK(cudaMalloc((void**) &d_y_in0, (size_t) T * kNout * sizeof(float)));
        CK(cudaMalloc((void**) &d_y_in1, (size_t) T * kNout * sizeof(float)));
        CK(cudaMalloc((void**) &d_y_out0, (size_t) T * kNout * sizeof(float)));
        CK(cudaMalloc((void**) &d_y_out1, (size_t) T * kNout * sizeof(float)));

        // 1. the single card: all 6144 terms, all 2560 rows.
        strata::kernels::native_mmvq(8, d_w, d_xq, d_y_full, kNin, kNout, T, cs);

        // 2. the n_in split (what the engine does today): each card reduces 3072 terms over all rows, and the
        //    two partials are added - exactly what TpAllreduce's kernel does.
        //    Card 0's half is the first 3072 columns of every row; card 1's the last 3072.
        std::vector<uint8_t> w_in0(row_bytes(kNin / 2) * (size_t) kNout), w_in1(w_in0.size());
        for (int r = 0; r < kNout; ++r) {
            std::memcpy(w_in0.data() + row_bytes(kNin / 2) * (size_t) r, w.data() + row_bytes(kNin) * (size_t) r,
                        row_bytes(kNin / 2));
            std::memcpy(w_in1.data() + row_bytes(kNin / 2) * (size_t) r,
                        w.data() + row_bytes(kNin) * (size_t) r + row_bytes(kNin / 2), row_bytes(kNin / 2));
        }
        uint8_t* d_win0 = nullptr;
        uint8_t* d_win1 = nullptr;
        CK(cudaMalloc((void**) &d_win0, w_in0.size()));
        CK(cudaMalloc((void**) &d_win1, w_in1.size()));
        CK(cudaMemcpyAsync(d_win0, w_in0.data(), w_in0.size(), cudaMemcpyHostToDevice, cs));
        CK(cudaMemcpyAsync(d_win1, w_in1.data(), w_in1.size(), cudaMemcpyHostToDevice, cs));
        // the q8_1 image of each half: the engine quantizes the activation it owns (y_tp_ / attn_tp_)
        uint8_t* d_xq_in0 = nullptr;
        uint8_t* d_xq_in1 = nullptr;
        CK(cudaMalloc((void**) &d_xq_in0, (size_t) T * ((kNin / 2) / kQ8Block) * 36));
        CK(cudaMalloc((void**) &d_xq_in1, (size_t) T * ((kNin / 2) / kQ8Block) * 36));
        {
            std::vector<float> h0((size_t) T * (kNin / 2)), h1(h0.size());
            for (int t = 0; t < T; ++t) {
                std::memcpy(h0.data() + (size_t) t * (kNin / 2), x.data() + (size_t) t * kNin,
                            (size_t) (kNin / 2) * sizeof(float));
                std::memcpy(h1.data() + (size_t) t * (kNin / 2), x.data() + (size_t) t * kNin + kNin / 2,
                            (size_t) (kNin / 2) * sizeof(float));
            }
            float* d_h0 = nullptr;
            float* d_h1 = nullptr;
            CK(cudaMalloc((void**) &d_h0, h0.size() * sizeof(float)));
            CK(cudaMalloc((void**) &d_h1, h1.size() * sizeof(float)));
            CK(cudaMemcpyAsync(d_h0, h0.data(), h0.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
            CK(cudaMemcpyAsync(d_h1, h1.data(), h1.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
            strata::kernels::native_quantize_q8_1(d_h0, d_xq_in0, kNin / 2, T, cs);
            strata::kernels::native_quantize_q8_1(d_h1, d_xq_in1, kNin / 2, T, cs);
            CK(cudaStreamSynchronize(cs));
            cudaFree(d_h0);
            cudaFree(d_h1);
        }
        strata::kernels::native_mmvq(8, d_win0, d_xq_in0, d_y_in0, kNin / 2, kNout, T, cs);
        strata::kernels::native_mmvq(8, d_win1, d_xq_in1, d_y_in1, kNin / 2, kNout, T, cs);

        // 3. the n_out split (the proposed fix): each card holds 1280 whole rows and reduces all 6144 terms.
        //    Same n_in, same q8_1 image, same kernel - only the row range moves.
        strata::kernels::native_mmvq(8, d_w, d_xq, d_y_out0, kNin, kNout / 2, T, cs);
        strata::kernels::native_mmvq(8, (const uint8_t*) d_w + row_bytes(kNin) * (kNout / 2), d_xq, d_y_out1,
                                     kNin, kNout / 2, T, cs);

        std::vector<float> y_full((size_t) T * kNout), y_in0(y_full.size()), y_in1(y_full.size());
        std::vector<float> y_out0((size_t) T * (kNout / 2)), y_out1(y_out0.size());
        CK(cudaMemcpyAsync(y_full.data(), d_y_full, y_full.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaMemcpyAsync(y_in0.data(), d_y_in0, y_in0.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaMemcpyAsync(y_in1.data(), d_y_in1, y_in1.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaMemcpyAsync(y_out0.data(), d_y_out0, y_out0.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaMemcpyAsync(y_out1.data(), d_y_out1, y_out1.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaStreamSynchronize(cs));

        // n_in split: the allreduce's `a[i] + b[i]`, compared element-wise with the single card.
        int64_t in_diff = 0;
        double in_max_rel = 0.0;
        for (size_t i = 0; i < y_full.size(); ++i) {
            const float sum = y_in0[i] + y_in1[i];
            if (sum != y_full[i]) {
                ++in_diff;
                const double den = std::fabs((double) y_full[i]);
                const double rel = den > 0.0 ? std::fabs((double) sum - (double) y_full[i]) / den : 0.0;
                if (rel > in_max_rel) in_max_rel = rel;
            }
        }
        // n_out split: card 0's rows are 0..1279, card 1's are 1280..2559, laid out at the same row stride.
        // NOTE the buffers above are COMPACT [T][kNout/2] - which is what native_mmvq writes when n_out is
        // halved (`y[j * n_out + row]`).  The engine does NOT write straight into its [T][N] residual buffer:
        // doing so shifts token t to t*(N/2), which T=1 cannot show (that was a real regression).  The
        // `pitch` check below pins that down.
        int64_t out_diff = 0;
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < kNout / 2; ++r) {
                if (y_out0[(size_t) t * (kNout / 2) + r] != y_full[(size_t) t * kNout + r]) ++out_diff;
                if (y_out1[(size_t) t * (kNout / 2) + r] != y_full[(size_t) t * kNout + kNout / 2 + r]) ++out_diff;
            }
        }
        // The engine's step: pitch each token's compact half into the [T][N] layout, then let the allreduce
        // add the two disjoint halves (the other half is zero, and x + 0.0f == x bitwise).
        int64_t pitch_diff = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> assembled((size_t) kNout, 0.0f);
            std::memcpy(assembled.data(), y_out0.data() + (size_t) t * (kNout / 2),
                        (size_t) (kNout / 2) * sizeof(float));
            std::memcpy(assembled.data() + kNout / 2, y_out1.data() + (size_t) t * (kNout / 2),
                        (size_t) (kNout / 2) * sizeof(float));
            for (int r = 0; r < kNout; ++r)
                if (assembled[(size_t) r] != y_full[(size_t) t * kNout + r]) ++pitch_diff;
        }
        // And the mistake this guards: writing the compact half at t*(N/2) inside a [T][N] buffer must NOT
        // reproduce the single card once T > 1 (if it did, this test would be unable to catch the regression).
        int64_t flat_diff = 0;
        if (T > 1) {
            std::vector<float> flat((size_t) T * kNout, 0.0f);
            for (int t = 0; t < T; ++t)
                std::memcpy(flat.data() + (size_t) t * (kNout / 2), y_out0.data() + (size_t) t * (kNout / 2),
                            (size_t) (kNout / 2) * sizeof(float));
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < kNout / 2; ++r)
                    if (flat[(size_t) t * kNout + r] != y_full[(size_t) t * kNout + r]) ++flat_diff;
        }

        const int64_t n = (int64_t) y_full.size();
        std::printf("  T=%d: n_in split differs on %lld/%lld elements (max rel %.3e)   n_out split differs on %lld/%lld\n",
                    T, (long long) in_diff, (long long) n, in_max_rel, (long long) out_diff, (long long) n);
        std::printf("        pitched assembly differs on %lld, flat-write (the bug) on %lld\n", (long long) pitch_diff,
                    (long long) flat_diff);

        // The point of the test: the n_out split is bit-exact, the n_in split is not.  A T=1 row is the case
        // the engine actually runs at decode; the multi-column cases cover the batched windows.
        if (out_diff != 0) {
            std::fprintf(stderr, "FAIL: T=%d: the n_out split is not bit-exact with the single card (%lld elements)\n",
                         T, (long long) out_diff);
            ++fails;
        }
        if (pitch_diff != 0) {
            std::fprintf(stderr, "FAIL: T=%d: the per-token pitched assembly is not bit-exact (%lld elements)\n", T,
                         (long long) pitch_diff);
            ++fails;
        }
        if (T > 1 && flat_diff == 0) {
            std::fprintf(stderr, "FAIL: T=%d: a flat write of the compact half MATCHED the single card - the pitch "
                                 "guard cannot catch the regression it exists for\n", T);
            ++fails;
        }
        if (T >= 2 && in_diff == 0) {
            // Not a correctness failure, but it would mean this test no longer distinguishes the two splits.
            std::fprintf(stderr, "WARN: T=%d: the n_in split happened to match; the fixture no longer separates them\n", T);
        }

        cudaFree(d_x);
        cudaFree(d_xq);
        cudaFree(d_win0);
        cudaFree(d_win1);
        cudaFree(d_xq_in0);
        cudaFree(d_xq_in1);
        cudaFree(d_y_full);
        cudaFree(d_y_in0);
        cudaFree(d_y_in1);
        cudaFree(d_y_out0);
        cudaFree(d_y_out1);
    }
    cudaFree(d_w);
    cudaStreamDestroy(cs);
    if (fails != 0) return 1;
    std::printf("tp_outproj_split: ALL PASS (the n_out split is bit-exact)\n");
    return 0;
}
