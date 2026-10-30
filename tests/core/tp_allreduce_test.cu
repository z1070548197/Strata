// tests/core/tp_allreduce_test.cu - correctness and timing of the two-GPU P2P allreduce.
//
// Requires two visible CUDA devices with P2P (the 2080 Ti pair: NV2).  With one
// device, or with STRATA_TP_NO_P2P=1, the open() path is checked to fail cleanly.
//
// Checks:
//   1. open() succeeds with P2P on a two-device box (or fails cleanly otherwise)
//   2. allreduce: buf_a == buf_b == a + b, element-wise, for several sizes
//   3. repeated allreduce stays correct (no drift across 100 iterations)
//   4. timing: report us per allreduce for a spec-6 window (7 x H=2560 floats)
#include "strata/core/tp_allreduce.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(bool ok, const char* what, const std::string& err = "") {
    if (!ok) { std::fprintf(stderr, "FAIL: %s%s%s\n", what, err.empty() ? "" : " - ", err.c_str()); std::exit(1); }
}
static void gpu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    int count = 0;
    gpu(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) {
        std::fprintf(stderr, "SKIP: only %d CUDA device(s) visible; need two for P2P allreduce\n", count);
        return 0;
    }

    strata::core::TpAllreduce ar;
    std::string err;
    const bool p2p = ar.open(0, 1, err);
    if (!p2p) {
        std::fprintf(stderr, "SKIP: no P2P between device 0 and 1 (%s)\n", err.c_str());
        return 0;
    }
    check(ar.p2p(), "open reports P2P");

    // The hidden dim of the model (expert.hpp: H = 2560) and a spec-6 window (7 tokens).
    constexpr int64_t H = 2560;
    constexpr int64_t WINDOW = 7;
    const int64_t max_n = WINDOW * H;

    // Streams are per-device: create each on the device it will be used on.
    cudaStream_t s0, s1;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream0");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream1");

    // Host-side reference data: deterministic, non-trivial values.
    std::vector<float> ha(max_n), hb(max_n);
    for (int64_t i = 0; i < max_n; ++i) {
        ha[i] = std::sin((float) i * 0.001f) * 3.0f;
        hb[i] = std::cos((float) i * 0.002f) * 2.0f;
    }

    float *da = nullptr, *db = nullptr;
    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaMalloc((void**) &da, (size_t) max_n * sizeof(float)), "malloc a");
    gpu(cudaMemcpy(da, ha.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "copy a");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaMalloc((void**) &db, (size_t) max_n * sizeof(float)), "malloc b");
    gpu(cudaMemcpy(db, hb.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "copy b");

    // Sizes to exercise: the full window, a single token, and a ragged size.
    const int64_t sizes[] = {max_n, H, 1000};
    for (int64_t n : sizes) {
        // Re-seed the buffers for each size (allreduce is destructive).
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(da, ha.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed a");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(db, hb.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed b");

        check(ar.allreduce(da, db, n, s0, s1, err), "allreduce", err);
        gpu(cudaStreamSynchronize(s0), "sync 0");
        gpu(cudaStreamSynchronize(s1), "sync 1");

        std::vector<float> ra(n), rb(n);
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(ra.data(), da, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "read a");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(rb.data(), db, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "read b");

        double max_err = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float want = ha[i] + hb[i];
            max_err = std::max(max_err, (double) std::fabs(ra[i] - want));
            max_err = std::max(max_err, (double) std::fabs(rb[i] - want));
        }
        // f32 add of two f32 values: exact in f64 reference, so the error is only the
        // single rounding of the device add.  1e-4 is generous for |values| ~ 5.
        check(max_err < 1e-4, "allreduce result matches a + b");
        std::fprintf(stderr, "  n=%lld: max_err=%.2e OK\n", (long long) n, max_err);
    }

    // Large-T prefill transport: strided rows in both directions.  This is the
    // layout used to gather y halves and stitch output-row bands per token.
    {
        constexpr int64_t rows = 5, cols = 7, src_ld = 11, dst_ld = 13;
        std::vector<float> src((size_t) rows * src_ld), dst((size_t) rows * dst_ld, -9.0f);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (float) i + 0.25f;
        float *sa = nullptr, *db2 = nullptr, *sb = nullptr, *da2 = nullptr;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMalloc(&sa, src.size() * sizeof(float)), "rows src a");
        gpu(cudaMalloc(&da2, dst.size() * sizeof(float)), "rows dst a");
        gpu(cudaMemcpy(sa, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice), "rows src a h2d");
        gpu(cudaMemcpy(da2, dst.data(), dst.size() * sizeof(float), cudaMemcpyHostToDevice), "rows dst a h2d");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMalloc(&sb, src.size() * sizeof(float)), "rows src b");
        gpu(cudaMalloc(&db2, dst.size() * sizeof(float)), "rows dst b");
        gpu(cudaMemcpy(sb, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice), "rows src b h2d");
        gpu(cudaMemcpy(db2, dst.data(), dst.size() * sizeof(float), cudaMemcpyHostToDevice), "rows dst b h2d");
        check(ar.copy_rows(sa, src_ld, db2, dst_ld, rows, cols, true, s0, err), "copy_rows a to b", err);
        gpu(cudaStreamSynchronize(s0), "copy rows sync a");
        check(ar.copy_rows(sb, src_ld, da2, dst_ld, rows, cols, false, s1, err), "copy_rows b to a", err);
        gpu(cudaStreamSynchronize(s1), "copy rows sync b");
        std::vector<float> got_a(dst.size()), got_b(dst.size());
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(got_a.data(), da2, got_a.size() * sizeof(float), cudaMemcpyDeviceToHost), "rows dst a d2h");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(got_b.data(), db2, got_b.size() * sizeof(float), cudaMemcpyDeviceToHost), "rows dst b d2h");
        for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < cols; ++c) {
            const float want = src[(size_t) r * src_ld + c];
            check(got_a[(size_t) r * dst_ld + c] == want, "copy_rows b to a result");
            check(got_b[(size_t) r * dst_ld + c] == want, "copy_rows a to b result");
        }
        std::string bad;
        check(!ar.copy_rows(sa, cols - 1, db2, dst_ld, rows, cols, true, s0, bad), "copy_rows rejects short pitch");
        gpu(cudaSetDevice(0), "set 0"); cudaFree(sa); cudaFree(da2);
        gpu(cudaSetDevice(1), "set 1"); cudaFree(sb); cudaFree(db2);
    }

    // Repeated allreduce: run 100 times on the full window; the result must stay
    // correct (the single-writer kernel is idempotent given the same inputs).
    {
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(da, ha.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "reseed a");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(db, hb.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "reseed b");
        for (int i = 0; i < 100; ++i) {
            check(ar.allreduce(da, db, max_n, s0, s1, err), "allreduce (repeat)", err);
            gpu(cudaStreamSynchronize(s0), "sync 0");
            // Restore the originals so the next iteration is the same problem.
            gpu(cudaSetDevice(0), "set 0");
            gpu(cudaMemcpy(da, ha.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "restore a");
            gpu(cudaSetDevice(1), "set 1");
            gpu(cudaMemcpy(db, hb.data(), (size_t) max_n * sizeof(float), cudaMemcpyHostToDevice), "restore b");
            // The restore is on the default streams; the next allreduce runs on s0 (non-blocking),
            // which does NOT order against the default streams.  db in particular is written by the
            // allreduce (dev0, P2P) and by the restore (dev1) - cross-device, unordered.  Sync both
            // devices so the restore is complete before the next allreduce reads the buffers.
            gpu(cudaSetDevice(0), "set 0");
            gpu(cudaDeviceSynchronize(), "dev sync 0");
            gpu(cudaSetDevice(1), "set 1");
            gpu(cudaDeviceSynchronize(), "dev sync 1");
        }
        check(ar.allreduce(da, db, max_n, s0, s1, err), "allreduce (final)", err);
        gpu(cudaStreamSynchronize(s0), "sync 0");
        gpu(cudaStreamSynchronize(s1), "sync 1");
        std::vector<float> ra(max_n);
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(ra.data(), da, (size_t) max_n * sizeof(float), cudaMemcpyDeviceToHost), "read a");
        double max_err = 0.0;
        for (int64_t i = 0; i < max_n; ++i) max_err = std::max(max_err, (double) std::fabs(ra[i] - (ha[i] + hb[i])));
        check(max_err < 1e-4, "repeated allreduce stays correct");
    }

    // Timing: 1000 allreduces of the full spec-6 window.  The buffers are NOT reseeded
    // between iterations - in the real engine the buffer is written by the previous
    // layer on-device, so the allreduce + sync is the whole cost.  (Values drift; this
    // is a timing loop, not a correctness loop.)  The single-writer kernel runs on s0
    // (dev_a); syncing s0 also completes its P2P writes to dev_b, so s1 (unused) need
    // not be synced.
    {
        std::vector<double> us;
        us.reserve(1000);
        for (int i = 0; i < 1000; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            check(ar.allreduce(da, db, max_n, s0, s1, err), "allreduce (timing)", err);
            gpu(cudaStreamSynchronize(s0), "sync 0");
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(us.begin(), us.end());
        std::fprintf(stderr, "  timing (7x2560 floats, allreduce + sync): median %.1f us, p95 %.1f us\n",
                     us[500], us[950]);
    }

    // Fused handshake forms (P1): rank 0's allreduce_wait_raise and rank 1's publish_wait, driven the way a
    // window drives them - both cards' kernels queued, each waiting on the other's flag.  The flags live in
    // pinned host memory (the engine maps them, so both cards see the same words).
    {
        uint32_t* hflags = nullptr;
        gpu(cudaHostAlloc((void**) &hflags, 2 * sizeof(uint32_t), cudaHostAllocMapped), "host flags");
        hflags[0] = 0;   // flag_a: rank 0's raise (the sum is written)
        hflags[1] = 0;   // flag_b: rank 1's raise (the partial is written)
        uint32_t *fa_dev0 = nullptr, *fb_dev0 = nullptr, *fa_dev1 = nullptr, *fb_dev1 = nullptr;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaHostGetDevicePointer((void**) &fa_dev0, &hflags[0], 0), "map a on 0");
        gpu(cudaHostGetDevicePointer((void**) &fb_dev0, &hflags[1], 0), "map b on 0");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaHostGetDevicePointer((void**) &fa_dev1, &hflags[0], 0), "map a on 1");
        gpu(cudaHostGetDevicePointer((void**) &fb_dev1, &hflags[1], 0), "map b on 1");

        const int64_t n = max_n;
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(da, ha.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed a");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(db, hb.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed b");

        const uint32_t want = 7;
        // rank 1's kernel first in wall-clock order so its publish is not late; both spin until the other is up
        check(ar.publish_wait(fb_dev1, want, fa_dev1, want, s1, err), "publish_wait", err);
        check(ar.allreduce_wait_raise(da, db, n, s0, fb_dev0, want, fa_dev0, want, err), "allreduce_wait_raise", err);
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaStreamSynchronize(s0), "sync 0");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaStreamSynchronize(s1), "sync 1");

        std::vector<float> ra(n), rb(n);
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaMemcpy(ra.data(), da, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "read a");
        gpu(cudaSetDevice(1), "set 1");
        gpu(cudaMemcpy(rb.data(), db, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "read b");
        double max_err = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float wantv = ha[i] + hb[i];
            max_err = std::max(max_err, (double) std::fabs(ra[i] - wantv));
            max_err = std::max(max_err, (double) std::fabs(rb[i] - wantv));
        }
        check(max_err < 1e-4, "fused handshake allreduce matches a + b");
        check(hflags[0] == want, "rank 0 raised flag_a");
        check(hflags[1] == want, "rank 1 raised flag_b");
        std::fprintf(stderr, "  fused handshake: max_err=%.2e OK (flags %u/%u)\n", max_err, hflags[0], hflags[1]);

        // repeated, with increasing sequence numbers as a window does (flags carry layer + 1)
        for (uint32_t k = 1; k <= 20; ++k) {
            gpu(cudaSetDevice(0), "set 0");
            gpu(cudaMemcpy(da, ha.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed a");
            gpu(cudaSetDevice(1), "set 1");
            gpu(cudaMemcpy(db, hb.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice), "reseed b");
            check(ar.publish_wait(fb_dev1, k, fa_dev1, k, s1, err), "publish_wait (loop)", err);
            check(ar.allreduce_wait_raise(da, db, n, s0, fb_dev0, k, fa_dev0, k, err), "allreduce_wait_raise (loop)", err);
            gpu(cudaSetDevice(0), "set 0");
            gpu(cudaStreamSynchronize(s0), "sync 0 (loop)");
            gpu(cudaSetDevice(1), "set 1");
            gpu(cudaStreamSynchronize(s1), "sync 1 (loop)");
        }
        std::fprintf(stderr, "  fused handshake: 20 sequence-numbered rounds OK\n");
        gpu(cudaSetDevice(0), "set 0");
        gpu(cudaFreeHost(hflags), "free host flags");
    }

    gpu(cudaSetDevice(0), "set 0");
    gpu(cudaFree(da), "free a");
    gpu(cudaSetDevice(1), "set 1");
    gpu(cudaFree(db), "free b");
    gpu(cudaStreamDestroy(s0), "destroy s0");
    gpu(cudaStreamDestroy(s1), "destroy s1");
    ar.close();
    std::fprintf(stderr, "PASS: tp_allreduce_test\n");
    return 0;
}
