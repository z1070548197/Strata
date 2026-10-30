// moe_combine_cols_test - strata::prefill::moe_combine_cols (one TP rank's column block of the MoE combine) against
// moe_combine over the full width, bit for bit: each block must reproduce that rank's columns of the one-call result
// exactly, at N/2 blocks and at a padded block pitch, with and without row_sd, and on the scalar path (a block whose
// width is not a multiple of four, which the vectorized kernel refuses).  The TP MoE path stitches the two blocks, so
// this is the gate that the stitch equals the full-width combine.  Synthetic, no model.  Exit 77 without a device.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace sp = strata::prefill;

namespace {
constexpr int64_t N = 2560, K = 10;

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

int failures = 0;

// ref = moe_combine over [0, N); one moe_combine_cols block per rank reads the same Dm values from column 0 (a TP
// rank's down output lands in its own block there) and must give ref's columns [off, off + cols), bit for bit.
void case_block(const char* name, int64_t T, int64_t cols, int64_t off, int64_t shared_ld, int64_t bo_ld, bool with_sd,
                uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    const int64_t rows = T * K + 37;   // Dm has more rows than the tokens route (a CPU share's tail, padding)
    std::vector<float> Dm((size_t) (rows * N)), shared((size_t) (T * N)), sg((size_t) T), w((size_t) (T * K)),
        sd((size_t) rows);
    for (auto& v : Dm) v = nd(rng) * 0.3f;
    for (auto& v : shared) v = nd(rng) * 0.1f;
    for (auto& v : sg) v = nd(rng);
    for (auto& v : w) v = u01(rng);
    for (auto& v : sd) v = 0.001f + u01(rng) * 0.01f;
    std::vector<int32_t> slot((size_t) (T * K));
    for (int64_t i = 0; i < T * K; ++i) slot[(size_t) i] = (int32_t) ((i * 7919 + 13) % (T * K));   // a permutation

    // This rank's down output: the block's values, in place at column 0 (what the TP GEMM leaves there).
    std::vector<float> Dblk((size_t) (rows * N), 0.f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t d = 0; d < cols; ++d) Dblk[(size_t) (r * N + d)] = Dm[(size_t) (r * N + off + d)];
    // The block's own view of `shared`: same values in [off, off + cols), junk outside, so a read outside the block
    // shows up in the comparison instead of passing on a coincidence.
    std::vector<float> shblk((size_t) (T * shared_ld));
    for (auto& v : shblk) v = nd(rng) * 0.7f;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < cols; ++d) shblk[(size_t) (t * shared_ld + off + d)] = shared[(size_t) (t * N + off + d)];

    float *dDm = dev(Dm), *dDblk = dev(Dblk), *dsh = dev(shared), *dshb = dev(shblk), *dsg = dev(sg), *dw = dev(w),
          *dsd = dev(sd);
    int32_t* dslot = dev(slot);
    float *dref = nullptr, *dblk = nullptr;
    cudaMalloc(&dref, (size_t) (T * N) * 4);
    cudaMalloc(&dblk, (size_t) (T * bo_ld) * 4);
    cudaMemset(dref, 0, (size_t) (T * N) * 4);
    cudaMemset(dblk, 0xcd, (size_t) (T * bo_ld) * 4);   // sentinel: the kernel must write every column of [0, cols)

    sp::moe_combine(dDm, dslot, dw, dsh, dsg, dref, T, nullptr, with_sd ? dsd : nullptr);
    sp::moe_combine_cols(dDblk, dslot, dw, dshb, dsg, dblk, T, cols, shared_ld, off, bo_ld, nullptr,
                         with_sd ? dsd : nullptr);
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::printf("FAIL %s: %s\n", name, cudaGetErrorString(status));
        ++failures;
        return;
    }
    const std::vector<float> ref = host(dref, (size_t) (T * N)), blk = host(dblk, (size_t) (T * bo_ld));
    int64_t bad = 0, bad_pad = 0, first = -1;
    uint32_t first_bits = 0, ref_bits = 0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t d = 0; d < cols; ++d) {
            const float a = blk[(size_t) (t * bo_ld + d)], b = ref[(size_t) (t * N + off + d)];
            uint32_t ab, bb;
            std::memcpy(&ab, &a, 4);
            std::memcpy(&bb, &b, 4);
            if (ab != bb) {
                if (bad == 0) { first = t * cols + d; first_bits = ab; ref_bits = bb; }
                ++bad;
            }
        }
        for (int64_t d = cols; d < bo_ld; ++d) {   // the pad stays untouched
            uint32_t ab;
            const float a = blk[(size_t) (t * bo_ld + d)];
            std::memcpy(&ab, &a, 4);
            if (ab != 0xcdcdcdcdU) ++bad_pad;
        }
    }
    if (bad != 0 || bad_pad != 0) {
        std::printf("FAIL %s: %lld of %lld columns differ (first at flat %lld: %08x vs %08x), %lld pad writes\n", name,
                    (long long) bad, (long long) (T * cols), (long long) first, first_bits, ref_bits,
                    (long long) bad_pad);
        ++failures;
        return;
    }
    std::printf("ok   %s: T=%lld cols=%lld off=%lld shared_ld=%lld bo_ld=%lld sd=%d bitwise\n", name, (long long) T,
                (long long) cols, (long long) off, (long long) shared_ld, (long long) bo_ld, with_sd ? 1 : 0);
}
}  // namespace

int main() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        std::printf("skip: no CUDA device\n");
        return 77;
    }
    // The TP MoE split: two N/2 column blocks per layer, dispatched as the rank's block at column 0.
    for (int rank = 0; rank < 2; ++rank) {
        case_block(rank == 0 ? "half-r0" : "half-r1", 37, N / 2, rank * (N / 2), N, N / 2, false, 11 + rank);
        case_block(rank == 0 ? "half-r0-sd" : "half-r1-sd", 5, N / 2, rank * (N / 2), N, N / 2, true, 21 + rank);
    }
    // A padded destination pitch (the stitch reads a tight [T][cols], the combine test's pad must stay untouched).
    case_block("half-pad", 9, N / 2, 0, N, N / 2 + 16, false, 31);
    case_block("quarter-pad", 9, 640, N / 2, N + 8, 648, true, 32);
    // The scalar path: a width, an offset and a pitch the vectorized kernel refuses.
    case_block("scalar-cols", 7, 1250, 5, N + 1, 1253, false, 41);
    case_block("scalar-sd", 3, 641, 1919, N, 641, true, 42);
    // One token, the smallest shape the decode handoff can produce.
    case_block("one-token", 1, N / 2, N / 2, N, N / 2, false, 51);

    if (failures != 0) {
        std::printf("%d case(s) failed\n", failures);
        return 1;
    }
    std::printf("all moe_combine_cols cases bitwise moe_combine\n");
    return 0;
}
