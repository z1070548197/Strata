// strata_nvfp4_q8_0_cpu_order - the CPU pool's NVFP4 expert rows, computed on the GPU, bit for bit.
//
// REFERENCE: src/kernels/cpu/nvfp4_avx2.cpp `row_dot` - the kernel an AVX2 CPU runs, which
// `nvfp4_avx512_parity.cpp:50` declares `exact = true` (bit-equal to ggml-cpu's AVX2 dot) and which
// `native_gu_mt_min` selects on this machine (it has no AVX-512).  The model below was validated against the
// real `nvfp4_256_rows` on 256 (row, token) pairs: 0 differ, worst relative 0.000e+00.
//
// WHY THIS EXISTS.  The engine computes a routed expert on the CPU pool when it is a miss and on the GPU when it
// is a hit.  On the native NVFP4 pack those two paths do not produce the same number, so changing which experts
// are resident changes the output tokens (measured: fork at token 2 / 8 / 23 as the affected fraction grows).
// A GPU expert kernel bit-identical to the CPU pool removes that constraint: residency, cache size and TP
// splitting become output-neutral, which is the prerequisite for using the second card's VRAM at all.
//
// THE ARITHMETIC, and which parts may be rearranged:
//
//   * INTEGER - exactly associative.  |e2m1| <= 12 (kvalues_fp4 holds the values DOUBLED) and |q8_0| <= 127, so
//     four products sum to at most 4*12*127 = 6096 per int32 partial: no overflow, no saturation.  The CPU's
//     vpsignb trick (`|v| * (q * sign(v))` = `v * q`) is a plain integer dot product, so a GPU may use dp4a
//     and any lane assignment without changing a bit.
//   * FP32 - NOT associative, so it is transcribed literally.  The reference keeps EIGHT fp32 accumulators;
//     per 64-value block each takes exactly TWO fma steps, in this order:
//        step 1: acc[L] = fma( ue4m3(d[half])   * dy[2*il],     (float) p1, acc[L] )
//        step 2: acc[L] = fma( ue4m3(d[2+half]) * dy[2*il + 1], (float) p2, acc[L] )
//     with lane L: half = L >> 2, q = L & 3.  `ue4m3 * dy` is ONE rounded multiply (not an fma) and the fma is a
//     single fused rounding of `scale*p + acc`; both are reproduced with the rounding-explicit intrinsics.
//   * the row's value is the reference's `hsum8` tree over those 8 accumulators, reproduced literally:
//        t0 = a4+a0; t1 = a5+a1; t2 = a6+a2; t3 = a7+a3;   result = (t0+t2) + (t1+t3)
//
// ONE THREAD per (row, token) holds all 8 accumulators in registers, so the tree is a plain sequence of adds -
// no shuffles, no shared memory, and no chance of the compiler re-associating across a lane boundary.
//
// DO NOT add this file to the `native_*` --use_fast_math list in CMakeLists.txt: `/` must stay IEEE, because the
// reference's SiLU is glibc's `std::exp`.  `expf` itself is NOT the CUDA libdevice one: `strata_expf_exact`
// (expf_exact.h) is a bit-exact transcription of glibc's `__expf_fma`, verified on all 2^32 float bit patterns.
// libdevice's `expf` differs from glibc in the last ulp for a small fraction of inputs, which showed up as
// 2 differing rows out of 2560 on some seeds.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "strata/kernels/expf_exact.h"

namespace strata::kernels {
namespace {

constexpr int kQK = 64;          // values per block_nvfp4
constexpr int kBlockBytes = 36;  // uint8 d[4] (ue4m3 per 16-value sub-block) + uint8 qs[32] (nibbles)
constexpr int kActBytes = 34;    // block_q8_0: fp16 d + int8 qs[32]

// kvalues_fp4: E2M1 magnitudes, doubled.  Code bit 3 is the sign.  Constant memory: every thread indexes the
// same 16 entries, so these are broadcasts.
//
// Held as four 32-bit words rather than an int8_t[16] so a four-code lookup is one ALIGNED load: an int8_t array
// has byte alignment, and a 4-byte read of it would be a misaligned access.  Little-endian, so word 0 is
// {0, 1, 2, 3} and the last is {-4, -6, -8, -12}.
__device__ __constant__ int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ggml's ggml_cuda_ue4m3_to_fp32, halved (kvalues_fp4 is doubled).  Equal to the CPU's kUe4m3 table for all 256
// inputs - every such value is exact in fp32 - with 0x00 and 0x7F read as 0 in both.
__device__ __forceinline__ float ue4m3_half(uint32_t x) {
    if (x == 0u || x == 0x7Fu) return 0.0f;
    const uint32_t e = (x >> 3) & 0xFu, m = x & 0x7u;
    return e == 0u ? (float) m * 0.0009765625f : __uint_as_float(((e + 119u) << 23) | (m << 20));
}

// The four E2M1 values a lane sums in one step, from 4 consecutive nibble bytes, packed as SIGNED bytes
// (dp4a is a signed x signed dot product, and kFp4w holds negative values).
// `nib_hi` selects the high nibble (values 8..15 of the sub-block) over the low one (values 0..7).
//
// The four bytes are loaded as ONE 32-bit word when the address allows it, and the nibble split plus the table
// lookup are done with `__byte_perm` against kFp4w, the table pre-packed as four words.
//
// Both halves matter.  Indexing an `int8_t kFp4[16]` in __constant__ memory with a per-thread index is a
// DIVERGENT constant access: the hardware serializes it into as many transactions as there are distinct indices,
// and a warp of eight chains holds most of the sixteen.  Four of those per step, two steps per block, was the
// whole 3.5 GB/s (against 299 GB/s for a pure stream).  `__byte_perm` is a single ALU op on words already in
// registers.
// The four E2M1 values a lane sums in one step, from 4 consecutive nibble bytes, packed as SIGNED bytes
// (dp4a is a signed x signed dot product, and kFp4 holds negative values).
// `nib_hi` selects the high nibble (values 8..15 of the sub-block) over the low one (values 0..7).
//
// WHY NOT A TABLE.  `kFp4[code]` is a `__constant__` array indexed by a per-thread value, which is a DIVERGENT
// constant access: it serializes into one transaction per distinct index, and a warp of eight chains holds most of
// the sixteen.  Measured on the real geometry, that alone was the difference between 247 GB/s (the same loads with
// no decode) and 18.7 GB/s.  The replacement is pure ALU:
//
//   magnitude = ((2 | man) << exp) >> 1   with exp = m>>1, man = m&1, and `man` alone when exp == 0
//
// which reproduces {0,1,2,3,4,6,8,12} for codes 0..7 (verified on all 262144 four-code combinations), and bit 3
// negates.  The magnitudes are packed one nibble per code into a 64-bit constant, so the whole lookup is a shift
// and a mask: 18.7 -> 29.3 GB/s on the same arithmetic and the same bits (0/640 rows differ).
//
// (`__byte_perm` would do it in three ops, but PRMT's per-byte selector semantics differ on sm_75 - measured
// 130809/131072 wrong there against 0/131072 on sm_120 - and this build targets sm_75.)
// A 32-BIT table, indexed by the code's magnitude bits only (code & 7), with the sign taken from bit 3 separately.
// The table therefore fits in one register and the shift is a 32-bit SHF.R.U32 (1 cycle) rather than the 64-bit
// SHF.R.U64 (2 cycles, plus a register pair) that a 16-entry table forces.
__device__ __constant__ unsigned int kMag32 = 0xC8643210u;   // code 0..7 -> nibble 0..7 = 0,1,2,3,4,6,8,12

__device__ __forceinline__ int codes4(uint32_t w) {
    w &= 0x0F0F0F0Fu;
    int r = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t c = (w >> (8 * k)) & 0xFu;
        const uint32_t v = (kMag32 >> (4 * (c & 7u))) & 0xFu;     // magnitude, 0..12
        // bit 3 is the sign; a select rather than a negate keeps this two ops.  The `& 0xFF` matters: a negated
        // uint32 is all-ones in the high bytes and the shift below would drag them into the neighbouring byte.
        r |= (int) ((((c & 8u) ? (0u - v) : v) & 0xFFu) << (8 * k));
    }
    return r;
}

// The HIGH nibble of each of the four bytes, as four codes.  Must be done per byte: a whole-word `w >> 4` drags
// byte k+1's low nibble into byte k's top, which is not the same code (that mistake put 132096/132096 rows off).
__device__ __forceinline__ int codes4_hi(uint32_t w) { return codes4((w >> 4) & 0x0F0F0F0Fu); }

// The four-byte load `codes4` takes.  A lane's four nibble bytes are contiguous inside an 8-byte sub-block, and
// every caller's address is 4-byte aligned (the qs payload of a block_nvfp4 starts at blk+4, and a lane's offset
// inside it is a multiple of 4), so this is a single LDG.32.
//
// It must be a genuine single load.  The obvious `if (aligned) *(uint32_t*)p else p[0]|p[1]<<8|...` is a trap:
// the compiler cannot prove the fallback's alignment and emits four `LDG.E.U8` - four transactions for one word,
// once per lane per sub-block.  An unconditional 32-bit load cannot be used instead either: LDG.32 on this
// hardware REQUIRES 4-byte alignment and faults otherwise (measured: "misaligned address"), which is why
// `q8_4` below, whose addresses are 2 mod 4, does not call this.
__device__ __forceinline__ uint32_t load4(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// The four q8_0 activation bytes a lane pairs with, packed as SIGNED bytes.
// A block_q8_0 is 34 bytes with its payload at offset 2, so a lane's address is 2 mod 4 and no 32-bit load can
// reach it directly.  Two aligned 16-bit loads stitched into one word give the same result with LDG.16s, and keep
// the four bytes in one register pair instead of four separate byte loads.
__device__ __forceinline__ int q8_4(const int8_t* y) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(y);
    const uint32_t lo = *reinterpret_cast<const uint16_t*>(p);
    const uint32_t hi = *reinterpret_cast<const uint16_t*>(p + 2);
    return (int) (lo | (hi << 16));
}

// The reference's hsum8, over eight values that sit one per lane of an 8-lane group:
//   t0=a4+a0, t1=a5+a1, t2=a6+a2, t3=a7+a3, result=(t0+t2)+(t1+t3).
// Reproduced with shuffles rather than a butterfly, because a butterfly is a different floating-point expression
// (the reference's pairing is 4,1,2 - see s2_expert_grouped.cu's row_dot_cpu_order, which had to match it too).
// `width` 8 keeps the four groups of a warp independent; the source lane is relative to each group.
// Lane L holds a[L].  Each lane computes one of the four pairwise sums, `a[L] + a[L^4]` - which is exactly the
// reference's t0..t3 pairing (t0=a4+a0, t1=a5+a1, t2=a6+a2, t3=a7+a3) - then one shuffle brings the even lane's
// half-sum to its odd partner's neighbour, and the final add is (t0+t2)+(t1+t3).  Three shuffles, and the
// floating-point association is identical to the reference: the original eight-shuffle version broadcast all eight
// values to all eight lanes and formed the same tree, just with five more shuffles.
__device__ __forceinline__ float hsum8_group(float a) {
    const unsigned m = 0xffffffffu;
    const float t = a + __shfl_xor_sync(m, a, 4, 8);   // lane L now holds t_L = a_L + a_{L^4}
    const float u = t + __shfl_xor_sync(m, t, 2, 8);   // lane L holds (t_L + t_{L^2})
    const float v = u + __shfl_xor_sync(m, u, 1, 8);   // all lanes hold (t0+t2) + (t1+t3)
    return v;
}

// The WEIGHT side of one block, decoded once per lane.  A lane's nibble select depends only on (q, half), which are
// fixed for the life of the thread, so the same four code bytes and the same two sub-block scales serve every entry
// that shares this row.  The grouped kernels decode once per block and apply to every entry; `lane_step` below is
// the two calls back to back, for the kernels that have only one entry to apply to.
struct WStep {
    int c1, c2;      // E2M1 codes packed as signed bytes, ready for dp4a
    float sc1, sc2;  // ue4m3 sub-block scales, as the reference multiplies them
};

// A lane loads the two 4-byte words it needs, at qs + half*8 + bbase and qs + 16 + half*8 + bbase.
//
// MEASURED AND REJECTED: coalescing these into "lane p loads word p, then two shuffles hand it round".  That is 8
// requests per block instead of 16, and it is bit-exact, but the two full-mask shuffles cost more than the requests
// they save - gu went 0.208 -> 0.268 ms (the kernel calls this twice per block, once for gate and once for up) and
// down was unchanged.  Loading directly leaves the eight lanes' words inside one 32-byte span, which the coalescer
// merges anyway; see tests/cuda/nvfp4_q8_0_grouped_bench.cpp for the numbers.
__device__ __forceinline__ WStep w_step(const uint8_t* __restrict__ blk, int q, int half) {
    const uint8_t* qs = blk + 4;
    const int bbase = (q & 1) * 4;
    // step 1: sub-block `half` against q8_0 block 2*il; step 2: sub-block 2+half against block 2*il+1
    // sub-block bytes 0..7 are the LOW nibbles of qs[half*8 ..], bytes 8..15 the HIGH nibbles, so the same
    // 4-byte word serves both and the nibble select is the shift.
    const uint32_t w1 = load4(qs + half * 8 + bbase);
    const uint32_t w2 = load4(qs + 16 + half * 8 + bbase);
    WStep s;
    s.c1 = q >= 2 ? codes4_hi(w1) : codes4(w1);
    s.c2 = q >= 2 ? codes4_hi(w2) : codes4(w2);
    s.sc1 = ue4m3_half((uint32_t) blk[half]);
    s.sc2 = ue4m3_half((uint32_t) blk[2 + half]);
    return s;
}

// The ACTIVATION side of one block: the two q8_0 scales and, per lane, the four int8 values it pairs with the
// weight.  A gu entry steps the SAME activation block twice (once against gate, once against up), so decoding it
// once and applying to both saves a load pair and a half-conversion per block per entry.
//
// `x_scales` IS NOT AN OPTIMIZATION - IT IS WHAT MAKES A HIT EQUAL A MISS.  The CPU pool multiplies by the fp32
// `ActQ::scale`; a `block_q8_0` stores an fp16 `d`.  Those are 4.761e-04 relative apart on 80 of 80 chunks
// (bench/micro/act_quant_parity.cu), which is pure fp16 rounding and is why enabling the GPU cache used to change
// the generated tokens.  When non-null, `x_scales[i]` is block i's multiplier and the block's own `d` is ignored.
// Null keeps the previous behaviour exactly, so the existing parity tests still compare what they always did.
struct AStep {
    int q1, q2;      // the lane's four int8 values from the two q8_0 blocks, packed for dp4a
    float dy1, dy2;  // their block scales
};

__device__ __forceinline__ AStep a_step(const uint8_t* __restrict__ a_lo, const uint8_t* __restrict__ a_hi, int q,
                                        int half, const float* __restrict__ sc_lo,
                                        const float* __restrict__ sc_hi) {
    const int abase = 16 * half + 4 * q;
    AStep s;
    s.dy1 = sc_lo != nullptr ? *sc_lo : __half2float(*reinterpret_cast<const __half*>(a_lo));
    s.dy2 = sc_hi != nullptr ? *sc_hi : __half2float(*reinterpret_cast<const __half*>(a_hi));
    s.q1 = q8_4(reinterpret_cast<const int8_t*>(a_lo + 2) + abase);
    s.q2 = q8_4(reinterpret_cast<const int8_t*>(a_hi + 2) + abase);
    return s;
}

// One accumulator step of an already-decoded weight against an already-decoded activation: the same two fma's, in
// the same order, with the same rounding as the reference.
__device__ __forceinline__ void a_apply(const WStep& w, const AStep& a, float& acc) {
    const int p1 = __dp4a(w.c1, a.q1, 0);
    acc = __fmaf_rn(w.sc1 * a.dy1, (float) p1, acc);
    const int p2 = __dp4a(w.c2, a.q2, 0);
    acc = __fmaf_rn(w.sc2 * a.dy2, (float) p2, acc);
}

// Both sides in one call, for the kernels with a single consumer of each block.
__device__ __forceinline__ void w_apply(const WStep& w, const uint8_t* __restrict__ a_lo,
                                        const uint8_t* __restrict__ a_hi, int q, int half,
                                        const float* __restrict__ sc_lo, const float* __restrict__ sc_hi,
                                        float& a) {
    a_apply(w, a_step(a_lo, a_hi, q, half, sc_lo, sc_hi), a);
}

// One block of one entry's chain: exactly the accumulator step the reference takes, so a caller that walks the
// blocks itself (and reuses the decoded weight across entries) stays bit-exact.
__device__ __forceinline__ void lane_step(const uint8_t* __restrict__ blk, const uint8_t* __restrict__ a_lo,
                                          const uint8_t* __restrict__ a_hi, int q, int half, float& a) {
    w_apply(w_step(blk, q, half), a_lo, a_hi, q, half, nullptr, nullptr, a);
}

// One (row, activation) dot product: the reference's operation sequence, all eight lanes in registers.
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ row, const uint8_t* __restrict__ act, int n) {
    const int nb = n / kQK;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;

    for (int il = 0; il < nb; ++il) {
        const uint8_t* blk = row + (size_t) il * kBlockBytes;
        const uint8_t* qs = blk + 4;                 // 32 nibble bytes: sub-block s at qs[8s .. 8s+7]
        const uint8_t* a_lo = act + (size_t) (2 * il) * kActBytes;
        const uint8_t* a_hi = act + (size_t) (2 * il + 1) * kActBytes;
        const float dy_lo = __half2float(*reinterpret_cast<const __half*>(a_lo));
        const float dy_hi = __half2float(*reinterpret_cast<const __half*>(a_hi));
        const int8_t* y_lo = reinterpret_cast<const int8_t*>(a_lo + 2);
        const int8_t* y_hi = reinterpret_cast<const int8_t*>(a_hi + 2);

        // sub-block halves 0 and 1 pair with q8_0 block 2*il (dy_lo); halves 2 and 3 with 2*il+1 (dy_hi).
        // Lane L: q = L&3 chooses 4 of the sub-block's 16 values, half = L>>2 chooses the sub-block pair.
        //   values 0..7  of a sub-block -> low nibbles  of qs[..+0..3] (q<2) / qs[..+4..7] (q>=2)
        //   values 8..15               -> high nibbles of the same bytes
        #define STRATA_LANE_STEP(L, QSB, SCALE_IDX, ACTB, DYB)                                                    \
            do {                                                                                             \
                const int _q = (L) & 3, _half = (L) >> 2;                                                     \
                const uint32_t _w = load4(QSB + _half * 8 + (_q & 1) * 4);                                    \
                const int _p4 = __dp4a(_q >= 2 ? codes4_hi(_w) : codes4(_w),                                      \
                                       q8_4(ACTB + 16 * _half + 4 * _q), 0);                                  \
                const float _sc = ue4m3_half((uint32_t) blk[SCALE_IDX]) * (DYB);                              \
                a##L = __fmaf_rn(_sc, (float) _p4, a##L);                                                     \
            } while (0)

        STRATA_LANE_STEP(0, qs, 0, y_lo, dy_lo);
        STRATA_LANE_STEP(1, qs, 0, y_lo, dy_lo);
        STRATA_LANE_STEP(2, qs, 0, y_lo, dy_lo);
        STRATA_LANE_STEP(3, qs, 0, y_lo, dy_lo);
        STRATA_LANE_STEP(4, qs, 1, y_lo, dy_lo);
        STRATA_LANE_STEP(5, qs, 1, y_lo, dy_lo);
        STRATA_LANE_STEP(6, qs, 1, y_lo, dy_lo);
        STRATA_LANE_STEP(7, qs, 1, y_lo, dy_lo);
        STRATA_LANE_STEP(0, qs + 16, 2, y_hi, dy_hi);
        STRATA_LANE_STEP(1, qs + 16, 2, y_hi, dy_hi);
        STRATA_LANE_STEP(2, qs + 16, 2, y_hi, dy_hi);
        STRATA_LANE_STEP(3, qs + 16, 2, y_hi, dy_hi);
        STRATA_LANE_STEP(4, qs + 16, 3, y_hi, dy_hi);
        STRATA_LANE_STEP(5, qs + 16, 3, y_hi, dy_hi);
        STRATA_LANE_STEP(6, qs + 16, 3, y_hi, dy_hi);
        STRATA_LANE_STEP(7, qs + 16, 3, y_hi, dy_hi);
        #undef STRATA_LANE_STEP
    }
    // the reference's hsum8, literally
    const float t0 = a4 + a0, t1 = a5 + a1, t2 = a6 + a2, t3 = a7 + a3;
    return (t0 + t2) + (t1 + t3);
}

}  // namespace

/// Plain rows: out[tok][row] = scale * dot(weight row, activation of `tok`), rows [r0, r1).
/// Activations are `n_tok` contiguous rows of `act_stride` bytes each, `tok`'s row at act + tok*act_stride.
__global__ void nvfp4_q8_0_rows_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int n,
                                       const uint8_t* __restrict__ act, int act_stride, int n_tok,
                                       float* __restrict__ out, int r0, int r1, float scale) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nwork = (int64_t) (r1 - r0) * n_tok;
    if (gid >= nwork) return;
    const int row = r0 + (int) (gid / n_tok);
    const int tok = (int) (gid % n_tok);
    const float v = row_dot(w + (size_t) row * row_bytes, act + (size_t) tok * act_stride, n);
    out[(size_t) tok * (size_t) (r1 - r0) + (size_t) (row - r0)] = v * scale;
}

/// Gate/up with the blob tail's global scales, then SiLU: ff[tok][row] = silu(sg*G) * (su*U), the expression
/// `native_gu_rows` ends with (native_expert.cpp:243-245) and `nvfp4_avx2.cpp:132-135` repeats.
__global__ void nvfp4_q8_0_gu_kernel(const uint8_t* __restrict__ blob, size_t gu_row, size_t up_off, int n,
                                     const uint8_t* __restrict__ act, int act_stride, int n_tok,
                                     float* __restrict__ ff, int r0, int r1, float sg, float su) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nwork = (int64_t) (r1 - r0) * n_tok;
    if (gid >= nwork) return;
    const int row = r0 + (int) (gid / n_tok);
    const int tok = (int) (gid % n_tok);
    const uint8_t* ar = act + (size_t) tok * act_stride;
    const float g = row_dot(blob + (size_t) row * gu_row, ar, n) * sg;
    const float u = row_dot(blob + up_off + (size_t) row * gu_row, ar, n) * su;
    ff[(size_t) tok * (size_t) (r1 - r0) + (size_t) (row - r0)] = (g / (1.0f + strata_expf_exact(-g))) * u;
}

// ---------------------------------------------------------------- grouped (the native expert path)

// WARP-COOPERATIVE grouped gu: eight lanes per row, several rows per block.  Each lane keeps one of the
// reference's eight accumulators and walks the blocks in order, so the arithmetic is `row_dot_lane`; the group's
// values are then combined with the reference's own tree.  A thread-per-row version measured 3.5 GB/s against
// 299 GB/s for a pure stream - one thread's instruction stream hides no memory latency; this spreads it over
// eight threads.
//
// THE BLOCK LOOP IS OUTSIDE THE ENTRY LOOP.  Every entry of a group reads the SAME weight row, so decoding a
// weight block once and applying it to all of the group's entries removes the decode from the inner loop
// entirely: entry count multiplies only the activation read and the dp4a.  The accumulators are per entry
// (`a[kMaxEntPerRow]`), and each is updated exactly in block order, so every entry's chain is still the
// reference's operation sequence - the reuse changes when the decode happens, never the arithmetic or its order.
constexpr int kMaxEntPerRow = 8;   // entries a single group may hold; beyond this the kernel falls back

__global__ void nvfp4_q8_0_grouped_gu_warp_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                  const int32_t* __restrict__ grp_start,
                                                  const int32_t* __restrict__ n_groups,
                                                  const int32_t* __restrict__ ent_tok, int64_t cap_entries,
                                                  int n_embd, int n_ff, size_t gu_row, size_t up_off,
                                                  size_t tail_off, const uint8_t* __restrict__ act, int act_stride,
                                                  const float* __restrict__ sc_lo, int sc_stride,
                                                  float* __restrict__ ff) {
    const int g = (int) blockIdx.y;
    if (g >= *n_groups) return;
    const int lane = (int) (threadIdx.x & 7);
    const int sub = (int) (threadIdx.x >> 3);            // which row this 8-lane group owns in the block
    const int r = (int) blockIdx.x * (int) (blockDim.x >> 3) + sub;
    if (r >= n_ff) return;

    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 > cap_entries || e1 <= e0) return;
    const int ne_all = e1 - e0;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wg = blob + (size_t) r * gu_row;
    const uint8_t* wu = blob + up_off + (size_t) r * gu_row;
    const float* tail = (const float*) (blob + tail_off);
    const float sg = tail[0], su = tail[1];

    const int q = lane & 3, half = lane >> 2;
    const int nb = n_embd / kQK;

    // A group larger than the register budget is walked in windows of kMaxEntPerRow.  Every entry's chain is
    // independent and is still accumulated in block order, so windowing changes nothing but which entries share
    // one pass over the weights.
    for (int base = 0; base < ne_all; base += kMaxEntPerRow) {
        const int ne = ne_all - base < kMaxEntPerRow ? ne_all - base : kMaxEntPerRow;

        // one activation row pointer per entry, resolved once
        const uint8_t* ar[kMaxEntPerRow];
        for (int k = 0; k < ne; ++k) ar[k] = act + (size_t) ent_tok[e0 + base + k] * (size_t) act_stride;

        float ag[kMaxEntPerRow], au[kMaxEntPerRow];
        for (int k = 0; k < ne; ++k) ag[k] = au[k] = 0.f;

        for (int il = 0; il < nb; ++il) {
            const uint8_t* bg = wg + (size_t) il * kBlockBytes;
            const uint8_t* bu = wu + (size_t) il * kBlockBytes;
            const int ao = 2 * il * kActBytes;
            // decode the two weight blocks ONCE, then apply them to every entry in the window: the nibbles and the
            // sub-block scales do not depend on the entry, so this was `ne` redundant decodes of the same word.
            const WStep swg = w_step(bg, q, half);
            const WStep swu = w_step(bu, q, half);
            for (int k = 0; k < ne; ++k) {
                const uint8_t* al = ar[k] + ao;
                const uint8_t* ah = al + kActBytes;
                // gate and up share one activation block, so read and decode its two scale halves once.
                // `sc_lo` is the token's fp32 scale row: one float per q8_0 block, so block 2*il and 2*il+1.
                const float* s_lo = sc_lo ? sc_lo + (size_t) ent_tok[e0 + base + k] * (size_t) sc_stride + 2 * il : nullptr;
                const AStep av = a_step(al, ah, q, half, s_lo, s_lo ? s_lo + 1 : nullptr);
                a_apply(swg, av, ag[k]);
                a_apply(swu, av, au[k]);
            }
        }

        for (int k = 0; k < ne; ++k) {
            const float gv = hsum8_group(ag[k]) * sg;
            const float uv = hsum8_group(au[k]) * su;
            if (lane == 0)
                ff[(size_t) (e0 + base + k) * (size_t) n_ff + (size_t) r] =
                    (gv / (1.0f + strata_expf_exact(-gv))) * uv;
        }
    }
}

// One thread per (group, row): the gate and the up row are the same index, so a thread computes both and applies
// the pool's composition.  A group is one expert and its entries are the tokens routed to it, so the weights are
// read once per group and reused across its entries - the shape of `native_gu_multi_kernel`, with the pool's
// dot product.  `cap_groups` bounds the grid; `*n_groups` is the real count, read on the device.
__global__ void nvfp4_q8_0_grouped_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                             const int32_t* __restrict__ grp_start,
                                             const int32_t* __restrict__ n_groups,
                                             const int32_t* __restrict__ ent_tok, int64_t cap_entries,
                                             int n_embd, int n_ff, size_t gu_row, size_t up_off, size_t tail_off,
                                             const uint8_t* __restrict__ act, int act_stride,
                                             float* __restrict__ ff) {
    const int g = (int) blockIdx.y;
    if (g >= *n_groups) return;
    const int r = blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (r >= n_ff) return;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wg = blob + (size_t) r * gu_row;
    const uint8_t* wu = blob + up_off + (size_t) r * gu_row;
    const float* tail = (const float*) (blob + tail_off);
    const float sg = tail[0], su = tail[1];

    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 > cap_entries) return;
    for (int e = e0; e < e1; ++e) {
        const uint8_t* ar = act + (size_t) ent_tok[e] * (size_t) act_stride;
        const float gv = row_dot(wg, ar, n_embd) * sg;
        const float uv = row_dot(wu, ar, n_embd) * su;
        // the expression native_gu_rows / nvfp4_256_gu_rows end with, in the same order
        ff[(size_t) e * (size_t) n_ff + (size_t) r] = (gv / (1.0f + strata_expf_exact(-gv))) * uv;
    }
}

// WARP-COOPERATIVE grouped down: as the gu kernel above, eight lanes per row.
__global__ void nvfp4_q8_0_grouped_down_warp_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                    const int32_t* __restrict__ grp_start,
                                                    const int32_t* __restrict__ n_groups,
                                                    const int32_t* __restrict__ ent_dst,
                                                    const int32_t* __restrict__ ent_tok, int64_t cap_entries,
                                                    int n_embd, int n_ff, size_t d_row, size_t down_off,
                                                    size_t tail_off, const uint8_t* __restrict__ act,
                                                    int act_stride, const float* __restrict__ sc_lo, int sc_stride,
                                                    float* __restrict__ out) {
    const int g = (int) blockIdx.y;
    if (g >= *n_groups) return;
    const int lane = (int) (threadIdx.x & 7);
    const int sub = (int) (threadIdx.x >> 3);
    const int row = (int) blockIdx.x * (int) (blockDim.x >> 3) + sub;
    if (row >= n_embd) return;

    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 > cap_entries || e1 <= e0) return;
    const int ne_all = e1 - e0;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + down_off + (size_t) row * d_row;
    const float sd = ((const float*) (blob + tail_off))[2];   // s_down

    const int q = lane & 3, half = lane >> 2;
    const int nb = n_ff / kQK;

    // windowed like the gu kernel: see the note there
    for (int base = 0; base < ne_all; base += kMaxEntPerRow) {
        const int ne = ne_all - base < kMaxEntPerRow ? ne_all - base : kMaxEntPerRow;

        const uint8_t* ar[kMaxEntPerRow];
        // `ent_tok == nullptr` means the activation row IS the entry's index - what a down reading the phase-1
        // intermediate wants, since that is stored one Q8_0 row per entry.
        for (int k = 0; k < ne; ++k)
            ar[k] = act + (size_t) (ent_tok ? ent_tok[e0 + base + k] : (e0 + base + k)) * (size_t) act_stride;

        float av[kMaxEntPerRow];
        for (int k = 0; k < ne; ++k) av[k] = 0.f;

        for (int il = 0; il < nb; ++il) {
            const uint8_t* br = wr + (size_t) il * kBlockBytes;
            const int ao = 2 * il * kActBytes;
            const WStep sw = w_step(br, q, half);   // decoded once, applied to every entry
            for (int k = 0; k < ne; ++k) {
                const uint8_t* al = ar[k] + ao;
                const float* s_lo = sc_lo ? sc_lo + (size_t) (ent_tok ? ent_tok[e0 + base + k] : (e0 + base + k)) * (size_t) sc_stride + 2 * il : nullptr;
                w_apply(sw, al, al + kActBytes, q, half, s_lo, s_lo ? s_lo + 1 : nullptr, av[k]);
            }
        }

        // hsum8_group is a warp shuffle: EVERY lane of the group must execute it.  Computing it inside
        // `if (lane == 0)` would read the other seven lanes' registers while they are inactive - undefined, and
        // measured as a total mismatch (46080/46080) before this was fixed.
        for (int k = 0; k < ne; ++k) {
            const float v = hsum8_group(av[k]) * sd;
            if (lane == 0) out[(size_t) ent_dst[e0 + base + k] * (size_t) n_embd + (size_t) row] = v;
        }
    }
}

__global__ void nvfp4_q8_0_grouped_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                               const int32_t* __restrict__ grp_start,
                                               const int32_t* __restrict__ n_groups,
                                               const int32_t* __restrict__ ent_dst,
                                               const int32_t* __restrict__ ent_tok, int64_t cap_entries,
                                               int n_embd, int n_ff, size_t d_row, size_t down_off,
                                               size_t tail_off, const uint8_t* __restrict__ act, int act_stride,
                                               float* __restrict__ out) {
    const int g = (int) blockIdx.y;
    if (g >= *n_groups) return;
    const int row = blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (row >= n_embd) return;

    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr = blob + down_off + (size_t) row * d_row;
    const float sd = ((const float*) (blob + tail_off))[2];   // s_down
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    if (e1 > cap_entries) return;
    for (int e = e0; e < e1; ++e)
        out[(size_t) ent_dst[e] * (size_t) n_embd + (size_t) row] =
            row_dot(wr, act + (size_t) ent_tok[e] * (size_t) act_stride, n_ff) * sd;
}

// ---------------------------------------------------------------- host entry points

void nvfp4_q8_0_rows(const uint8_t* w, size_t row_bytes, int n, const uint8_t* act, int act_stride, int n_tok,
                     float* out, int r0, int r1, float scale, void* stream) {
    const int64_t nwork = (int64_t) (r1 - r0) * n_tok;
    if (nwork <= 0) return;
    const int threads = 128;
    const int blocks = (int) ((nwork + threads - 1) / threads);
    nvfp4_q8_0_rows_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(w, row_bytes, n, act, act_stride, n_tok,
                                                                         out, r0, r1, scale);
}

void nvfp4_q8_0_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const uint8_t* act,
                        int act_stride, int n_tok, float* ff, int r0, int r1, float sg, float su, void* stream) {
    const int64_t nwork = (int64_t) (r1 - r0) * n_tok;
    if (nwork <= 0) return;
    const int threads = 128;
    const int blocks = (int) ((nwork + threads - 1) / threads);
    nvfp4_q8_0_gu_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(blob, gu_row, up_off, n, act, act_stride,
                                                                       n_tok, ff, r0, r1, sg, su);
}

void nvfp4_q8_0_grouped_gu(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                           const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, int64_t n_embd,
                           int64_t n_ff, size_t gu_row, size_t up_off, size_t tail_off, const uint8_t* act,
                           int act_stride, float* ff, void* stream, const float* sc_lo, int sc_stride) {
    if (cap_groups <= 0 || cap_entries <= 0 || n_ff <= 0) return;
    const int threads = 256;                            // 32 eight-lane groups -> 32 rows per block
    const int rows_per_block = threads / 8;
    const dim3 grid((unsigned) ((n_ff + rows_per_block - 1) / rows_per_block), (unsigned) cap_groups);
    nvfp4_q8_0_grouped_gu_warp_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(
        grp_ptr, grp_start, n_groups, ent_tok, cap_entries, (int) n_embd, (int) n_ff, gu_row, up_off, tail_off, act,
        act_stride, sc_lo, sc_stride, ff);
}

void nvfp4_q8_0_grouped_down(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                             const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                             int64_t cap_entries, int64_t n_embd, int64_t n_ff, size_t d_row, size_t down_off,
                             size_t tail_off, const uint8_t* act, int act_stride, float* out, void* stream,
                             const float* sc_lo, int sc_stride) {
    if (cap_groups <= 0 || cap_entries <= 0 || n_embd <= 0) return;
    const int threads = 256;
    const int rows_per_block = threads / 8;
    const dim3 grid((unsigned) ((n_embd + rows_per_block - 1) / rows_per_block), (unsigned) cap_groups);
    nvfp4_q8_0_grouped_down_warp_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(
        grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_entries, (int) n_embd, (int) n_ff, d_row, down_off,
        tail_off, act, act_stride, sc_lo, sc_stride, out);
}

}  // namespace strata::kernels
