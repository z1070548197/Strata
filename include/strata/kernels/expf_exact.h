/* expf_exact.h -- bit-exact transcription of glibc 2.35 __expf_fma (e_expf.c, FMA variant).
 *
 * Identified by resolving the expf IFUNC at runtime on this machine
 * (AMD Ryzen 5 5600GT, Zen 3): dlsym(RTLD_DEFAULT,"expf") -> libm.so.6 vaddr 0x7aba0
 * = __expf_fma.  Source-level algorithm (glibc sysdeps/ieee754/flt-32/e_expf.c,
 * EXP2F_TABLE_BITS=5, N=32):
 *
 *   z  = fma(x, InvLn2N, Shift)        ; Shift = 1.5*2^52 (round-to-int trick)
 *   ki = bits(z) & 31                  ; low bits of the shifted integer
 *   kd = z - Shift                     ; rounded n as a double
 *   r  = fma(x, InvLn2N, -kd)          ; NOTE: single-rounded FMSUB, not z-kd
 *   p  = fma(C0, r, C1)
 *   q  = fma(C2, r, 1.0)
 *   poly = fma(p, r*r, q)
 *   s  = asdouble(tab[ki] + (ki << 47))   ; integer add on the bit pattern
 *   return (float)(poly * s)
 *
 * All arithmetic is DOUBLE; the only float operation is the final
 * double->float conversion (vcvtsd2ss, round-to-nearest-even).
 *
 * Because s is built by an integer add of (bits(z)<<47) to tab[ki], the
 * exponent scaling costs zero rounding error: s is exactly the correctly
 * rounded double 2^(ki/32) with its exponent shifted.
 *
 * CUDA notes:
 *   - every operation below is IEEE-754 exact and identical on device:
 *     double fma (__fma_rn / ::fma), double mul/sub, uint64 add/shift,
 *     double->float conversion.
 *   - do NOT compile the device code with -ftz=true / -use_fast_math: the
 *     result is subnormal over a real input band (x in [-103.97, -87.3)),
 *     and flushing would change bits.
 */
#ifndef STRATA_EXPF_EXACT_H
#define STRATA_EXPF_EXACT_H

#include <stdint.h>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#  define STRATA_HD __host__ __device__
#else
#  define STRATA_HD
#endif

#if defined(__CUDA_ARCH__)
#  define STRATA_FMA_D(a, b, c) __fma_rn((a), (b), (c))
#  define STRATA_D2F(d)         ((float)(d))
static __device__ __forceinline__ uint32_t strata_f2u(float f) { return __float_as_uint(f); }
static __device__ __forceinline__ uint64_t strata_d2u(double d) { return (uint64_t)__double_as_longlong(d); }
static __device__ __forceinline__ double   strata_u2d(uint64_t u) { return __longlong_as_double((long long)u); }
static __device__ __forceinline__ float    strata_u2f(uint32_t u) { return __uint_as_float(u); }
#else
#  define STRATA_FMA_D(a, b, c) __builtin_fma((a), (b), (c))
#  define STRATA_D2F(d)         ((float)(d))
static inline uint32_t strata_f2u(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return u; }
static inline uint64_t strata_d2u(double d) { uint64_t u; __builtin_memcpy(&u, &d, 8); return u; }
static inline double   strata_u2d(uint64_t u) { double d; __builtin_memcpy(&d, &u, 8); return d; }
static inline float    strata_u2f(uint32_t u) { float f; __builtin_memcpy(&f, &u, 4); return f; }
#endif

/* __exp2f_data.tab[32], read out of libm.so.6 .rodata at vaddr 0xb2b80.
 * tab[i] = bits(2^(i/32)) - (i << 47), so the runtime integer add
 * tab[k] + (ki<<47) = bits(2^(ki/32)) with zero rounding error.
 *
 * CUDA: this lives in device global memory (~256 B, stays in L1).  It is
 * deliberately NOT __constant__, because `ki` diverges across threads inside a
 * warp and __constant__ serialises divergent accesses. */
#if defined(__CUDACC__)
#  define STRATA_TAB_QUAL static __device__ const
#else
#  define STRATA_TAB_QUAL static const
#endif
STRATA_TAB_QUAL uint64_t strata_exp2f_tab[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

/* __exp2f_data scalars, libm.so.6 vaddrs 0xb2ca0/0xb2ca8/0xb2cb0/0xb2cb8/0xb2cc0 */
#define STRATA_EXP2F_SHIFT   0x4338000000000000ULL  /* 1.5 * 2^52            */
#define STRATA_EXP2F_INVLN2N 0x40471547652b82feULL  /* 32/ln2                */
#define STRATA_EXP2F_C0      0x3ebc6af84b912394ULL  /* (ln2/32)^3/6  scaled  */
#define STRATA_EXP2F_C1      0x3f2ebfce50fac4f3ULL  /* (ln2/32)^2/2  scaled  */
#define STRATA_EXP2F_C2      0x3f962e42ff0c52d6ULL  /* (ln2/32)      scaled  */

/* Special-case thresholds, libm.so.6 .rodata at 0x9c9e0/0x9c9e4/0x9c9e8.
 * (float bit patterns, exactly as glibc compares them) */
#define STRATA_EXPF_OFLOW    0x42b17217U  /*  88.72283172607421875  */
#define STRATA_EXPF_UFLOW    0xc2cff1b4U  /* -103.9720764160156250  */
#define STRATA_EXPF_MAYUFLOW 0xc2ce8ecfU  /* -103.2789230346679688  */

/* __math_oflowf/uflowf/may_uflowf constants (libm.so.6 .rodata 0xb2b68..0xb2b70).
 * glibc returns c*c (a single float multiply) with errno=ERANGE; we return the
 * bit value and skip errno. */
#define STRATA_EXPF_UFLOW_C    0x10000000U  /* 2^-95    -> square = 2^-190 -> +0.0 */
#define STRATA_EXPF_MAYUFLOW_C 0x1a200000U  /* 1.25*2^-75 -> square -> 2^-149     */
#define STRATA_EXPF_OFLOW_C    0x70000000U  /* 2^97     -> square = 2^194 -> +inf */

#if defined(__CUDACC__)
#  define STRATA_FN __device__ __forceinline__
#else
#  define STRATA_FN static inline
#endif

/* libm.so.6 vaddr 0x7aba0 <__expf_fma> */
STRATA_FN float strata_expf_exact(float x)
{
    uint32_t xi = strata_f2u(x);
    uint32_t abstop = (xi >> 20) & 0x7ffu;   /* abstop11(|x|): exp<<3 | mantop3 */

    if (abstop > 0x42au) {                   /* |x| >= 80: slow path */
        if (xi == 0xff800000u)               /* -inf -> +0 */
            return 0.0f;
        if (abstop > 0x7f7u)                 /* +inf or NaN */
            return x + x;
        if (x > strata_u2f(STRATA_EXPF_OFLOW))
            return strata_u2f(STRATA_EXPF_OFLOW_C) * strata_u2f(STRATA_EXPF_OFLOW_C);
        if (x < strata_u2f(STRATA_EXPF_UFLOW))
            return strata_u2f(STRATA_EXPF_UFLOW_C) * strata_u2f(STRATA_EXPF_UFLOW_C);
        if (x < strata_u2f(STRATA_EXPF_MAYUFLOW))
            return strata_u2f(STRATA_EXPF_MAYUFLOW_C) * strata_u2f(STRATA_EXPF_MAYUFLOW_C);
        /* else: 80 <= |x| in the representable band -> fall through to core */
    }

    {
        const double shift    = strata_u2d(STRATA_EXP2F_SHIFT);
        const double invln2n  = strata_u2d(STRATA_EXP2F_INVLN2N);
        const double c0       = strata_u2d(STRATA_EXP2F_C0);
        const double c1       = strata_u2d(STRATA_EXP2F_C1);
        const double c2       = strata_u2d(STRATA_EXP2F_C2);
        const double xd       = (double)x;

        double   z  = STRATA_FMA_D(xd, invln2n, shift);   /* vfmadd132sd */
        uint64_t zi = strata_d2u(z);
        double   kd = z - shift;                          /* vsubsd      */
        double   r  = STRATA_FMA_D(xd, invln2n, -kd);     /* vfmsub132sd */
        double   p  = STRATA_FMA_D(c0, r, c1);
        double   q  = STRATA_FMA_D(c2, r, 1.0);
        double   rr = r * r;
        double   poly = STRATA_FMA_D(p, rr, q);
        uint64_t si = (zi << 47) + strata_exp2f_tab[zi & 31u];
        double   s  = strata_u2d(si);

        return STRATA_D2F(poly * s);                      /* vcvtsd2ss */
    }
}

#endif /* STRATA_EXPF_EXACT_H */
