/* fixed_math.h - Q15 fixed-point helpers for the noise-reduction port.
 *
 * Target: CV32E40P-class RV32IMC core, no FPU (see firmware/README.md).
 * All fractional values are Q15 (int16_t, -32768..32767 ~= -1.0..~0.99997).
 *
 * Overflow policy (fail-closed): every operation saturates - a saturated
 * result is *wrong but bounded* and stays in range; an overflowing int16
 * result could alias sign and become wildly wrong, so we never let one out.
 */

#ifndef FIXED_MATH_H
#define FIXED_MATH_H

#include <stdint.h>

typedef int16_t q15;

/* Saturate an int32 to int16. */
static inline int16_t sat16(int32_t v)
{
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Saturating Q15 add. */
static inline q15 q15_add(q15 a, q15 b)
{
    return sat16((int32_t)a + (int32_t)b);
}

/* Saturating Q15 multiply (with rounding). a*b in [-1,1) -> Q15. */
static inline q15 q15_mul(q15 a, q15 b)
{
    return sat16(((int32_t)a * (int32_t)b + 0x4000) >> 15);
}

/* Q15 quotient num/den with rounding. Contract: den > 0 and |num| <= 32767
 * (callers pass magnitudes/fractions, both Q15-bounded), so num*32768 stays
 * safely inside int32. Returns clamp(num/den * 32768, +/-32768). */
static inline q15 q15_div_q15(int32_t num, int32_t den)
{
    int32_t q;
    if (den <= 0) return 0;              /* fail-closed: undefined -> 0 */
    if (num > 32767) num = 32767;
    if (num < -32767) num = -32767;
    q = (num * 32768 + (den / 2)) / den;
    return sat16(q);
}

/* 32-bit integer square root (bit-by-bit), returns floor(sqrt(v)).
 * Used for |re, im| magnitude; v up to ~2^31 stays in int32. */
static inline uint32_t u32_sqrt(uint32_t v)
{
    uint32_t res = 0, bit = 1u << 30;
    while (bit > v) bit >>= 2;
    while (bit != 0) {
        if (v >= res + bit) {
            v -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

#endif /* FIXED_MATH_H */