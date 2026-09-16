/* fft.c - radix-2 DIT complex FFT in Q15 (see fft.h for scale conventions).
 *
 * Scaling: at the START of every stage (including the first) all values are
 * halved, so every butterfly stays within Q15 and no post-butterfly
 * saturation is needed - the classic "scale per stage" radix-2 form. Total
 * scale after lb stages is 1/N, matching the reference's rfft()/N (the gain
 * ratio N/|X| is scale-invariant, so the golden comparison is unaffected).
 */

#include "fft.h"
#include "fixed_math.h"
#include "tables.h"

#include <stdint.h>

#define MAX_N 512

/* Bit-reversal permutation (in-place via swaps). */
static void bit_reverse(int16_t *re, int16_t *im, int n)
{
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            int16_t tr = re[i]; re[i] = re[j]; re[j] = tr;
            int16_t ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}

/* Halve both components (arithmetic shift; deterministic on the rv32imc
 * target and clang/gcc hosts). */
static void halve(int16_t *re, int16_t *im, int n)
{
    for (int i = 0; i < n; i++) {
        re[i] = (int16_t)(re[i] >> 1);
        im[i] = (int16_t)(im[i] >> 1);
    }
}

void fft_q15(int16_t *re, int16_t *im, int n)
{
    int lb = 0;
    int ntmp = n;
    while (ntmp > 1) { ntmp >>= 1; lb++; }
    if (n < 2 || lb > 9 || n > MAX_N) return;   /* fail-closed: no-op */

    bit_reverse(re, im, n);

    for (int s = 1; s <= lb; s++) {
        int m = 1 << s;                 /* current FFT size */
        int mh = m >> 1;                /* half size */

        /* Scale inputs by 1/2 so butterflies never need saturation. This is
         * the anti-overflow guard for the Q15 butterflies (Phase-3 class:
         * arithmetic overflow); sat16 below is a never-triggered backstop. */
        halve(re, im, n);

        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < mh; j++) {
                /* twiddle index: w_j^(m) = W^(j * n/m) = table[j * (n/m)] */
                int tw_idx = j * (n / m);
                int16_t wr = TWIDDLES_Q15[2 * tw_idx];
                int16_t wi = TWIDDLES_Q15[2 * tw_idx + 1];

                int32_t t_re = (int32_t)q15_mul(wr, re[k + j + mh]) -
                               (int32_t)q15_mul(wi, im[k + j + mh]);
                int32_t t_im = (int32_t)q15_mul(wr, im[k + j + mh]) +
                               (int32_t)q15_mul(wi, re[k + j + mh]);

                re[k + j + mh] = sat16((int32_t)re[k + j] - t_re);
                im[k + j + mh] = sat16((int32_t)im[k + j] - t_im);
                re[k + j] = sat16((int32_t)re[k + j] + t_re);
                im[k + j] = sat16((int32_t)im[k + j] + t_im);
            }
        }
    }
}

void ifft_q15(int16_t *re, int16_t *im, int n)
{
    /* inverse via conjugate: ifft(X) = N * conj( fft( conj(X) ) ) */
    for (int i = 0; i < n; i++) im[i] = (int16_t)(-im[i]);
    fft_q15(re, im, n);
    for (int i = 0; i < n; i++) im[i] = (int16_t)(-im[i]);
    /* fft_q15 scaled by 1/N; multiply by N to undo (net unscaled inverse,
     * matching numpy's irfft). N <= 512, int32 intermediates safe. */
    for (int i = 0; i < n; i++) {
        re[i] = sat16((int32_t)re[i] * n);
        im[i] = sat16((int32_t)im[i] * n);
    }
}