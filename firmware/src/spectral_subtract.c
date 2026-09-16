/* spectral_subtract.c - fixed-point noise reduction chain (see header).
 *
 * Every scaling choice here is tied to the reference implementation and to
 * the golden test; changing one without regenerating tables/golden breaks
 * the host test on purpose (fail-closed).
 *
 * No libc dependency: buffer/accumulator shifts are explicit loops so the
 * cross build links with -nostdlib and no shims (hearing-aid firmware).
 */

#include "spectral_subtract.h"

#include "fft.h"
#include "fixed_math.h"
#include "tables.h"

#include <stdint.h>

/* Per-frame scratch. Safe for sequential single-instance use; NOT re-entrant
 * (an interrupt must not preempt nss_process_frame - documented in README). */
static int16_t re[NSS_N_FFT];
static int16_t im[NSS_N_FFT];
static int32_t mag[NSS_BINS];
static q15 gain[NSS_N_FFT];

void nss_init(nss_t *s)
{
    for (int i = 0; i < NSS_N_FFT; i++) {
        s->buffer[i] = 0;
        s->ola_acc[i] = 0;
    }
    for (int k = 0; k < NSS_BINS; k++) {
        s->noise_mag[k] = 0;
    }
    s->state = NSS_COLD_START;
    s->frames_in_noise_est = 0;
    s->total_consumed = 0;
}

void nss_process_frame(nss_t *s, const q15 in[NSS_HOP], q15 out[NSS_HOP])
{
    /* 1. Shift the analysis buffer left by HOP, append the new samples.
     *    Buffer layout matches the reference: newest N samples, frame
     *    boundary at position N-H. */
    for (int i = 0; i < NSS_N_FFT - NSS_HOP; i++) {
        s->buffer[i] = s->buffer[i + NSS_HOP];
    }
    for (int i = 0; i < NSS_HOP; i++) {
        s->buffer[NSS_N_FFT - NSS_HOP + i] = in[i];
    }
    s->total_consumed += NSS_HOP;

    /* 2. Cold start: not a full frame yet -> emit silence (never a partial
     *    artifact; mirrored from the reference, Phase-2 no-op). */
    if (s->state == NSS_COLD_START && s->total_consumed < NSS_N_FFT) {
        for (int i = 0; i < NSS_HOP; i++) out[i] = 0;
        return;
    }
    if (s->state == NSS_COLD_START) {
        s->state = NSS_NOISE_EST;
    }

    /* 3. Window + forward FFT (1/N scaled, matching the reference rfft). */
    for (int n = 0; n < NSS_N_FFT; n++) {
        re[n] = q15_mul(s->buffer[n], WIN_Q15[n]);
        im[n] = 0;
    }
    fft_q15(re, im, NSS_N_FFT);

    /* 4. Magnitude per bin (Q15, saturate at 1.0). */
    for (int k = 0; k < NSS_BINS; k++) {
        int32_t re2 = (int32_t)re[k] * (int32_t)re[k];   /* <= 2^30, safe */
        int32_t im2 = (int32_t)im[k] * (int32_t)im[k];
        int32_t m = (int32_t)u32_sqrt((uint32_t)(re2 + im2)); /* <= ~46340 */
        mag[k] = m > 32767 ? 32767 : m;
    }

    /* 5. Noise estimation (NOISE_EST state): accumulate the magnitude;
     *    at the NOISE_FRAMES-th accumulation, average (>> lb for /8) and
     *    move to ACTIVE. Mirrors the reference exactly (sum-then-divide).
     *    WARNING: >>3 is only correct because NSS_NOISE_FRAMES == 8. */
    if (s->state == NSS_NOISE_EST) {
        for (int k = 0; k < NSS_BINS; k++) {
            s->noise_mag[k] += mag[k];
        }
        s->frames_in_noise_est++;
        if (s->frames_in_noise_est >= NSS_NOISE_FRAMES) {
            for (int k = 0; k < NSS_BINS; k++) {
                s->noise_mag[k] >>= 3;
            }
            s->state = NSS_ACTIVE;
        }
    }

    /* 6. Berouti gain + apply. ACTIVE only; NOISE_EST passes through. */
    if (s->state == NSS_ACTIVE) {
        for (int k = 0; k < NSS_BINS; k++) {
            int32_t denom = mag[k]; if (denom < 1) denom = 1;
            /* r = N/|X| in Q15 (q15_div_q15 contract: |num| <= 32767) */
            int32_t r = q15_div_q15((int32_t)s->noise_mag[k], denom);
            /* alpha*r; alpha=2.0 -> 65536 in Q15; product <= INT32_MAX */
            int32_t ratio = (NSS_ALPHA_Q15 * r + 0x4000) >> 15;
            q15 g = sat16(32768 - ratio);
            gain[k] = (g < NSS_FLOOR_Q15) ? (q15)NSS_FLOOR_Q15 : g;
        }
        for (int k = 1; k < NSS_N_FFT / 2; k++) {
            gain[NSS_N_FFT - k] = gain[k];   /* Hermitian symmetry */
        }
        /* gain[N/2] (Nyquist) was set by the k-loop above (k == N/2). */
        for (int k = 0; k < NSS_N_FFT; k++) {
            re[k] = q15_mul(gain[k], re[k]);
            im[k] = q15_mul(gain[k], im[k]);
        }
    }

    /* 7. Inverse FFT (scale-matched to the reference irfft). */
    ifft_q15(re, im, NSS_N_FFT);

    /* 8. Synthesis window + overlap-add + coverage normalization.
     *    out[p] = ola_acc[p] * inv_cov[p] (Q15) computed with a 32-bit-safe
     *    split: inv_cov = 32768 + extra, so out = acc + (acc*extra)>>15.
     *    This is the arithmetic-overflow guard for the [0.5, 1.0] coverage
     *    region (naive acc*inv_cov would exceed INT32_MAX). */
    for (int j = 0; j < NSS_N_FFT; j++) {
        q15 contrib = q15_mul(re[j], WIN_Q15[j]);
        s->ola_acc[j] += (int32_t)contrib;
    }
    for (int p = 0; p < NSS_HOP; p++) {
        int32_t extra = INV_COV_Q15[p] - 32768;      /* [0, 32768] */
        int32_t acc = s->ola_acc[p];                 /* [-655xx, 655xx] */
        int32_t prod = acc * extra;                  /* <= ~2.147e9 (safe) */
        out[p] = sat16(acc + (prod >> 15));
    }
    /* Shift OLA accumulator left by HOP; zero the tail. */
    for (int i = 0; i < NSS_N_FFT - NSS_HOP; i++) {
        s->ola_acc[i] = s->ola_acc[i + NSS_HOP];
    }
    for (int i = 0; i < NSS_HOP; i++) {
        s->ola_acc[NSS_N_FFT - NSS_HOP + i] = 0;
    }
}