/* spectral_subtract.h - fixed-point noise reduction (Boll 1979 + Berouti
 * over-subtraction), the firmware counterpart of dsp/src/spectral_subtraction.py.
 *
 * Structural mirror of the validated reference: periodic Hann^2 WOLA with
 * coverage normalization, static noise estimate from the first NOISE_FRAMES
 * frames (leader = noise-only in real use), Berouti gain with spectral floor.
 * States: COLD_START (buffer filling, emits silence) -> NOISE_EST (passthrough
 * while averaging) -> ACTIVE (subtraction).
 */

#ifndef SPECTRAL_SUBTRACT_H
#define SPECTRAL_SUBTRACT_H

#include "fixed_math.h"

#include <stdint.h>

#define NSS_N_FRAMES_STATE_NONE 0

/* Target configuration - MUST match the values used to generate tables.h
 * (N_FFT/HOP) and golden.h (NOISE_FRAMES/alpha/floor). */
#define NSS_N_FFT 128
#define NSS_HOP 64
#define NSS_BINS (NSS_N_FFT / 2 + 1)
#define NSS_NOISE_FRAMES 8
#define NSS_ALPHA_Q15 65536     /* 2.0 in Q15 (int32) */
#define NSS_FLOOR_Q15 328       /* 0.01 in Q15 */

enum nss_state {
    NSS_COLD_START = 0,
    NSS_NOISE_EST = 1,
    NSS_ACTIVE = 2
};

typedef struct {
    q15 buffer[NSS_N_FFT];        /* newest N samples; new samples at the end */
    int32_t ola_acc[NSS_N_FFT];   /* overlap-add accumulator (Q15 units) */
    int32_t noise_mag[NSS_BINS];  /* Q15 magnitudes; avg after NOISE_FRAMES */
    uint8_t state;                /* enum nss_state */
    uint16_t frames_in_noise_est;
    uint32_t total_consumed;
} nss_t;

void nss_init(nss_t *s);

/* Consume exactly NSS_HOP new Q15 samples, emit NSS_HOP Q15 samples. */
void nss_process_frame(nss_t *s, const q15 in[NSS_HOP], q15 out[NSS_HOP]);

#endif /* SPECTRAL_SUBTRACT_H */