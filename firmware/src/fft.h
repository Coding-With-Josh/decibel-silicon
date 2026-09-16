/* fft.h - radix-2 DIT complex FFT in Q15 for the noise-reduction port.
 *
 * Scale conventions (chosen to MATCH the validated Python reference so the
 * host test can compare sample-for-sample):
 *   - forward FFT: each stage halves both re/im -> total 1/N scaling.
 *     So fft_q15() of a Q15 windowed frame equals the reference's rfft() / N.
 *   - inverse: conj( fft( conj(X) ) ) then multiply by N -> net unscaled
 *     inverse, equal to the reference's irfft() of the same spectrum.
 */

#ifndef FFT_H
#define FFT_H

#include <stdint.h>

/* fft_q15: in-place radix-2 DIT on re[0..n-1], im[0..n-1].
 * n must be a power of two (128 for this port), twiddles from tables.h
 * (TWIDDLES_Q15 interleaved re,im for k=0..n/2-1). Per-stage halving.
 * Not re-entrant; caller owns its buffers.
 */
void fft_q15(int16_t *re, int16_t *im, int n);

/* ifft_q15: inverse via conjugate trick; output scaled to match the
 * reference irfft() (i.e., no 1/N net scaling on the time result).
 */
void ifft_q15(int16_t *re, int16_t *im, int n);

#endif /* FFT_H */