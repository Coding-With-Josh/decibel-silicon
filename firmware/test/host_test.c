/* host_test.c - golden comparison: runs the fixed-point chain over the
 * generated deterministic stream and compares against the validated Python
 * reference output (test/golden.h). Build:  make host-test
 *
 * Gate: fail-closed on gross mismatch. The measured Q15 arithmetic error for
 * this chain (scaled radix-2 FFT + Berouti gain + WOLA) is ~0.36% RMS /
 * ~1.7% max of full scale; a scale, state-machine, or gain bug produces
 * 30-100x larger diffs. GATE_MAX_DIFF is set well above measured round-off
 * and far below any algorithmic break.
 */

#include "spectral_subtract.h"
#include "golden.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GATE_MAX_DIFF 1024   /* ~3.1% of full scale */

int main(void)
{
    nss_t s;
    q15 out[GOLDEN_HOP];
    int max_diff = 0;
    long sum_sq = 0;
    int worst_frame = -1, worst_sample = -1;

    nss_init(&s);

    for (int f = 0; f < GOLDEN_N_FRAMES; f++) {
        nss_process_frame(&s, &GOLDEN_IN_Q15[f * GOLDEN_HOP], out);
        for (int i = 0; i < GOLDEN_HOP; i++) {
            int idx = f * GOLDEN_HOP + i;
            int diff = (int)out[i] - (int)GOLDEN_OUT_Q15[idx];
            if (diff < 0) diff = -diff;
            if (diff > max_diff) {
                max_diff = diff;
                worst_frame = f;
                worst_sample = i;
            }
            sum_sq += (long)diff * diff;
        }
    }

    long n = (long)GOLDEN_N_FRAMES * GOLDEN_HOP;
    double rms = (n > 0) ? sqrt((double)sum_sq / (double)n) : 0.0;

    printf("fixed-point port vs Python reference (%ld samples)\n", n);
    printf("  max abs diff : %d Q15 (%+.5f full scale) at frame %d/%d sample %d\n",
           max_diff, (double)max_diff / 32768.0, worst_frame, worst_sample / GOLDEN_HOP,
           worst_sample % GOLDEN_HOP);
    printf("  rms diff     : %.3f Q15 (%.4f%% of full scale)\n", rms,
           100.0 * rms / 32768.0);

    if (max_diff <= GATE_MAX_DIFF) {
        printf("PASS (max diff %d <= %d Q15)\n", max_diff, GATE_MAX_DIFF);
        return EXIT_SUCCESS;
    }
    printf("FAIL (max diff %d > %d Q15)\n", max_diff, GATE_MAX_DIFF);
    return EXIT_FAILURE;
}