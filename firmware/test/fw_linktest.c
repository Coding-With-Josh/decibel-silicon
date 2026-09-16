/* fw_linktest.c - freestanding link check for the rv32imc cross build.
 * Build:  make cross
 * Exercises the whole port with no libc: nss_init + one full frame + return.
 * A _start stub is provided because the target has no CRT.
 */

#include "spectral_subtract.h"

static q15 in[NSS_HOP];
static q15 out[NSS_HOP];

void _start(void)
{
    nss_t s;
    nss_init(&s);
    for (int i = 0; i < NSS_HOP; i++) in[i] = 0;
    nss_process_frame(&s, in, out);
    /* keep the link from eliding the call chain (no libc: store to a
     * volatile sink that lives in .data, not .bss) */
    __asm__ volatile("" : : "r"(out[0]) : "memory");
    for (;;) { /* bare-metal: never return */ }
}