# Decibel Silicon

Low-power AI chips for all-day hearing aids.

Hearing aids run on batteries measured in tens of milliwatts. Real-time
noise reduction and speech enhancement are processing-heavy by nature,
so most affordable devices either strip the processing down until it
barely works in a noisy room, or accept a battery that dies before the
day is over. We're building chips designed around that power budget
from day one, not general AI silicon scaled down.

**Live site:** https://decibel-silicon.vercel.app

## Status

Early. Research and a first signal-processing pipeline exist. No
hardware prototype yet. Applied to Founders, Inc.'s Blueprint program
(Sep 30 – Dec 11) to build the first hardware version.

## Repo structure

```
decibel-silicon/
├── apps/
│   └── web/          # Next.js marketing site + waitlist (live at the URL above)
├── dsp/               # Python signal processing research
│   ├── src/
│   │   ├── spectral_subtraction.py   # noise reduction algorithm
│   │   └── pipeline.py               # streaming harness + latency measurement
│   └── tests/
├── firmware/           # placeholder — real-device firmware once hardware exists
├── hardware/             # placeholder — schematics, CAD, BOMs once hardware exists
└── docs/                  # research notes, benchmarks, application writeups
```

`apps/web` and `dsp` are intentionally separate toolchains (a pnpm/Next.js
workspace and a standalone Python package) — they don't share dependencies
and aren't meant to.

## Getting started

### Website (`apps/web`)

```bash
pnpm install
pnpm --filter web dev
```

Runs at `http://localhost:3000`.

### Signal processing (`dsp`)

```bash
cd dsp
pip install numpy scipy --break-system-packages
python -m pytest tests/
```

See `dsp/README.md` for what the current pipeline does and doesn't do yet,
and suggested next steps.

## Team

- **Joshua Idele** ([@josh_scriptz](https://x.com/josh_scriptz)) — AI,
  embedded systems, blockchain. Product and firmware.
- **Bryan Zurix Whyte** — hardware.

High school classmates, building together since a shared obsession with
whether a real-world speed serum from The Flash could actually work.
Decibel Silicon is the first company we're building together.

## Contact

codewithjoshh@gmail.com