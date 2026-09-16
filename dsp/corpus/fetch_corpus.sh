#!/usr/bin/env bash
# Re-fetch and rebuild dsp/corpus from canonical sources.
#
# Produces (in this directory):
#   librispeech_clean_3sent.wav   LibriSpeech dev-clean, 3 sentences -> 23.16 s
#   demand_dkitchen_noise_15s.wav DEMAND DKITCHEN ch1, 15 s slice at 2 s
#
# Depends on: curl, afconvert (macOS), python3 with scipy (dsp/.venv).
# The DEMAND scene zip is ~110 MB; downloads are one-time per machine.
set -euo pipefail

CORPUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # dsp/
DL="$(mktemp -d)"
trap 'rm -rf "$DL"' EXIT

echo ">> librispeech: using OpenSLR dev-clean tarball (http://www.openslr.org/12/)"
curl -sL --max-time 600 -o "$DL/dev-clean.tar.gz" \
  "http://www.openslr.org/resources/12/dev-clean.tar.gz"
# Pick the first three flac files of the first speaker present.
flacs=($(tar -tzf "$DL/dev-clean.tar.gz" | grep '\.flac$' | head -3))
for f in "${flacs[@]}"; do
  tar -xzf "$DL/dev-clean.tar.gz" -C "$DL" "$f"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$DL/$f" "$DL/$(basename "${f%.flac}").wav"
done

echo ">> demand: DKITCHEN_16k.zip from zenodo record 1227121"
curl -sL --max-time 600 -o "$DL/DKITCHEN_16k.zip" \
  "https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip?download=1"
unzip -q -o "$DL/DKITCHEN_16k.zip" -d "$DL/demand"

"$CORPUS_DIR/.venv/bin/python" - "$DL" "$CORPUS_DIR/corpus" <<'EOF'
import sys
from pathlib import Path
from scipy.io import wavfile
import numpy as np

dl = Path(sys.argv[1]); out = Path(sys.argv[2])

def read_afconvert_wav(p: Path):
    sr, x = wavfile.read(p)
    assert sr == 16000 and x.ndim == 1, (p, sr, x.ndim)
    return x.astype(np.float64) / 32768.0

parts = sorted(dl.glob("*.wav"))
assert len(parts) == 3, parts
clean = np.concatenate([read_afconvert_wav(p) for p in parts])
clean /= np.max(np.abs(clean)) or 1.0
wavfile.write(out / "librispeech_clean_3sent.wav", 16000,
              (clean * 32767).astype(np.int16))
print("clean:", round(clean.size / 16000, 2), "s")

nz_files = list((dl / "demand" / "DKITCHEN").glob("ch01.wav"))
assert len(nz_files) == 1
nz = read_afconvert_wav(nz_files[0])
nz = nz[2 * 16000 : 2 * 16000 + 15 * 16000]
nz /= np.max(np.abs(nz)) or 1.0
wavfile.write(out / "demand_dkitchen_noise_15s.wav", 16000,
              (nz * 32767).astype(np.int16))
print("noise:", round(nz.size / 16000, 2), "s")
EOF
echo "done: $(ls -la "$CORPUS_DIR/corpus"/*.wav)"