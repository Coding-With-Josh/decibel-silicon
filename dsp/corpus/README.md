# Real-speech/real-noise bridge corpus (normal-hearing proxy)

Two small, permissively licensed files used by the DSP benchmark's
`--clean/--noise/--corpus-name` path. This is a **bridge**, not the final
validation: it replaces the synthetic formant signal with real speech for the
*normal-hearing* proxy metrics (STOI/segSNR). It does **not** answer what a
person with a hearing-aid prescription experiences — that is the
impaired-hearing step (HASPI/HAAQI or listener testing) and is an **open
next step**, deliberately not conflated with this corpus.

## Files

| file | source | license | content |
|---|---|---|---|
| `librispeech_clean_3sent.wav` | LibriSpeech dev-clean (3 sentences, concatenated) | CC-BY 4.0 | 23.16 s, 16 kHz mono, 16-bit, no leading silence added |
| `demand_dkitchen_noise_15s.wav` | DEMAND scene `DKITCHEN`, channel 1 | CC-BY 4.0 | 15.0 s slice (starting at 2 s), 16 kHz mono, 16-bit |

- LibriSpeech: Panayotov et al. 2015, "Librispeech: an ASR corpus based on
  public domain audio books", ICASSP 2015; corpus CC-BY 4.0.
  Canonical source: http://www.openslr.org/12/ (dev-clean).
  The committed clip was pulled from the HuggingFace mirror
  `hf-internal-testing/librispeech_asr_dummy` (config `clean`, split
  `validation`) on 2026-09-16.
- DEMAND: Thiemann, Ito, Vincent (2013), "The Diverse Environments
  Multi-channel Acoustic Noise Database (DEMAND)"; CC-BY 4.0.
  Canonical source: https://zenodo.org/record/1227121
  (`DKITCHEN_16k.zip`; the committed slice is channel 1, starting at 2 s).

## Label taxonomy (never conflate)

1. **synthetic proxy** — formant-synthesized speech + AM babble (the default
   benchmark signal; deterministic, seeded).
2. **real speech, normal-hearing proxy** — THIS corpus: real LibriSpeech
   speech + real DEMAND noise, but STOI/segSNR still predict *normal*
   intelligibility, not aided-impaired experience. Quoted as
   `--corpus-name "LibriSpeech + DEMAND (CC-BY 4.0)"` so the label is in
   the metric line itself.
3. **impaired-hearing validation** — NOT DONE. HASPI/HAAQI or listener
   testing with the NAL-R/NAL-NL2 prescription; the named next milestone.

## Reproduce

```sh
# From dsp/ : real-speech cells at several SNRs (default adaptive v2).
.venv/bin/python benchmark.py \
  --clean corpus/librispeech_clean_3sent.wav \
  --noise corpus/demand_dkitchen_noise_15s.wav \
  --corpus-name "LibriSpeech + DEMAND (CC-BY 4.0)" \
  --snr-db 0 --json /tmp/corpus_0db.json
```

## Re-fetch (if the committed wavs need rebuilding)

`corpus/fetch_corpus.sh` downloads from the canonical sources and slices.
The clips are **equivalent, not byte-identical** to the committed ones: the
canonical fetch pulls from OpenSLR dev-clean / zenodo, whereas the committed
artifacts came from the HF mirror rows on 2026-09-16 (same corpus, different
sentences/offsets). The committed wavs are what the tables in
`benchmark-methodology.md` quote. Note: the DEMAND 16k scene zip is ~110 MB;
the committed outputs are ~0.7 MB + ~0.5 MB.

Licenses: the committed files are redistributed under the same CC-BY 4.0
terms; this README + the fetch script are the attribution. See also
`docs/research/benchmark-methodology.md` (real-speech bridge section).