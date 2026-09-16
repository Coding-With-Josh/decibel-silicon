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
| `demand_dkitchen_noise_15s.wav` | DEMAND scene `DKITCHEN`, channel 2 | CC-BY-SA 3.0 | 15.0 s slice (starting at 2 s), 16 kHz mono, 16-bit |
| `demand_straffic_noise_15s.wav` | DEMAND scene `STRAFFIC`, channel 6 | CC-BY-SA 3.0 | 15.0 s slice (starting at 2 s), 16 kHz mono, 16-bit |
| `demand_tmetro_noise_15s.wav` | DEMAND scene `TMETRO`, channel 6 | CC-BY-SA 3.0 | 15.0 s slice (starting at 2 s), 16 kHz mono, 16-bit |
| `demand_pcafeter_noise_15s.wav` | DEMAND scene `PCAFETER`, channel 6 (HELD OUT) | CC-BY-SA 3.0 | 15.0 s slice (starting at 2 s), 16 kHz mono, 16-bit |

- LibriSpeech: Panayotov et al. 2015, "Librispeech: an ASR corpus based on
  public domain audio books", ICASSP 2015; corpus CC-BY 4.0.
  Canonical source: http://www.openslr.org/12/ (dev-clean).
  The committed clean clip was pulled from the HuggingFace mirror
  `hf-internal-testing/librispeech_asr_dummy` (config `clean`, split
  `validation`) on 2026-09-16.
- DEMAND: Thiemann, Ito, Vincent (2013), "The Diverse Environments
  Multi-channel Acoustic Noise Database (DEMAND)". **License: CC-BY-SA 3.0**
  (Attribution-ShareAlike) per the authors' own text in `DEMAND.pdf`
  (bundled in the zenodo record; license text:
  https://creativecommons.org/licenses/by-sa/3.0/). NOTE: the zenodo
  *metadata* field says `cc-by-4.0`; that is a depositor inconsistency with
  the document the authors ship, and the more restrictive CC-BY-SA 3.0
  governs redistribution of derived slices. Canonical source:
  https://zenodo.org/record/1227121 (`DKITCHEN_16k.zip`, `STRAFFIC_16k.zip`,
  `TMETRO_16k.zip`, `PCAFETER_16k.zip`).

## Held-out noise category (designated BEFORE any training code runs)

- **HELD OUT: `PCAFETER`** (public cafeteria — multi-talker babble). It is the
  closest DEMAND category to the real device target (a hearing aid in a
  crowd) and the type most likely to reproduce the original low-SNR failure
  the synthetic AM-babble test stressed. Keeping it untouched is the honest
  test of generalization: if the learned model improves over the classical
  baseline **on PCAFETER specifically**, that is real generalization
  evidence; if not, the honest claim is narrower and this README's
  designation already committed us to saying so.
- **Training categories:** `DKITCHEN` (domestic, near-stationary),
  `STRAFFIC` (street/open-air, bursty), `TMETRO` (transportation,
  non-stationary). Chosen so training is not "two flavors of steady hum".
- Rule: `PCAFETER` never appears in the training manifest, the validation
  manifest, or any hyperparameter-tuning signal. It is evaluated **exactly
  once**, at the end of the Task 4 table, by scripts that do not import the
  training DataLoader. Moved later = moved never (the loader asserts).
- Eval slices are taken from channels NOT used for training
  (train: ch01/ch05/ch09/ch13; eval: DKITCHEN ch02, others ch06), so no
  eval number can reflect memorization of the exact training audio.

## Label taxonomy (never conflate)

1. **synthetic proxy** — formant-synthesized speech + AM babble (the default
   benchmark signal; deterministic, seeded).
2. **real speech, normal-hearing proxy** — THIS corpus: real LibriSpeech
   speech + real DEMAND noise, but STOI/segSNR still predict *normal*
   intelligibility, not aided-impaired experience. Quoted as
   `--corpus-name "LibriSpeech + DEMAND (CC BY-SA 3.0)"` so the label is in
   the metric line itself.
3. **impaired-hearing validation** — NOT DONE. HASPI/HAAQI or listener
   testing with the NAL-R/NAL-NL2 prescription; the named next milestone.

## Reproduce

```sh
# From dsp/ : real-speech cells at several SNRs (default adaptive v2).
.venv/bin/python benchmark.py \
  --clean corpus/librispeech_clean_3sent.wav \
  --noise corpus/demand_dkitchen_noise_15s.wav \
  --corpus-name "LibriSpeech + DEMAND (CC BY-SA 3.0)" \
  --snr-db 0 --json /tmp/corpus_0db.json
```

## Re-fetch (if the committed wavs need rebuilding)

`corpus/fetch_corpus.sh` downloads from the canonical sources and slices.
The clips are **equivalent, not byte-identical** to the committed ones: the
canonical fetch pulls from OpenSLR dev-clean / zenodo, whereas the committed
artifacts came from the HF mirror rows (clean speech) and the zenodo record
(noise) on 2026-09-16. The committed wavs are what the tables in
`benchmark-methodology.md` / `learned-model.md` quote. Note: each DEMAND 16k
scene zip is ~80-130 MB and LibriSpeech dev-clean is ~330 MB; the committed
outputs are ~0.5-0.8 MB each.

Licenses: LibriSpeech slices are redistributed under CC-BY 4.0; DEMAND
slices under CC-BY-SA 3.0 (per `DEMAND.pdf`; see the license note above).
This README + the fetch script are the attribution. See also
`docs/research/benchmark-methodology.md` (real-speech bridge section) and
`docs/research/learned-model.md` (learned-model evaluation).