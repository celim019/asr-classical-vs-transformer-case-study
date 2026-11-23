# Classical MFCC + GMM-HMM vs End-to-End Transformer Speech Recognition

Case study for *Speech Recognition and Conversational Systems* (22AIMLDE24), Jain Deemed to be University, B.Tech CSE-AIML, 7th semester.
Option C: Traditional Modular Pipelines vs. End-to-End Deep Learning.

Author: Muhammad Saalim (USN 23BTRCL019, AIML-C)

## What this repository does

It compares one classical recogniser and four end-to-end Transformer configurations on the same 160 utterances (16 speakers, 8 US dialect regions, 8.2 minutes) from the TIMIT sample distributed with NLTK:

| System | WER % | CER % | RTF | Peak RAM (MB) |
|---|---|---|---|---|
| PocketSphinx 5.1.1 (MFCC + GMM-HMM) | 35.6 | 19.2 | 0.210 | 137 |
| Whisper tiny.en int8 | 9.4 | 3.8 | 0.128 | 309 |
| Whisper tiny.en fp32 | 8.9 | 3.6 | 0.155 | 352 |
| Whisper base.en int8 | 13.9 | 8.7 | 0.217 | 456 |
| Whisper base.en fp32 | 6.4 | 2.5 | 0.304 | 602 |

All systems run on one CPU thread. No model is trained or fine-tuned; all 160 utterances are test data. Full numbers (confidence intervals, error types, latency, group analysis) are in `results/summary.json`.

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | Runs one system over the dataset in its own process; stores hypotheses, per-utterance latency and peak memory in `results/<system>.json` |
| `run_all.sh` | Runs all five systems sequentially |
| `analyze.py` | Text normalisation, WER/CER, bootstrap confidence intervals, group analysis, parameter counts, writes `results/summary.json` |
| `figures.py` | Pipeline diagram, trade-off charts, per-speaker plot, annotated spectrogram |
| `requirements.txt` | Python packages |
| `results/summary.json` | Final metrics used in the report |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Data: NLTK TIMIT sample (16 kHz wav + word/phone alignments)
mkdir -p /tmp/work && cd /tmp/work
curl -L -o timit.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/timit.zip
unzip -q timit.zip            # creates /tmp/work/timit

# 2. Models: Whisper tiny.en and base.en exported for sherpa-onnx
for m in tiny.en base.en; do
  curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-$m.tar.bz2
  tar xjf sherpa-onnx-whisper-$m.tar.bz2
done

# 3. Run, analyse, plot (back in this repository)
bash run_all.sh
python analyze.py
python figures.py
```

Paths can be changed with the environment variables `TIMIT_DIR` and `MODEL_DIR` (default `/tmp/work/...`).

Implementation notes:

- PocketSphinx uses its bundled en-us model (PTM GMM-HMM acoustic model, CMUdict lexicon, n-gram language model).
- sherpa-onnx limits Whisper output to 6 tokens per second by default, which truncates fast speakers. `run_experiment.py` raises this to 25 tokens per second for every Whisper run.
- `run_all.sh` sets `OMP_NUM_THREADS=1` and `SANDBOX_SHERPA_ONNX_WHISPER_FILES=0` (the latter only matters in the original sandbox, where it makes the runtime use the unmodified published model files).
- The same text normaliser (lower-case, punctuation removed, digits spelled out) is applied to references and to all systems before scoring with `jiwer`.

## Data and licences

TIMIT sample (LDC93S1) is distributed through NLTK data under CC BY-NC-SA 3.0. It is not included in this repository. Whisper models are released under the MIT licence; PocketSphinx under a BSD-style licence.
# Classical MFCC + GMM-HMM vs End-to-End Transformer Speech Recognition

Case study for *Speech Recognition and Conversational Systems* (22AIMLDE24), Jain Deemed to be University, B.Tech CSE-AIML, 7th semester.
Option C: Traditional Modular Pipelines vs. End-to-End Deep Learning.

Author: Muhammad Saalim (USN 23BTRCL019, AIML-C)

## What this repository does

It compares one classical recogniser and four end-to-end Transformer configurations on the same 160 utterances (16 speakers, 8 US dialect regions, 8.2 minutes) from the TIMIT sample distributed with NLTK:

| System | WER % | CER % | RTF | Peak RAM (MB) |
|---|---|---|---|---|
| PocketSphinx 5.1.1 (MFCC + GMM-HMM) | 35.6 | 19.2 | 0.210 | 137 |
| Whisper tiny.en int8 | 9.4 | 3.8 | 0.128 | 309 |
| Whisper tiny.en fp32 | 8.9 | 3.6 | 0.155 | 352 |
| Whisper base.en int8 | 13.9 | 8.7 | 0.217 | 456 |
| Whisper base.en fp32 | 6.4 | 2.5 | 0.304 | 602 |

All systems run on one CPU thread. No model is trained or fine-tuned; all 160 utterances are test data. Full numbers (confidence intervals, error types, latency, group analysis) are in `results/summary.json`.

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | Runs one system over the dataset in its own process; stores hypotheses, per-utterance latency and peak memory in `results/<system>.json` |
| `run_all.sh` | Runs all five systems sequentially |
| `analyze.py` | Text normalisation, WER/CER, bootstrap confidence intervals, group analysis, parameter counts, writes `results/summary.json` |
| `figures.py` | Pipeline diagram, trade-off charts, per-speaker plot, annotated spectrogram |
| `requirements.txt` | Python packages |
| `results/summary.json` | Final metrics used in the report |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Data: NLTK TIMIT sample (16 kHz wav + word/phone alignments)
mkdir -p /tmp/work && cd /tmp/work
curl -L -o timit.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/timit.zip
unzip -q timit.zip            # creates /tmp/work/timit

# 2. Models: Whisper tiny.en and base.en exported for sherpa-onnx
for m in tiny.en base.en; do
  curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-$m.tar.bz2
  tar xjf sherpa-onnx-whisper-$m.tar.bz2
done

# 3. Run, analyse, plot (back in this repository)
./run_all.sh
python analyze.py
python figures.py
```

Paths can be changed with the environment variables `TIMIT_DIR` and `MODEL_DIR` (default `/tmp/work/...`).

Implementation notes:

- PocketSphinx uses its bundled en-us model (PTM GMM-HMM acoustic model, CMUdict lexicon, n-gram language model).
- sherpa-onnx limits Whisper output to 6 tokens per second by default, which truncates fast speakers. `run_experiment.py` raises this to 25 tokens per second for every Whisper run.
- `run_all.sh` sets `OMP_NUM_THREADS=1` and `SANDBOX_SHERPA_ONNX_WHISPER_FILES=0` (the latter only matters in the original sandbox, where it makes the runtime use the unmodified published model files).
- The same text normaliser (lower-case, punctuation removed, digits spelled out) is applied to references and to all systems before scoring with `jiwer`.

## Data and licences

TIMIT sample (LDC93S1) is distributed through NLTK data under CC BY-NC-SA 3.0. It is not included in this repository. Whisper models are released under the MIT licence; PocketSphinx under a BSD-style licence.
