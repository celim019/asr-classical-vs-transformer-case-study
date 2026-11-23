#!/usr/bin/env python3
"""
Case Study (Option C): classical MFCC + GMM-HMM  vs  end-to-end Transformer ASR.

Runs ONE system over the whole TIMIT sample and stores raw hypotheses, per-utterance
latency and peak memory in results/<system>.json.  Each system is run in its own process
so that peak RSS is attributable to that system alone.

    python run_experiment.py --system pocketsphinx
    python run_experiment.py --system whisper-tiny.en-fp32
    python run_experiment.py --system whisper-base.en-int8
"""
import argparse, glob, json, os, re, resource, time, sys
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
TIMIT = os.environ.get("TIMIT_DIR", "/tmp/work/timit")
MODELS = os.environ.get("MODEL_DIR", "/tmp/work")
THREADS = 1  # single thread for every system -> RTF values are directly comparable


def load_dataset():
    """Return list of dicts: id, speaker, dialect region, sex, wav path, reference text."""
    spk = {}
    for line in open(os.path.join(TIMIT, "spkrinfo.txt")):
        if line.startswith(";") or not line.strip():
            continue
        p = line.split()
        spk[p[0].upper()] = dict(sex=p[1], dr=int(p[2]))
    items = []
    for wav in sorted(glob.glob(os.path.join(TIMIT, "dr*", "*.wav"))):
        d = os.path.basename(os.path.dirname(wav))          # e.g. dr1-fvmh0
        sid = d.split("-")[1].upper()
        txt = open(wav[:-4] + ".txt").read().strip()
        ref = re.sub(r"^\d+\s+\d+\s+", "", txt)               # drop "0 54682 " sample counters
        items.append(dict(id=f"{sid}_{os.path.basename(wav)[:-4]}", speaker=sid,
                          sex=sid[0], dr=spk[sid[1:]]["dr"], wav=wav, ref=ref))
    return items


# ---------------------------------------------------------------- recognisers
def make_pocketsphinx():
    from pocketsphinx import Decoder
    dec = Decoder(samprate=16000, loglevel="FATAL")

    def run(x):                                   # x: float32 mono 16 kHz
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        dec.start_utt()
        dec.process_raw(pcm, False, True)
        dec.end_utt()
        h = dec.hyp()
        return h.hypstr if h else ""
    return run


def make_whisper(size, quant):
    import sherpa_onnx
    d = os.path.join(MODELS, f"sherpa-onnx-whisper-{size}")
    sfx = ".int8" if quant == "int8" else ""
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f"{d}/{size}-encoder{sfx}.onnx", decoder=f"{d}/{size}-decoder{sfx}.onnx",
        tokens=f"{d}/{size}-tokens.txt", language="en", task="transcribe",
        num_threads=THREADS, decoding_method="greedy_search")

    def run(x):
        s = rec.create_stream()
        # sherpa-onnx caps Whisper output at 6 tokens/s by default, which truncates fast speakers.
        # Raise the cap so that the decoder stops on <|endoftext|>, not on the safety limit.
        s.set_option("max_tokens_per_second", "25")
        s.accept_waveform(16000, x)
        rec.decode_stream(s)
        return s.result.text
    return run


def build(system):
    if system == "pocketsphinx":
        return make_pocketsphinx()
    m = re.fullmatch(r"whisper-(tiny\.en|base\.en)-(fp32|int8)", system)
    if not m:
        raise SystemExit(f"unknown system {system}")
    return make_whisper(m.group(1), m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    items = load_dataset()
    if a.limit:
        items = items[:a.limit]

    t0 = time.perf_counter()
    run = build(a.system)
    load_s = time.perf_counter() - t0

    # warm-up (first call pays one-off allocation costs) on a clip that is excluded from timing
    w, sr = sf.read(items[0]["wav"], dtype="float32"); assert sr == 16000
    run(w)

    out = []
    for it in items:
        x, sr = sf.read(it["wav"], dtype="float32")
        dur = len(x) / sr
        t = time.perf_counter()
        hyp = run(x)
        lat = time.perf_counter() - t
        out.append(dict(it, hyp=hyp, dur=dur, lat=lat))

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    res = dict(system=a.system, threads=THREADS, load_s=load_s, peak_rss_mb=peak_mb, utts=out)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", f"{a.system}.json"), "w"), indent=1)
    tot_d = sum(o["dur"] for o in out); tot_l = sum(o["lat"] for o in out)
    print(f"{a.system}: {len(out)} utts, {tot_d:.1f}s audio, RTF={tot_l/tot_d:.3f}, "
          f"load={load_s:.2f}s, peakRSS={peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
