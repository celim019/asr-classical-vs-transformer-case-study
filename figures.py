#!/usr/bin/env python3
"""Generate every figure used in the report (reads results/*.json, writes figures/*.png)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import librosa, soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
TIMIT = os.environ.get("TIMIT_DIR", "/tmp/work/timit")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
P = json.load(open(os.path.join(ROOT, "results", "per_utterance.json")))
SYS = ["pocketsphinx", "whisper-tiny.en-int8", "whisper-tiny.en-fp32", "whisper-base.en-int8", "whisper-base.en-fp32"]
SHORT = {"pocketsphinx": "PocketSphinx", "whisper-tiny.en-int8": "Whisper tiny.en int8", "whisper-tiny.en-fp32": "Whisper tiny.en fp32",
         "whisper-base.en-int8": "Whisper base.en int8", "whisper-base.en-fp32": "Whisper base.en fp32"}
COL = {"pocketsphinx": "#D55E00", "tiny": "#0072B2", "base": "#009E73"}
def col(s): return COL["pocketsphinx"] if s == "pocketsphinx" else COL["tiny"] if "tiny" in s else COL["base"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ Fig 1: pipeline block diagram
def box(ax, x, y, w, h, text, fc="#F2F2F2", ec="#555", fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1, color="#333"))

def fig_pipeline():
    W, G = 0.98, 0.09                      # box width / gap in inches (axes use inch units)
    fig, ax = plt.subplots(figsize=(7.5, 4.0)); ax.set_xlim(-0.06, 7.44); ax.set_ylim(0, 4.0); ax.axis("off")
    rows = [
        (2.85, "A. Classical modular pipeline (PocketSphinx 5.1.1, en-us)", COL["pocketsphinx"], "#FBE9DD",
         ["Audio input\n16 kHz, 16-bit\nmono PCM", "Framing\n25.6 ms window,\n10 ms hop", "MFCC extraction\n25 mel filters,\nlog + DCT", "Features\n13 MFCC+Δ+ΔΔ\n(39-d), CMN",
          "Acoustic model\nPTM GMM-HMM\n5126 senones", "Decoder + LM\nViterbi beam,\nCMUdict + n-gram", "Text output"]),
        (1.15, "B. End-to-end Transformer (Whisper tiny.en / base.en, ONNX Runtime)", COL["base"], "#DDF2EA",
         ["Audio input\n16 kHz, 16-bit\nmono PCM", "Padding\npad / trim to\n30 s window", "Log-Mel features\n80 bins, 25 ms\nwin, 10 ms hop", "Transformer\nencoder\n(self-attention)",
          "Transformer\ndecoder\n(cross-attention)", "Greedy search\nBPE tokens to\n<|endoftext|>", "Text output"])]
    for y, title, ec, fc, items in rows:
        ax.text(0.0, y + 0.82, title, fontsize=8.4, fontweight="bold", color=ec, va="bottom")
        for i, t in enumerate(items):
            x = i * (W + G); last = i == len(items) - 1
            box(ax, x, y, W, 0.72, t, fc="white" if last else fc, ec="#333" if last else ec, fs=6.0, bold=last)
            if i: arrow(ax, x - G + 0.005, y + 0.36, x - 0.005, y + 0.36)
    ax.text(0.0, 0.55, "In A, acoustic model, lexicon and language model are separate, individually trained components (about 37 MB in total).", fontsize=7.2, style="italic", color="#444")
    ax.text(0.0, 0.30, "In B, one network learns acoustics and language jointly from audio-text pairs (about 38 M / 73 M parameters).", fontsize=7.2, style="italic", color="#444")
    fig.savefig(os.path.join(FIG, "fig1_pipeline.png")); plt.close(fig)


# ------------------------------------------------------------------ Fig 2: accuracy / efficiency trade-off
def fig_tradeoff():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axs[0]
    for s in SYS:
        r = S["systems"][s]; fill = "fp32" in s or s == "pocketsphinx"
        ax.scatter(r["RTF"], r["WER"] * 100, s=r["peak_rss_mb"] * 0.9, facecolor=col(s) if fill else "white", edgecolor=col(s), lw=1.8, zorder=3)
        dx, dy = {"pocketsphinx": (0.018, -0.8), "whisper-tiny.en-int8": (-0.040, 3.6), "whisper-tiny.en-fp32": (0.010, -5.0),
                  "whisper-base.en-int8": (0.017, 2.6), "whisper-base.en-fp32": (-0.045, 4.4)}[s]
        ax.annotate(SHORT[s].replace("Whisper ", ""), (r["RTF"], r["WER"] * 100), (r["RTF"] + dx, r["WER"] * 100 + dy), fontsize=7.5)
    ax.set_xlabel("Real-time factor (RTF, lower = faster)"); ax.set_ylabel("Word error rate (%)")
    ax.set_title("(a) WER vs. speed"); ax.set_xlim(0.08, 0.36); ax.set_ylim(0, 42); ax.grid(alpha=.25)
    ax.text(0.02, 0.97, "marker area ∝ peak RAM\nfilled = fp32 / float, hollow = int8", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color="#444")
    ax = axs[1]
    names = [SHORT[s].replace("Whisper ", "").replace(" ", "\n", 1) for s in SYS]
    cer = [S["systems"][s]["CER"] * 100 for s in SYS]; wer = [S["systems"][s]["WER"] * 100 for s in SYS]
    ci = np.array([[S["systems"][s]["WER"] * 100 - S["systems"][s]["CI"][0] * 100, S["systems"][s]["CI"][1] * 100 - S["systems"][s]["WER"] * 100] for s in SYS]).T
    x = np.arange(len(SYS))
    ax.bar(x - 0.2, wer, 0.38, yerr=ci, capsize=2.5, color=[col(s) for s in SYS], error_kw=dict(lw=1, ecolor="#333"), label="WER (95% bootstrap CI)")
    ax.bar(x + 0.2, cer, 0.38, color=[col(s) for s in SYS], alpha=.45, label="CER")
    for xi, v, up in zip(x - 0.2, wer, ci[1]): ax.text(xi, v + up + 0.9, f"{v:.1f}", ha="center", fontsize=6.8)
    for xi, v in zip(x + 0.2, cer): ax.text(xi + 0.03, v + 0.8, f"{v:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7); ax.set_ylabel("Error rate (%)"); ax.set_title("(b) WER and CER"); ax.set_ylim(0, 48)
    ax.legend(frameon=False, fontsize=7, loc="upper right"); ax.grid(axis="y", alpha=.25); ax.set_ylim(0, 52)
    fig.tight_layout(w_pad=1.6); fig.savefig(os.path.join(FIG, "fig2_tradeoff.png")); plt.close(fig)


# ------------------------------------------------------------------ Fig 3: per-speaker WER (accent / speaker robustness)
def fig_speakers():
    def wer(rows):
        return 100 * sum(r["S"] + r["D"] + r["I"] for r in rows) / sum(r["N"] for r in rows)
    spk = sorted({(r["dr"], r["speaker"]) for r in P["pocketsphinx"]})
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    x = np.arange(len(spk))
    show = ["pocketsphinx", "whisper-tiny.en-fp32", "whisper-base.en-fp32"]
    for s, off, mk in zip(show, (-0.22, 0, 0.22), ("o", "s", "D")):
        v = [wer([r for r in P[s] if r["speaker"] == sp]) for _, sp in spk]
        ax.vlines(x + off, 0, v, color=col(s), lw=1.2, alpha=.5)
        ax.scatter(x + off, v, color=col(s), marker=mk, s=22, zorder=3, label=SHORT[s])
    ax.set_xticks(x); ax.set_xticklabels([f"{sp}\nDR{dr}" for dr, sp in spk], fontsize=5.6)
    ax.set_ylabel("WER per speaker (%)"); ax.set_title("Speaker and dialect-region variability (10 sentences per speaker)")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right"); ax.grid(axis="y", alpha=.25); ax.set_ylim(0, 78)
    for i, (dr, sp) in enumerate(spk):
        if dr % 2 == 0: ax.axvspan(i - 0.5, i + 0.5, color="#000", alpha=.04, lw=0)
    fig.savefig(os.path.join(FIG, "fig3_speakers.png")); plt.close(fig)


# ------------------------------------------------------------------ Fig 4: annotated spectrogram
def read_tx(path):
    return [(int(a) / 16000, int(b) / 16000, l) for a, b, l in (ln.split() for ln in open(path))]

def fig_spectrogram(utt="FVMH0_sx116", folder="dr1-fvmh0", name="sx116"):
    wav = os.path.join(TIMIT, folder, name + ".wav"); y, sr = sf.read(wav, dtype="float32")
    words = read_tx(os.path.join(TIMIT, folder, name + ".wrd")); phones = read_tx(os.path.join(TIMIT, folder, name + ".phn"))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=80, n_mels=96, fmin=0, fmax=8000, power=2.0)
    D = librosa.power_to_db(M, ref=np.max, top_db=70)
    t_end = len(y) / sr
    fig = plt.figure(figsize=(7.4, 4.3)); gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 3.3, 0.5], hspace=0.04)
    axw, axs, axp = [fig.add_subplot(gs[i]) for i in range(3)]
    mel_f = librosa.mel_frequencies(n_mels=96, fmin=0, fmax=8000)
    im = axs.imshow(D, origin="lower", aspect="auto", cmap="magma", extent=[0, t_end, 0, 96], interpolation="nearest")
    def yb(f): return float(np.interp(f, mel_f, np.arange(96)))
    ticks = [0, 500, 1000, 2000, 4000, 8000]
    axs.set_yticks([yb(f) for f in ticks]); axs.set_yticklabels(["0", "0.5", "1", "2", "4", "8"]); axs.set_ylabel("Frequency (kHz, mel axis)")
    axs.set_xlim(0.0, 1.95)
    pr = {w[2]: w for w in words}
    # word strip (colour: words that BOTH systems mis-recognise)
    hyp = {s: next(r for r in P[s] if r["id"] == utt)["hyp"] for s in ("pocketsphinx", "whisper-base.en-fp32")}
    axw.set_xlim(0, 1.95); axw.set_ylim(0, 1); axw.axis("off")
    bad = {"clasp"}
    for a, b, w in words:
        axw.add_patch(Rectangle((a, 0.05), b - a, 0.9, fc="#F7C6A8" if w in bad else "#E9E9E9", ec="#333", lw=.8))
        axw.text((a + b) / 2, 0.5, w, ha="center", va="center", fontsize=8, fontweight="bold" if w in bad else "normal")
        axs.axvline(a, color="white", lw=.4, alpha=.45)
    axp.set_xlim(0, 1.95); axp.set_ylim(0, 1); axp.axis("off")
    for a, b, p in phones:
        if p == "h#": continue
        axp.add_patch(Rectangle((a, 0.05), b - a, 0.9, fc="white", ec="#777", lw=.5))
        if b - a > 0.028: axp.text((a + b) / 2, 0.5, p, ha="center", va="center", fontsize=5.6)
    axp.text(-0.012, 0.5, "TIMIT\nphones", ha="right", va="center", fontsize=6.5)
    axw.text(-0.012, 0.5, "word", ha="right", va="center", fontsize=6.5)
    # annotations (numbers measured from the waveform: see analyze notes in the report, Section 4.3)
    kw = dict(color="white", fontsize=7, ha="left", va="center", arrowprops=dict(arrowstyle="-|>", color="white", lw=.9))
    axs.annotate("(1) /s/ frication (77 ms):\nnoise above 4 kHz", xy=(0.44, yb(6500)), xytext=(0.62, yb(6000)), **kw)
    axs.annotate("(2) word-final /p/: 40 ms closure\n+ 50 ms release, ~21 dB below the\npreceding /s/ -> all five systems\noutput 'class' (deletion)", xy=(0.52, yb(1500)),
                 xytext=(0.70, yb(3500)), **dict(kw, color="#FFE08A", arrowprops=dict(arrowstyle="-|>", color="#FFE08A", lw=.9)))
    axs.annotate("(3) voiced vowel /ae/: harmonic\nstripes, energy mostly < 4 kHz", xy=(1.58, yb(1500)), xytext=(1.30, yb(5000)), **kw)
    axs.set_xticks([])
    axp.axis("on"); axp.spines[["left", "top", "right"]].set_visible(False); axp.set_yticks([])
    axp.set_xticks(np.arange(0, 2.0, 0.25)); axp.tick_params(labelsize=7); axp.set_xlabel("Time (s)", fontsize=8)
    cb = fig.colorbar(im, ax=[axw, axs, axp], pad=0.012, fraction=0.025); cb.set_label("dB re max", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.savefig(os.path.join(FIG, "fig4_spectrogram.png")); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_tradeoff(); fig_speakers(); fig_spectrogram()
    print("figures written:", sorted(os.listdir(FIG)))
