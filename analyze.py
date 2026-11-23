#!/usr/bin/env python3
"""Compute every number, table and figure used in the report from results/*.json."""
import json, os, re, glob, struct
import numpy as np
import jiwer
from num2words import num2words

ROOT = os.path.dirname(os.path.abspath(__file__))
TIMIT = os.environ.get("TIMIT_DIR", "/tmp/work/timit")
MODELS = os.environ.get("MODEL_DIR", "/tmp/work")
SYSTEMS = ["pocketsphinx", "whisper-tiny.en-int8", "whisper-tiny.en-fp32",
           "whisper-base.en-int8", "whisper-base.en-fp32"]
LABEL = {"pocketsphinx": "PocketSphinx (MFCC + GMM-HMM)",
         "whisper-tiny.en-int8": "Whisper tiny.en (int8)", "whisper-tiny.en-fp32": "Whisper tiny.en (fp32)",
         "whisper-base.en-int8": "Whisper base.en (int8)", "whisper-base.en-fp32": "Whisper base.en (fp32)"}
SHORT = {"pocketsphinx": "PocketSphinx", "whisper-tiny.en-int8": "tiny.en int8", "whisper-tiny.en-fp32": "tiny.en fp32",
         "whisper-base.en-int8": "base.en int8", "whisper-base.en-fp32": "base.en fp32"}


# ------------------------------------------------------------------ text normalisation
def _num(m):
    n = int(m.group(1))
    return num2words(n).replace("-", " ").replace(",", "") + (m.group(2) or "")

def norm(t):
    """One normaliser for references AND hypotheses of every system."""
    t = t.lower().replace("’", "'")
    t = re.sub(r"(\d+)('s)?", _num, t)          # 22 -> "twenty two", 1's -> "one's"
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)  # quotes that are not apostrophes
    return re.sub(r"\s+", " ", t).strip()


def per_utt_counts(ref, hyp):
    o = jiwer.process_words(ref, hyp)
    return dict(S=o.substitutions, D=o.deletions, I=o.insertions, H=o.hits, N=o.substitutions + o.deletions + o.hits,
                align=o.alignments[0], rt=o.references[0], ht=o.hypotheses[0])


def corpus_rates(rows):
    S = sum(r["S"] for r in rows); D = sum(r["D"] for r in rows); I = sum(r["I"] for r in rows); N = sum(r["N"] for r in rows)
    return dict(WER=(S + D + I) / N, S=S / N, D=D / N, I=I / N, N=N)


def bootstrap_ci(rows, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    e = np.array([r["S"] + r["D"] + r["I"] for r in rows]); N = np.array([r["N"] for r in rows])
    idx = rng.integers(0, len(rows), (n, len(rows)))
    w = e[idx].sum(1) / N[idx].sum(1)
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def load_wrd(utt):
    d = {"F": "", "M": ""}
    spk = utt["speaker"].lower(); name = utt["id"].split("_", 1)[1]
    path = os.path.join(TIMIT, f"dr{utt['dr']}-{spk}", name + ".wrd")
    return [(int(a) / 16000, int(b) / 16000, w) for a, b, w in (l.split() for l in open(path))]


def sphinx_params():
    base = os.path.join(os.path.dirname(__import__("pocketsphinx").__file__), "model", "en-us")
    am = os.path.join(base, "en-us")
    def n_floats(f):
        b = open(f, "rb").read(); i = b.index(b"endhdr\n") + 7
        _, n_mgau, n_feat, n_den = struct.unpack("<4I", b[i:i + 16])     # byte-order word, codebooks, streams, densities
        return n_mgau, n_feat, n_den
    mg, nf, nd = n_floats(os.path.join(am, "means"))
    b = open(os.path.join(am, "mdef"), "rb").read()                      # binary mdef: BMDF, version, strlen, text, counts
    slen = struct.unpack("<I", b[8:12])[0]
    n_ci, n_tri, n_emit, n_ci_sen, n_sen, n_tmat = struct.unpack("<6I", b[12 + slen:12 + slen + 24])
    gauss_floats = os.path.getsize(os.path.join(am, "means")) // 4          # ~ 39-d means
    files = [os.path.join(am, f) for f in os.listdir(am)] + [os.path.join(base, "en-us.lm.bin"), os.path.join(base, "cmudict-en-us.dict")]
    size_mb = sum(os.path.getsize(f) for f in files) / 1e6
    am_mb = sum(os.path.getsize(os.path.join(am, f)) for f in os.listdir(am)) / 1e6
    # parameters: means + variances + mixture weights (senones x streams x densities)
    params = 2 * gauss_floats + n_sen * nf * nd
    return dict(params=params, size_mb=size_mb, am_mb=am_mb, n_sen=n_sen, n_tri=n_tri, mgau=mg, nfeat=nf, nden=nd,
                lm_mb=os.path.getsize(os.path.join(base, "en-us.lm.bin")) / 1e6)


def whisper_params(size):
    import onnx
    d = os.path.join(MODELS, f"sherpa-onnx-whisper-{size}")
    tot = 0
    for part in ("encoder", "decoder"):
        m = onnx.load(os.path.join(d, f"{size}-{part}.onnx"), load_external_data=True)
        tot += sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    return tot


def whisper_size_mb(size, q):
    d = os.path.join(MODELS, f"sherpa-onnx-whisper-{size}"); s = ".int8" if q == "int8" else ""
    return sum(os.path.getsize(os.path.join(d, f"{size}-{p}{s}.onnx")) for p in ("encoder", "decoder")) / 1e6


def main():
    R = {s: json.load(open(os.path.join(ROOT, "results", f"{s}.json"))) for s in SYSTEMS}
    out = dict(systems={}, groups={}, dur={})
    first = R[SYSTEMS[0]]["utts"]
    n = len(first)
    total_dur = sum(u["dur"] for u in first)
    speakers = sorted({u["speaker"] for u in first})
    out["dataset"] = dict(n_utts=n, total_dur_s=total_dur, n_speakers=len(speakers),
                          n_female=sum(1 for s in speakers if s[0] == "F"), n_male=sum(1 for s in speakers if s[0] == "M"),
                          mean_dur=total_dur / n, min_dur=min(u["dur"] for u in first), max_dur=max(u["dur"] for u in first),
                          n_words=0, n_unique_sent=len({norm(u["ref"]) for u in first}))
    sp = sphinx_params()
    out["sphinx"] = sp
    wp = {"tiny.en": whisper_params("tiny.en"), "base.en": whisper_params("base.en")}
    out["whisper_params"] = wp

    per = {}
    for s in SYSTEMS:
        rows = []
        cer_r, cer_h = [], []
        for u in R[s]["utts"]:
            ref, hyp = norm(u["ref"]), norm(u["hyp"])
            c = per_utt_counts(ref, hyp)
            c.update(id=u["id"], speaker=u["speaker"], sex=u["sex"], dr=u["dr"], ref=ref, hyp=hyp, raw_ref=u["ref"], raw_hyp=u["hyp"],
                     dur=u["dur"], lat=u["lat"])
            rows.append(c); cer_r.append(ref); cer_h.append(hyp)
        per[s] = rows
        r = corpus_rates(rows)
        lo, hi = bootstrap_ci(rows)
        cer = jiwer.cer(cer_r, cer_h)
        lat = np.array([u["lat"] for u in R[s]["utts"]]); dur = np.array([u["dur"] for u in R[s]["utts"]])
        # loop guard: utterances whose hypothesis is >3x the reference length (decoder repetition loops)
        loops = [x["id"] for x in rows if len(x["ht"]) > 3 * len(x["rt"]) + 3]
        rows_noloop = [x for x in rows if x["id"] not in loops]
        out["systems"][s] = dict(**r, CI=[lo, hi], CER=cer, RTF=float(lat.sum() / dur.sum()), lat_mean_ms=float(lat.mean() * 1e3),
                                 lat_p95_ms=float(np.percentile(lat, 95) * 1e3), load_s=R[s]["load_s"], peak_rss_mb=R[s]["peak_rss_mb"],
                                 loops=loops, WER_noloop=corpus_rates(rows_noloop)["WER"],
                                 exact=float(np.mean([x["S"] + x["D"] + x["I"] == 0 for x in rows])))
        if s == "pocketsphinx":
            out["systems"][s].update(params=sp["params"], size_mb=sp["size_mb"])
        else:
            m = re.fullmatch(r"whisper-(tiny\.en|base\.en)-(fp32|int8)", s)
            out["systems"][s].update(params=wp[m.group(1)], size_mb=whisper_size_mb(m.group(1), m.group(2)))
    out["dataset"]["n_words"] = out["systems"]["pocketsphinx"]["N"]

    # ---------------- group analysis (dialect region, sex)
    for s in SYSTEMS:
        g = {}
        for key in ("dr", "sex"):
            for v in sorted({x[key] for x in per[s]}):
                rows = [x for x in per[s] if x[key] == v]
                g[f"{key}={v}"] = corpus_rates(rows)["WER"]
        out["groups"][s] = g

    # ---------------- word-duration analysis from TIMIT forced alignments
    bins = [(0, 0.15, "< 150 ms"), (0.15, 0.30, "150-300 ms"), (0.30, 9, "> 300 ms")]
    use = 0
    stat = {s: {b[2]: [0, 0] for b in bins} for s in SYSTEMS}
    stat_oov = {s: [0, 0] for s in SYSTEMS}
    dictw = set(l.split()[0].split("(")[0] for l in open(os.path.join(os.path.dirname(__import__("pocketsphinx").__file__),
                "model", "en-us", "cmudict-en-us.dict")))
    utt_by_id = {u["id"]: u for u in first}
    n_oov_tokens = 0; n_tokens = 0
    for i, u in enumerate(first):
        wrd = load_wrd(u)
        reft = per["pocketsphinx"][i]["rt"]
        wt = [norm(w).split() for _, _, w in wrd]
        flat = [t for ws in wt for t in ws]
        if flat != reft:
            continue
        use += 1
        durs = []
        for (a, b, w), ws in zip(wrd, wt):
            durs += [b - a] * len(ws)
        n_tokens += len(reft); n_oov_tokens += sum(t not in dictw for t in reft)
        for s in SYSTEMS:
            x = per[s][i]
            err = np.zeros(len(reft), bool)
            for ch in x["align"]:
                if ch.type in ("substitute", "delete"):
                    err[ch.ref_start_idx:ch.ref_end_idx] = True
            for k, t in enumerate(reft):
                for lo_, hi_, name in bins:
                    if lo_ <= durs[k] < hi_:
                        stat[s][name][0] += int(err[k]); stat[s][name][1] += 1
                if t not in dictw:
                    stat_oov[s][0] += int(err[k]); stat_oov[s][1] += 1
    out["dur"] = {s: {b: (v[0] / v[1] if v[1] else None, v[1]) for b, v in stat[s].items()} for s in SYSTEMS}
    out["oov"] = dict(n_tokens=n_tokens, n_oov=n_oov_tokens, utts_aligned=use,
                      err={s: (stat_oov[s][0] / stat_oov[s][1] if stat_oov[s][1] else None) for s in SYSTEMS})

    json.dump(out, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1, default=float)
    json.dump({s: [{k: v for k, v in x.items() if k not in ("align",)} for x in per[s]] for s in SYSTEMS},
              open(os.path.join(ROOT, "results", "per_utterance.json"), "w"), indent=1, default=str)
    d = out["dataset"]
    print(f"dataset: {d['n_utts']} utts, {d['total_dur_s']/60:.2f} min, {d['n_speakers']} spk, {d['n_words']} ref words, {d['n_unique_sent']} unique sentences")
    print("sphinx:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sp.items()})
    print("whisper params:", wp)
    for s in SYSTEMS:
        r = out["systems"][s]
        print(f"{SHORT[s]:14s} WER {r['WER']*100:5.2f} [{r['CI'][0]*100:.1f},{r['CI'][1]*100:.1f}]  S {r['S']*100:.1f} D {r['D']*100:.1f} I {r['I']*100:.1f}"
              f"  CER {r['CER']*100:5.2f}  RTF {r['RTF']:.3f}  lat {r['lat_mean_ms']:.0f}/{r['lat_p95_ms']:.0f}ms  RSS {r['peak_rss_mb']:.0f}  size {r['size_mb']:.1f}MB"
              f"  params {r['params']/1e6:.1f}M  exact {r['exact']*100:.0f}%  loops {r['loops']}  WERnoloop {r['WER_noloop']*100:.2f}")
    print("groups:", json.dumps({SHORT[s]: {k: round(v * 100, 1) for k, v in g.items()} for s, g in out["groups"].items()}, indent=0))
    print("dur-bins:", json.dumps({SHORT[s]: {b: (round(v[0] * 100, 1) if v[0] is not None else None, v[1]) for b, v in dd.items()} for s, dd in out["dur"].items()}))
    print("oov:", out["oov"])


if __name__ == "__main__":
    main()
