#!/usr/bin/env python3
"""Score a speaker-diarization program on the labeled meetings.

The program must define:
    diarize(samples: numpy.ndarray[float32 in -1..1], rate: int) -> list of (start_s, end_s, speaker_label)

Metric: diarization error rate (DER) = (missed speech + false alarm + wrong speaker) / reference speech,
computed on 10 ms frames, with the best one-to-one matching of predicted to real speakers and a 0.25 s
no-score collar around each reference turn boundary (the usual convention). Overlapped speech is scored.
per_case and reward are -DER (higher is better).

--perturb-copy K adds seeded background noise (12-25 dB SNR) and a gain change, so a version cannot win
by fitting exact sample values.
"""
import argparse
import importlib.util
import json
import traceback
import wave
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

DATA = Path("/Users/monidipa/meeting-memory/eval/diarization/data")
FRAME = 0.01
COLLAR = 0.25


def load(case):
    with wave.open(str(DATA / (case + ".wav"))) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    ref = json.loads((DATA / (case + ".json")).read_text())["turns"]
    return x, rate, ref


def perturb(x, case, k):
    rng = np.random.default_rng((sum(map(ord, case)) * 1000003 + k) % (2 ** 32))
    p = float(np.mean(x[np.abs(x) > 0.01] ** 2)) if np.any(np.abs(x) > 0.01) else 1e-4
    snr = rng.uniform(12, 25)
    noise = rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), size=x.shape).astype(np.float32)
    return np.clip((x + noise) * rng.uniform(0.5, 1.2), -1, 1).astype(np.float32)


def to_frames(turns, n, key):
    labels = sorted({t[key] if isinstance(t, dict) else t[2] for t in turns}, key=str)
    idx = {s: i for i, s in enumerate(labels)}
    m = np.zeros((n, max(1, len(labels))), dtype=bool)
    for t in turns:
        if isinstance(t, dict):
            s, e, spk = t["start"], t["end"], t[key]
        else:
            s, e, spk = float(t[0]), float(t[1]), t[2]
        a, b = max(0, int(round(s / FRAME))), min(n, int(round(e / FRAME)))
        if b > a:
            m[a:b, idx[spk]] = True
    return m


def der(ref_turns, hyp, dur):
    n = int(np.ceil(dur / FRAME)) + 1
    R = to_frames(ref_turns, n, "speaker")
    H = to_frames(hyp, n, "speaker") if hyp else np.zeros((n, 1), dtype=bool)
    score = np.ones(n, dtype=bool)
    c = int(round(COLLAR / FRAME))
    for t in ref_turns:
        for edge in (t["start"], t["end"]):
            f = int(round(edge / FRAME))
            score[max(0, f - c):min(n, f + c)] = False
    R, H = R[score], H[score]
    overlap = R.T.astype(np.int64) @ H.astype(np.int64)
    ri, hi = linear_sum_assignment(-overlap)
    nref, nhyp = R.sum(1), H.sum(1)
    correct = np.zeros(len(R), dtype=np.int64)
    for r, h in zip(ri, hi):
        correct += (R[:, r] & H[:, h])
    total = nref.sum()
    err = np.maximum(nref - nhyp, 0) + np.maximum(nhyp - nref, 0) + (np.minimum(nref, nhyp) - correct)
    return float(err.sum() / max(1, total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb-copy", type=int, default=0)
    a = ap.parse_args()
    res = {"ok": True, "per_case": {}}
    try:
        src = Path(a.program).read_text()
        for banned in ("open(", ".json", "diarization/data", "read_text", "glob(", "os.listdir", "Path("):
            if banned in src:
                raise ValueError("programs may not read files (found %r): labels live next to the audio" % banned)
        spec = importlib.util.spec_from_file_location("candidate", a.program)
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
        cases = sorted(p.stem for p in DATA.glob("train*.wav")) if a.cases == "all" else a.cases.split(",")
        ders = {}
        for case in cases:
            x, rate, ref = load(case)
            if a.perturb_copy > 0:
                x = perturb(x, case, a.perturb_copy)
            hyp = list(prog.diarize(x.copy(), rate))
            ders[case] = der(ref, hyp, len(x) / rate)
            res["per_case"][case] = -ders[case]
        res["reward"] = float(np.mean(list(res["per_case"].values())))
        res["mean_der_percent"] = round(100 * float(np.mean(list(ders.values()))), 2)
    except Exception as exc:
        res = {"ok": False, "reward": -1.0, "per_case": {}, "error": str(exc),
               "traceback": traceback.format_exc()[-2000:]}
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
