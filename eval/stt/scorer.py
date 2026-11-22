#!/usr/bin/env python3
"""Score a speech-to-text setup for Meeting Memory on quality and compute cost.

The program must define:
    transcribe_chunk(samples: numpy float32 array (-1..1, 16 kHz mono), rate: int, prev_text: str) -> str

The scorer drives it exactly like the app during a recording: 20 s chunks with 1.5 s overlap, the last
30 words passed as prev_text, and the app's overlap de-duplication. Only the Whisper models already on
this machine can be used (tiny.en, base.en, small.en; no downloads).

score = -(WER + 0.005 * CPU seconds per audio minute)   (higher is better)
    One percentage point of word error rate is worth 2 CPU-seconds per audio minute. CPU time (all
    threads) is used instead of wall time so parallel runs do not distort it. A setup that needs more than
    40 CPU-seconds per audio minute is rejected: it could not keep up with a live recording.
--perturb-copy K adds seeded background noise (15-25 dB SNR) and a gain change.
"""
import argparse
import importlib.util
import json
import os
import re
import time
import traceback
import wave
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np  # noqa: E402

ROOT = Path("/Users/monidipa/meeting-memory/eval/stt")
DIRS = [ROOT / "train", ROOT / "data"]
CHUNK, OVERLAP, RATE = 20.0, 1.5, 16000
LAMBDA, MAX_CPU_PER_MIN = 0.005, 40.0
_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten", "12": "twelve", "20th": "twentieth"}
_W = re.compile(r"[^\w']+")


def find(case):
    for d in DIRS:
        if (d / (case + ".wav")).exists():
            return d
    raise FileNotFoundError(case)


def load(case):
    d = find(case)
    with wave.open(str(d / (case + ".wav"))) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    turns = json.loads((d / (case + ".json")).read_text())["turns"]
    return x, " ".join(t["text"] for t in sorted(turns, key=lambda t: t["start"]))


def perturb(x, case, k):
    rng = np.random.default_rng((sum(map(ord, case)) * 1000003 + k) % (2 ** 32))
    p = float(np.mean(x[np.abs(x) > 0.01] ** 2))
    noise = rng.normal(0, np.sqrt(p / 10 ** (rng.uniform(15, 25) / 10)), size=x.shape).astype(np.float32)
    return np.clip((x + noise) * rng.uniform(0.5, 1.2), -1, 1).astype(np.float32)


def _norm(w):
    return _W.sub("", w.lower())


def dedupe_overlap(prev_text, new_text, max_words=15):  # same as backend/core.py
    pw, nw = prev_text.split(), new_text.split()
    if not pw or not nw:
        return new_text.strip()
    p, n = [_norm(w) for w in pw[-max_words:]], [_norm(w) for w in nw[:max_words]]
    for k in range(min(len(p), len(n)), 0, -1):
        if p[-k:] == n[:k] and any(p[-k:]):
            if k == 1 and len(p[-1]) < 5:
                break
            return " ".join(nw[k:]).strip()
    return new_text.strip()


def norm_words(text):
    text = text.lower().replace("%", " percent").replace("-", " ")
    return [_NUM.get(w, w).replace("'", "") for w in re.findall(r"[a-z0-9']+", text)]


def edit_distance(r, h):
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1]


def run_meeting(prog, x):
    size, step = int(CHUNK * RATE), int((CHUNK - OVERLAP) * RATE)
    text = ""
    for start in range(0, len(x), step):
        chunk = x[start:start + size]
        if len(chunk) < RATE // 2:
            break
        raw = str(prog.transcribe_chunk(chunk.copy(), RATE, " ".join(text.split()[-30:])) or "")
        piece = dedupe_overlap(text, raw) if raw.strip() else ""
        text = (text + " " + piece).strip()
        if start + size >= len(x):
            break
    return text


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
        for banned in ("open(", ".json", "eval/stt", "read_text", "glob(", "os.listdir", "Path(", "download_root",
                       "subprocess", "HF_HUB_OFFLINE"):
            if banned in src:
                raise ValueError("programs may not read files, download models or spawn processes (found %r)" % banned)
        spec = importlib.util.spec_from_file_location("candidate", a.program)
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
        cases = sorted(p.stem for p in (ROOT / "train").glob("*.wav")) if a.cases == "all" else a.cases.split(",")
        loaded = {c: load(c) for c in cases}
        if hasattr(prog, "warmup"):
            prog.warmup()
        errs = words = 0
        cpu_total = audio_min = 0.0
        wers = {}
        for c in cases:
            x, ref = loaded[c]
            if a.perturb_copy > 0:
                x = perturb(x, c, a.perturb_copy)
            t0 = time.process_time()
            hyp = run_meeting(prog, x)
            cpu = time.process_time() - t0
            r, h = norm_words(ref), norm_words(hyp)
            e = edit_distance(r, h)
            wers[c] = e / max(1, len(r))
            errs, words = errs + e, words + len(r)
            cpu_total, audio_min = cpu_total + cpu, audio_min + len(x) / RATE / 60
            res["per_case"][c] = -(wers[c] + LAMBDA * cpu / (len(x) / RATE / 60))
        cpu_per_min = cpu_total / max(1e-9, audio_min)
        wer = errs / max(1, words)
        res.update(wer_percent=round(100 * wer, 2), cpu_seconds_per_audio_minute=round(cpu_per_min, 2))
        if cpu_per_min > MAX_CPU_PER_MIN:
            raise ValueError("too slow for live use: %.1f CPU-seconds per audio minute (limit %.0f)"
                             % (cpu_per_min, MAX_CPU_PER_MIN))
        res["reward"] = float(np.mean(list(res["per_case"].values())))
    except Exception as exc:
        res = {"ok": False, "reward": -1.0, "per_case": {}, "error": str(exc),
               "traceback": traceback.format_exc()[-2000:]}
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
