#!/usr/bin/env python3
"""Compare speech-to-text models on meetings with a known script: quality, cost and speed.

Each model transcribes the test meetings exactly the way the app does during a recording (chunks of
chunk_seconds with overlap_seconds overlap, previous-text context, overlap de-duplication), so the numbers
match real use. Reports, per model:

    wer                        word error rate against the script (lower is better)
    usd_per_hour               Bedrock price from measured tokens (local models: $0)
    seconds_per_audio_minute   wall time per minute of audio

and stores the report as the app's stt_eval.json; the app then lists every model with these numbers in
Settings and, unless you picked a model yourself, switches to the recommended one (the cheapest model
within one percentage point of the best word error rate).

    python3 eval/stt/compare.py --data-dir ~/.kiro/crew/apps/meeting-memory/data \
        --aws-profile meeting-memory --aws-region us-west-2

Bedrock models make real, billed calls (about 10 minutes of audio per model; well under $0.05 at
2026-11-22 prices). Without --aws-profile only local models run. The test meetings are synthetic voices
(eval/diarization/make_meetings.py); real meetings will have higher error rates for every model.
"""
import argparse
import array
import json
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import core  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten", "12": "twelve", "20th": "twentieth"}


def norm_words(text):
    text = text.lower().replace("%", " percent").replace("-", " ")
    words = re.findall(r"[a-z0-9']+", text)
    return [_NUM.get(w, w).replace("'", "") for w in words]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1], len(r)


def load(case):
    with wave.open(str(DATA / (case + ".wav"))) as w:
        pcm = array.array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    turns = json.loads((DATA / (case + ".json")).read_text())["turns"]
    ref = " ".join(t["text"] for t in sorted(turns, key=lambda t: t["start"]))
    return pcm, ref


def transcribe_like_the_app(settings, data_dir, model, pcm):
    rate, size = 16000, int(settings["chunk_seconds"] * 16000)
    step = size - int(settings["overlap_seconds"] * 16000)
    text, toks_in, toks_out = "", 0, 0
    for start in range(0, len(pcm), step):
        chunk = pcm[start:start + size]
        if len(chunk) < rate // 2:
            break
        raw, usage = core.transcribe_wav(settings, core.wav_bytes(chunk), " ".join(text.split()[-30:]),
                                         data_dir=data_dir, model=model)
        toks_in += int(usage.get("inputTokens") or 0)
        toks_out += int(usage.get("outputTokens") or 0)
        piece = core.dedupe_overlap(text, raw) if raw else ""
        text = (text + " " + piece).strip()
        if start + size >= len(pcm):
            break
    return text, toks_in, toks_out


def main():
    ap = argparse.ArgumentParser(description="Compare speech-to-text models for Meeting Memory")
    ap.add_argument("--data-dir", type=Path, required=True, help="the app's data dir (engine, settings)")
    ap.add_argument("--aws-profile", default="")
    ap.add_argument("--aws-region", default="")
    ap.add_argument("--models", nargs="*", help="model ids (default: every installed local model, plus Bedrock "
                                                "models when --aws-profile is given)")
    ap.add_argument("--no-apply", action="store_true", help="store the report but do not change stt_model")
    a = ap.parse_args()
    settings = core.load_settings(a.data_dir)
    if a.aws_profile:
        settings["aws_profile"] = a.aws_profile
    if a.aws_region:
        settings["aws_region"] = a.aws_region
    models = a.models or [m["id"] for m in core.STT_MODELS
                          if (m["kind"] == "bedrock" and a.aws_profile) or
                          (m["kind"] == "local" and m["id"] != "local:large-v3-turbo")]
    price = {m["id"]: m for m in core.STT_MODELS}
    cases = sorted(p.stem for p in DATA.glob("*.wav"))
    if not cases:
        sys.exit("no test meetings in %s: run eval/diarization/make_meetings.py --out %s --train 0 --test 6 "
                 "--seed 11" % (DATA, DATA))
    audio = {c: load(c) for c in cases}
    minutes = sum(len(p) for p, _ in audio.values()) / 16000 / 60
    results = []
    for model in models:
        row = {"model": model, "ok": True}
        errs = total = tin = tout = 0
        started = time.time()
        try:
            per_case = {}
            for c in cases:
                pcm, ref = audio[c]
                hyp, i, o = transcribe_like_the_app(settings, a.data_dir, model, pcm)
                e, n = wer(ref, hyp)
                per_case[c] = round(e / max(1, n), 4)
                errs, total, tin, tout = errs + e, total + n, tin + i, tout + o
            secs = time.time() - started
            p = price.get(model, {})
            usd = (tin * p.get("price_in", 0) + tout * p.get("price_out", 0)) / 1e6
            row.update(wer=round(errs / max(1, total), 4), per_case=per_case, input_tokens=tin, output_tokens=tout,
                       usd_per_hour=round(usd / minutes * 60, 4),
                       seconds_per_audio_minute=round(secs / minutes, 2))
        except core.AppError as exc:
            row.update(ok=False, error=exc.message)
        results.append(row)
        print("%-32s %s" % (model, json.dumps({k: row.get(k) for k in ("wer", "usd_per_hour",
                                                                          "seconds_per_audio_minute", "error")})),
              flush=True)
    report = {"evaluated_at": time.time(), "results": results,
              "test_set": {"meetings": len(cases), "minutes": round(minutes, 1), "voices": "synthetic (macOS)",
                           "chunk_seconds": settings["chunk_seconds"], "overlap_seconds": settings["overlap_seconds"]}}
    report = core.save_stt_eval(a.data_dir, report, apply=not a.no_apply)
    print("recommended:", report["recommended"])
    print("stt_model now:", core.load_settings(a.data_dir)["stt_model"])


if __name__ == "__main__":
    main()
