#!/usr/bin/env python3
"""Local engine for Meeting Memory: speaker diarization and local speech-to-text.

Runs in the app's own virtual environment (numpy, scipy, scikit-learn, faster-whisper; see
scripts/setup_engine.py), so the gateway's Python stays standard-library only. core.py calls it as a
subprocess with one JSON request on stdin and reads one JSON reply on stdout.

Requests:
    {"op": "ping"}
    {"op": "diarize", "wav": "/path/meeting.wav"}
        -> {"turns": [[start_s, end_s, "spk0"], ...]}
    {"op": "transcribe", "wav": "/path/chunk.wav", "model": "base.en", "prompt": "...",
     "beam_size": 5, "word_timestamps": false}
        -> {"text": "...", "segments": [[start_s, end_s, "text"], ...], "words": [[s, e, "w"], ...]}
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_MODELS = {}


def read_wav(path):
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono WAV")
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        raise ValueError("expected 16 kHz audio, got %d" % rate)
    return x, rate


def whisper(name):
    if name not in _MODELS:
        from faster_whisper import WhisperModel
        _MODELS[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _MODELS[name]


def op_diarize(req):
    import diarize
    x, rate = read_wav(req["wav"])
    return {"turns": [[float(s), float(e), str(spk)] for s, e, spk in diarize.diarize(x, rate)]}


def op_transcribe(req):
    x, _ = read_wav(req["wav"])
    started = time.time()
    segs, _info = whisper(req.get("model") or "base.en").transcribe(
        x, beam_size=int(req.get("beam_size") or 5), initial_prompt=req.get("prompt") or None,
        word_timestamps=bool(req.get("word_timestamps")), vad_filter=bool(req.get("vad_filter", True)),
        condition_on_previous_text=False)
    segments, words = [], []
    for s in segs:
        segments.append([float(s.start), float(s.end), s.text.strip()])
        for w in (s.words or []):
            words.append([float(w.start), float(w.end), w.word.strip()])
    return {"text": " ".join(t for _, _, t in segments).strip(), "segments": segments, "words": words,
            "seconds": round(time.time() - started, 3)}


def main():
    req = json.loads(sys.stdin.read() or "{}")
    try:
        op = req.get("op")
        if op == "ping":
            import faster_whisper, sklearn  # noqa: F401
            out = {"ok": True}
        elif op == "diarize":
            out = op_diarize(req)
        elif op == "transcribe":
            out = op_transcribe(req)
        else:
            raise ValueError("unknown op %r" % op)
        out.setdefault("ok", True)
    except Exception as exc:  # reported to core.py, which turns it into an AppError
        out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
