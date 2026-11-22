"""Segment-level confidence cascade: base.en decodes; segments that look unsure are re-decoded by small.en on the audio slice."""
from faster_whisper import WhisperModel

_base = None
_small = None
LOGPROB_T = -0.35
COMP_T = 2.2
PAD = 0.3


def warmup():
    global _base, _small
    if _base is None:
        _base = WhisperModel("base.en", device="cpu", compute_type="int8")
    if _small is None:
        _small = WhisperModel("small.en", device="cpu", compute_type="int8")


def _unsure(s):
    return s.avg_logprob < LOGPROB_T or s.compression_ratio > COMP_T or (s.no_speech_prob > 0.5 and s.avg_logprob < -0.3)


def transcribe_chunk(samples, rate, prev_text):
    warmup()
    segs, _ = _base.transcribe(samples, beam_size=2, initial_prompt=prev_text or None,
                               condition_on_previous_text=False, vad_filter=True,
                               temperature=0.0)
    segs = list(segs)
    out = []
    n = len(samples)
    for s in segs:
        txt = s.text.strip()
        if _unsure(s):
            a = max(0, int((s.start - PAD) * rate))
            b = min(n, int((s.end + PAD) * rate))
            if b - a > int(0.3 * rate):
                prompt = out[-1] if out else (prev_text or None)
                try:
                    sub, _ = _small.transcribe(samples[a:b], beam_size=3, initial_prompt=prompt,
                                               condition_on_previous_text=False, vad_filter=False,
                                               temperature=0.0, without_timestamps=True)
                    t2 = " ".join(x.text.strip() for x in sub).strip()
                    if t2:
                        txt = t2
                except Exception:
                    pass
        if txt:
            out.append(txt)
    return " ".join(out).strip()
