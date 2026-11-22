"""Meeting Memory local speech-to-text setup: one Whisper model, default decoding (MODEL is replaced per founder)."""
from faster_whisper import WhisperModel

MODEL = "base.en"
_model = None


def warmup():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL, device="cpu", compute_type="int8")


def transcribe_chunk(samples, rate, prev_text):
    warmup()
    segs, _ = _model.transcribe(samples, beam_size=5, initial_prompt=prev_text or None,
                                condition_on_previous_text=False, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()
