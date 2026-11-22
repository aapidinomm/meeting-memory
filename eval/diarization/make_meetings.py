#!/usr/bin/env python3
"""Make synthetic multi-speaker meetings with known "who spoke when" labels.

Uses the free macOS voices (`say`) and ffmpeg, so no cloud calls and no cost.
Each meeting is a 16 kHz mono 16-bit WAV plus a JSON file of reference turns:
    {"speakers": [...], "turns": [{"start": s, "end": s, "speaker": "Daniel"}, ...]}

Synthetic voices are easier to tell apart than real people; treat scores as a
lower bound on the real error.

    python3 eval/diarization/make_meetings.py --out eval/diarization/data
"""
import argparse
import array
import json
import random
import subprocess
import tempfile
import wave
from pathlib import Path

RATE = 16000

# Natural-sounding voices only (the novelty voices are excluded).
TRAIN_VOICES = [
    "Daniel", "Samantha", "Karen", "Moira", "Rishi", "Tessa", "Fred", "Kathy",
    "Eddy (English (US))", "Flo (English (UK))", "Reed (English (US))", "Shelley (English (UK))",
    "Sandy (English (US))", "Grandpa (English (UK))",
]
# Held-out voices: never heard during the search, used only for the locked test set.
TEST_VOICES = [
    "Aman", "Tara", "Ralph", "Rocko (English (UK))", "Grandma (English (US))",
    "Eddy (English (UK))", "Flo (English (US))", "Shelley (English (US))",
]

LINES = [
    "Let's start with the status of the launch.", "I think we are on track for Thursday.",
    "The load tests finished last night and the numbers look fine.",
    "Can we go back to the budget for a second?", "I'm not sure that estimate is right.",
    "We still have two open bugs in the payment flow.", "Who owns the follow up with the design team?",
    "I can take that one.", "That makes sense to me.", "Let's write that down as an action item.",
    "The customer asked for a demo next week.", "We should check with legal before we ship.",
    "My worry is the migration on the old database.", "It took about four hours in staging.",
    "Could we split it into two smaller changes?", "Yes, that would lower the risk.",
    "I'll send the notes after this call.", "Does anyone have questions on the roadmap?",
    "The metrics dashboard is still missing the error rate.", "I added it this morning, it should be there now.",
    "Okay, great, thank you.", "Let's move to the next topic.", "Hiring is going slower than we hoped.",
    "We have three candidates in the final round.", "I liked the second one a lot.",
    "Can you share the interview feedback?", "Sure, I'll post it in the channel.",
    "What about the on call rotation for December?", "I'm out the week of the twentieth.",
    "Then let's swap our weeks.", "Sounds good.", "One more thing about the pricing page.",
    "The new copy tested better with users.", "By how much?", "Roughly twelve percent more sign ups.",
    "That's a big change, let's double check the sample size.", "Agreed, I'll ask the data team.",
    "I think we're done, anything else?", "No, that's all from me.", "Thanks everyone, see you next week.",
]


def synth(voice: str, text: str, tmp: Path) -> array.array:
    aiff, wav = tmp / "t.aiff", tmp / "t.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(RATE),
                    "-sample_fmt", "s16", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    return a


def trim(a: array.array, thresh: int = 300) -> array.array:
    idx = [i for i, v in enumerate(a) if abs(v) > thresh]
    return a[idx[0]:idx[-1] + 1] if idx else a


def make_meeting(rng: random.Random, voices: list, n_turns: int, tmp: Path):
    speakers = rng.sample(voices, rng.randint(2, min(5, len(voices))))
    mix = []           # list of ints, grown as needed
    turns = []
    cursor = int(0.5 * RATE)
    last = None
    for _ in range(n_turns):
        spk = rng.choice([s for s in speakers if s != last] or speakers)
        text = " ".join(rng.sample(LINES, rng.randint(1, 3)))
        clip = trim(synth(spk, text, tmp))
        gain = rng.uniform(0.6, 1.0)
        r = rng.random()
        if r < 0.12 and turns:            # overlap: start before the last turn ends
            start = max(0, cursor - int(rng.uniform(0.3, 1.0) * RATE))
        else:                             # pause
            start = cursor + int(rng.uniform(0.15, 0.9) * RATE)
        end = start + len(clip)
        if end > len(mix):
            mix.extend([0] * (end - len(mix)))
        for i, v in enumerate(clip):
            mix[start + i] += int(v * gain)
        turns.append({"start": round(start / RATE, 3), "end": round(end / RATE, 3), "speaker": spk, "text": text})
        cursor = max(cursor, end)
        last = spk
    mix.extend([0] * int(0.5 * RATE))
    out = array.array("h", (max(-32768, min(32767, v)) for v in mix))
    return speakers, turns, out


def write_wav(path: Path, a: array.array) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(a.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    jobs = [("train%02d" % i, TRAIN_VOICES) for i in range(a.train)] + \
           [("test%02d" % i, TEST_VOICES) for i in range(a.test)]
    with tempfile.TemporaryDirectory() as td:
        for name, voices in jobs:
            speakers, turns, audio = make_meeting(rng, voices, rng.randint(14, 24), Path(td))
            write_wav(out / (name + ".wav"), audio)
            (out / (name + ".json")).write_text(json.dumps({"speakers": speakers, "turns": turns}, indent=1))
            print(name, len(speakers), "speakers", round(len(audio) / RATE, 1), "s")


if __name__ == "__main__":
    main()
