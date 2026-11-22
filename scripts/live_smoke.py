#!/usr/bin/env python3
"""Live end-to-end check against Bedrock: synthesize a short meeting with Amazon Polly, cut it into overlapping
16 kHz WAV chunks the way the browser recorder does, then transcribe, summarize, index, search and ask.

    python3 scripts/live_smoke.py --aws-profile <name> [--aws-region us-west-2]

Uses a throwaway data dir (printed at the end); needs boto3, ffmpeg, and polly:SynthesizeSpeech on the profile.
Makes real, billed Bedrock calls (a few cents).
"""

import argparse
import array
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mm_core", str(ROOT / "backend" / "core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

SCRIPT = [
    ("Joanna", "Okay, let's start the weekly sync for the Kirin data flywheel. First topic is the reward registry."),
    ("Matthew", "I reviewed the reward registry design doc. I think we should keep the extractor inventory separate "
                "from the registry, and version every reward function."),
    ("Joanna", "Agreed. Decision: we version every reward function in the registry. Monidipa will draft the "
               "schema by next Wednesday."),
    ("Matthew", "Second topic. The prod pipeline promotion is still disabled, so we are deploying manually. "
                "Who owns turning it back on?"),
    ("Joanna", "Thanh will look into re-enabling promotion this week. Open question: do we need a new approval "
               "workflow before we turn it back on? Let's revisit Friday."),
]


def pcm16(data_dir: Path, profile: str, region: str) -> array.array:
    import boto3
    polly = boto3.Session(profile_name=profile, region_name=region).client("polly")
    out = array.array("h")
    for i, (voice, text) in enumerate(SCRIPT):
        audio = polly.synthesize_speech(Text=text, OutputFormat="pcm", SampleRate="16000", VoiceId=voice)["AudioStream"].read()
        part = array.array("h")
        part.frombytes(audio)
        out.extend(part)
        out.extend([0] * 8000)  # half a second between speakers
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws-profile", required=True)
    ap.add_argument("--aws-region", default="us-west-2")
    ap.add_argument("--keep", action="store_true", help="keep the throwaway data dir")
    a = ap.parse_args()
    data = Path(tempfile.mkdtemp(prefix="mm-smoke-"))
    settings = core.save_settings(data, {"aws_profile": a.aws_profile, "aws_region": a.aws_region,
                                         "vocabulary": ["Kirin", "Monidipa", "Thanh"]})
    pcm = pcm16(data, a.aws_profile, a.aws_region)
    rate, chunk, overlap = 16000, int(settings["chunk_seconds"] * 16000), int(settings["overlap_seconds"] * 16000)
    print("audio: %.1f s, chunk %d s, overlap %.1f s" % (len(pcm) / rate, settings["chunk_seconds"], settings["overlap_seconds"]))
    m = core.create_meeting(data, "")
    seq, pos, t0 = 0, 0, time.time()
    while pos < len(pcm):
        start = max(0, pos - overlap)
        samples = pcm[start:pos + chunk]
        row = core.add_chunk(data, settings, m["id"], seq, start * 1000 // rate, len(samples) * 1000 // rate,
                             core.wav_bytes(samples))
        print("chunk %d [%s] %d ms: %s" % (seq, core._fmt_ts(row["start_ms"]), row["latency_ms"], row["text"]))
        if row["raw"] != row["text"]:
            print("   raw (before overlap de-dup): %s" % row["raw"])
        seq, pos = seq + 1, pos + chunk
    print("transcription wall time %.1f s" % (time.time() - t0))
    meta = core.stop_meeting(data, settings, m["id"], background=False)
    print("finalize:", meta["status"], meta.get("error", ""))
    got = core.get_meeting(data, m["id"], include_transcript=False)
    print("\n" + got["summary_md"])
    res = core.search(data, settings, "who owns re-enabling pipeline promotion")
    print("search top hit:", json.dumps(res["hits"][0], ensure_ascii=False)[:300] if res["hits"] else None, res["notes"])
    ans = core.ask(data, settings, "What did we decide about reward functions, and who drafts the schema?")
    print("ask:", ans["answer"])
    print("data dir:", data)
    if not a.keep:
        import shutil
        shutil.rmtree(data)
    return 0 if meta["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
