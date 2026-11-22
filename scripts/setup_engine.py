#!/usr/bin/env python3
"""Create the local engine environment: speaker labels and local speech-to-text models.

    python3 scripts/setup_engine.py [--data-dir DIR] [--models base.en small.en]

Makes DIR/engine (a virtual environment with numpy, scipy, scikit-learn and faster-whisper) and, with
--models, downloads those Whisper models once from Hugging Face. Uses uv when it is installed, else
python -m venv and pip. Nothing here touches AWS.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "scipy", "scikit-learn", "faster-whisper"]


def default_data_dir() -> Path:
    env = os.environ.get("KIROCREW_APP_DATA_DIR")
    return Path(env) if env else APP_ROOT / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--models", nargs="*", default=["base.en"],
                    help="Whisper models to download now (tiny.en base.en small.en medium.en large-v3-turbo)")
    a = ap.parse_args()
    env_dir = a.data_dir / "engine"
    py = env_dir / "bin" / "python"
    uv = shutil.which("uv")
    if not py.exists():
        if uv:
            subprocess.run([uv, "venv", "-q", "--python", "3.12", str(env_dir)], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    if uv:
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py)] + PACKAGES, check=True)
    else:
        subprocess.run([str(py), "-m", "pip", "install", "-q"] + PACKAGES, check=True)
    for m in a.models:
        print("downloading Whisper model %s ..." % m, flush=True)
        subprocess.run([str(py), "-c", "from faster_whisper import WhisperModel as W; W(%r, device='cpu', "
                        "compute_type='int8')" % m], check=True)
    print("engine ready: %s" % py)
    return 0


if __name__ == "__main__":
    sys.exit(main())
