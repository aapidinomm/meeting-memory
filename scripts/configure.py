#!/usr/bin/env python3
"""Set meeting-memory's AWS profile, region and models after install (also editable later in the app's Settings).

    python3 scripts/configure.py --aws-profile my-bedrock-profile [--aws-region us-west-2] [--test]
    python3 scripts/configure.py            # interactive: asks for the profile and region

Only a profile NAME is stored (in the app's data/settings.json); credentials stay wherever that profile gets them
(~/.aws/config, credential_process, SSO). Nothing here reads or copies credentials.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]


def _core():
    spec = importlib.util.spec_from_file_location("meeting_memory_core", str(APP_ROOT / "backend" / "core.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def default_data_dir() -> Path:
    home = Path(os.environ.get("KIROCREW_HOME") or Path.home() / ".kiro" / "crew")
    return home / "apps" / "meeting-memory" / "data"


def _profiles() -> list:
    if not shutil.which("aws"):
        return []
    try:
        out = subprocess.run(["aws", "configure", "list-profiles"], capture_output=True, text=True, timeout=20)
        return [p for p in out.stdout.split() if p]
    except (OSError, subprocess.SubprocessError):
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aws-profile", help="named AWS profile with Bedrock access")
    ap.add_argument("--aws-region", help="Bedrock region (default us-west-2)")
    ap.add_argument("--stt-model", help="speech-to-text model id")
    ap.add_argument("--llm-model", help="summary/answer model id or inference profile")
    ap.add_argument("--embed-model", help="embedding model id")
    ap.add_argument("--vocabulary", help="comma-separated names and terms to spell correctly")
    ap.add_argument("--data-dir", type=Path, default=None, help="app data dir (default: %s)" % default_data_dir())
    ap.add_argument("--test", action="store_true", help="check the profile and each model with one tiny call")
    ap.add_argument("--show", action="store_true", help="print the current settings and exit")
    args = ap.parse_args(argv)

    core = _core()
    data_dir = args.data_dir or default_data_dir()
    if args.show:
        print(json.dumps(core.load_settings(data_dir), indent=2))
        return 0

    current = core.load_settings(data_dir)
    patch = {}
    profile = args.aws_profile
    if profile is None and sys.stdin.isatty():
        names = _profiles()
        if names:
            print("AWS profiles on this machine: " + ", ".join(names))
        profile = input("AWS profile for Bedrock [%s]: " % (current["aws_profile"] or "")).strip() or current["aws_profile"]
        region = input("Bedrock region [%s]: " % current["aws_region"]).strip() or current["aws_region"]
        patch["aws_region"] = region
    if profile is None:
        ap.error("--aws-profile is required when not running interactively")
    patch["aws_profile"] = profile
    for key in ("aws_region", "stt_model", "llm_model", "embed_model", "vocabulary"):
        val = getattr(args, key)
        if val is not None:
            patch[key] = val
    try:
        settings = core.save_settings(data_dir, patch)
    except core.AppError as exc:
        print("error: %s" % exc.message, file=sys.stderr)
        return 2
    print("Saved %s" % (data_dir / "settings.json"))
    print(json.dumps({k: settings[k] for k in ("aws_profile", "aws_region", "stt_model", "llm_model", "embed_model")}, indent=2))
    if args.test:
        try:
            result = core.check_connection(settings)
        except core.AppError as exc:
            print("test failed: %s" % exc.message, file=sys.stderr)
            return 3
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
