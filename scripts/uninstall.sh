#!/usr/bin/env bash
# Run by Kiro Crew before it removes the app's files. Meetings are private recordings, so data/ is deleted only when
# the user asked for a purge (PURGE_DATA=1, set by `kirocrew app uninstall --purge-data`).
set -u
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${PURGE_DATA:-0}" = "1" ] && [ -d "$APP_DIR/data" ]; then
  rm -rf -- "$APP_DIR/data"
  echo "meeting-memory: deleted $APP_DIR/data"
else
  echo "meeting-memory: kept $APP_DIR/data (meetings, transcripts, settings)"
fi
