# AGENTS.md: Meeting Memory

Instructions for coding agents that install, configure, or change this Kiro Crew app.

## Install (agent procedure)

1. **Ask the user which AWS profile and region to use for Bedrock.** Never pick one yourself, never hardcode
   one, and never read, copy or print credential files (`~/.aws/credentials`, SSO caches). Only the profile name
   is stored.
2. Kiro Crew refuses `kirocrew app install` from an agent's sandboxed shell; this is an intentional ownership
   check. Do not try to work around it. Give the user these commands to run in their own terminal:

   ```bash
   kirocrew app install <path-to-this-repo>
   kirocrew app enable meeting-memory
   python3 ~/.kiro/crew/apps/meeting-memory/scripts/configure.py --aws-profile <profile> --aws-region <region> --test
   ```

   If the agent's shell can install (no sandbox), run the same commands yourself, then the configure step.
3. Verify the install:
   - `kirocrew app list` shows `meeting-memory` as `enabled`.
   - The `mm_status` MCP tool returns `"configured": true`.
   - `mm_configure` with `{"test": true}`, or `configure.py --test`, reports `"ok": true`. If a check fails,
     report the exact message. Typical causes: the profile is missing in the gateway's environment, the region
     has no access to that model, or the Anthropic model-access request is still pending.
4. Tell the user to open the dashboard over `https://` or `http://localhost` so the browser allows the microphone
   (for a remote host: `ssh -NL 5476:localhost:5476 <host>`), then sidebar → **Meeting Memory** → **Start recording**.

To change settings later, use `mm_configure`, `scripts/configure.py`, or the app page's Settings tab. All three
write the same `data/settings.json`.

## Codebase map

| Path | What it is |
|---|---|
| `app.json` | Kiro Crew manifest: UI page, in-gateway routes hook, MCP server, skill, permissions |
| `backend/core.py` | All logic: settings validation, Bedrock calls, chunk transcription and overlap de-dup, summary, SQLite index, hybrid search, ask. Standard library plus lazily imported boto3. |
| `backend/routes.py` | aiohttp handlers mounted at `/api/apps/meeting-memory/`. Owner-only, thin wrappers over `core` |
| `mcp/server.py` | stdio MCP server (JSON-RPC, standard library), tools `mm_*`, same `core` and data dir (`KIROCREW_APP_DATA_DIR`) |
| `ui/index.mjs` | The page: recorder (getUserMedia/getDisplayMedia → ScriptProcessor → 16 kHz WAV chunks), meeting list and detail, search and ask, settings. Plain ESM, React via the host's import map, no build step |
| `scripts/configure.py` | Install-time and later configuration (profile, region, models, vocabulary, `--test`, `--show`) |
| `scripts/live_smoke.py` | Real Bedrock end-to-end check with a Polly-synthesized meeting |
| `scripts/uninstall.sh` | Keeps `data/` unless `PURGE_DATA=1` |
| `skills/meeting-memory/SKILL.md` | What the chat agent loads to use the MCP tools |
| `tests/test_core.py` | unittest suite with a fake Bedrock client |

## Rules for changes

- **No hardcoded AWS profile, account or region-specific ARNs.** Defaults in `DEFAULT_SETTINGS` keep
  `aws_profile` empty, so an unconfigured install fails closed with `not_configured`. Keep it that way.
- **No credentials in code, settings, logs or test fixtures.** `boto3.Session(profile_name=...)` resolves them.
  On an expired or invalid token, `core._call` rebuilds the client once. Keep that path when you add calls,
  because botocore clients snapshot credentials when they are created.
- **No network listener.** Routes are served by the gateway. Every handler goes through `_guard`, which requires
  the dashboard owner (`ctx.require_owner`) or this app's own token, and fails closed when the host has no owner
  gate. New routes must use `_guard` too.
- **Validate at the boundary.** Meeting ids must match `_MEETING_ID_RE` before any path is built. Audio must be
  a RIFF/WAVE file of at most `MAX_CHUNK_BYTES`. Settings go through `validate_settings_patch`.
- **Transcripts are untrusted data.** Prompts that include transcript text (summary, ask) say so in the system
  prompt. Keep that wording when you edit prompts.
- **Logic lives in `core.py`.** Routes and MCP tools stay thin, so the dashboard and the agent behave the same.
- **Voxtral on Bedrock accepts `wav` audio**, not ogg, webm or flac (tested 2026-11-22). That is why the browser
  encodes PCM WAV itself instead of using MediaRecorder.
- The UI must not unmount the recorder while recording: it stays mounted across the Meetings/Settings switch.

## Test before you hand off

```bash
python3 -m unittest discover -s tests -v
node --check ui/index.mjs
python3 -m py_compile backend/*.py mcp/server.py scripts/*.py
```

For changes to Bedrock calls, prompts or chunking, also run `scripts/live_smoke.py --aws-profile <profile>`. It
makes real, billed calls, so ask the user first. Then sync the installed copy with
`kirocrew app update meeting-memory --source <repo>`. That command, too, runs only from the user's own terminal
when the agent is sandboxed.
