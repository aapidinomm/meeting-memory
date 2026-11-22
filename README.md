# Meeting Memory

A [Kiro Crew](https://github.com/kirodotdev/KiroCrew) app that records meetings from your browser microphone,
transcribes them with **Amazon Bedrock in your own AWS account**, and keeps searchable summaries, decisions and
action items on your machine.

- **Record:** Start, Pause and Stop on the app page. The browser records your mic, and optionally a shared tab's
  audio (Chime, Zoom or Meet in a browser tab) so remote voices are captured even when you wear headphones.
- **Live transcript:** Bedrock speech-to-text, Mistral Voxtral Small by default. Audio is sent as 20-second
  16 kHz WAV chunks with a 1.5-second overlap, and repeated words at chunk boundaries are removed.
  Silent chunks are skipped, so dead air costs nothing.
- **Vocabulary:** a list of names, acronyms and product terms that is sent with every chunk as a spelling hint.
- **Summary on stop:** a Bedrock LLM (Claude Sonnet 4.6 by default) writes a title, a short summary, key points,
  decisions, action items with owners and due dates, open questions and topic tags. You can edit or regenerate it.
- **Search and ask:** keyword (SQLite FTS5) plus semantic search (Titan Text Embeddings v2) across every meeting,
  and an Ask box that answers with citations to the meeting and timestamp.
- **Agent access:** MCP tools (`mm_search`, `mm_ask`, `mm_get_meeting`, ...) let your Kiro agent search and read
  your meetings from chat.
- **Local storage:** everything lives under the app's data directory. The only network calls go to Bedrock
  (and STS for the connection test) with the AWS profile you choose.

## Requirements

- Kiro Crew with the gateway running (Linux or macOS).
- An AWS profile with access to these Bedrock models in your region (defaults shown; all are configurable):

  | Purpose | Default model id |
  |---|---|
  | Speech-to-text | `mistral.voxtral-small-24b-2507` (or `mistral.voxtral-mini-3b-2507`, faster and cheaper) |
  | Summary and answers | `us.anthropic.claude-sonnet-4-6` (or `us.anthropic.claude-haiku-4-5-20251001-v1:0`) |
  | Embeddings | `amazon.titan-embed-text-v2:0` (512 dimensions) |

  Minimum IAM permissions: `bedrock:InvokeModel` on those models and inference profiles, and
  `sts:GetCallerIdentity` for the connection test. Anthropic models can require a one-time model-access request
  in the Bedrock console.
- `boto3` available to the Kiro Crew gateway's Python (it ships with Kiro Crew) and, for semantic search and
  Ask from chat, to the `python3` that starts the MCP server. Without boto3 there, the MCP tools still list,
  read and keyword-search meetings.
- A browser that allows microphone access to the dashboard. Browsers allow it only on `https://` or
  `http://localhost` pages. If Kiro Crew runs on a remote host (a cloud desktop, for example), open the dashboard
  through an SSH tunnel from your laptop: `ssh -NL 5476:localhost:5476 <host>`, then browse to
  `http://localhost:5476`.

## Install

Run these in **your own terminal** on the machine where the Kiro Crew gateway runs. Kiro Crew refuses app
installs started from an agent's sandboxed shell.

```bash
# 1. Get the code (any directory works; it becomes the app's recorded source for updates)
git clone <this-repo-url> ~/meeting-memory

# 2. Install and enable the app. The dashboard asks you once to trust this third-party app.
kirocrew app install ~/meeting-memory
kirocrew app enable meeting-memory

# 3. Choose the AWS profile and region for Bedrock (only the profile NAME is stored).
python3 ~/.kiro/crew/apps/meeting-memory/scripts/configure.py \
    --aws-profile <your-bedrock-profile> --aws-region us-west-2 --test
```

Step 3 can also be done in the browser: until a profile is set, the app page opens on a **Set up Bedrock access**
screen. Run `configure.py` with no arguments to be prompted interactively; it lists the profiles it finds.

The profile has to work for the **gateway process**, not only for your shell. Check it with
`aws sts get-caller-identity --profile <name>` in the environment that starts the gateway. Profiles that use
`credential_process`, SSO or static keys in `~/.aws/` all work. If you change `~/.aws/config` later, nothing
needs restarting, because the app re-creates its Bedrock client when credentials expire.

Open **Meeting Memory** from the sidebar (route `/apps/meeting-memory`) and press **Start recording**.

## Change settings later

Use any of these:

- **App page → Settings**: profile, region, models, chunk length, overlap, vocabulary, language hint. **Save and
  test** checks the profile and makes one tiny call to each model.
- **CLI**: `python3 ~/.kiro/crew/apps/meeting-memory/scripts/configure.py --aws-profile other-profile --test`.
  Use `--show` to print the current settings.
- **Chat**: ask your agent, for example "switch meeting-memory to the `team-bedrock` profile and test it". It calls
  `mm_configure`.

Settings live in `~/.kiro/crew/apps/meeting-memory/data/settings.json`. Changing the embedding model clears the
stored vectors; they are rebuilt automatically on the next search.

## Update and uninstall

```bash
cd ~/meeting-memory && git pull
kirocrew app update meeting-memory --source ~/meeting-memory   # keeps data/

kirocrew app uninstall meeting-memory                 # keeps your meetings in data/
kirocrew app uninstall meeting-memory --purge-data    # also deletes recordings, transcripts and settings
```

## Data layout

```
~/.kiro/crew/apps/meeting-memory/data/
├── settings.json
├── index.sqlite                  # chunk text, FTS5 index, embeddings
└── meetings/<id>/
    ├── meta.json                 # title, status, timestamps
    ├── transcript.jsonl          # one transcribed chunk per line (text, raw, timing, model, latency)
    ├── audio/00000.wav ...       # the raw 16 kHz mono chunks
    ├── summary.md                # rendered (or hand-edited) summary
    └── summary.json              # structured summary
```

## How it works

```
Browser (dashboard page)                        Kiro Crew gateway (in-process app routes)      Amazon Bedrock
mic [+ tab audio] → AudioContext → 16 kHz PCM
 → 20 s WAV chunks (+1.5 s overlap) ── POST /api/apps/meeting-memory/meetings/<id>/chunks ──→ Converse (Voxtral)
                                        de-dup overlap, append transcript.jsonl
 Stop ──────────────────────────────── POST …/stop → background: summary ─────────────────→ Converse (Claude)
                                                      index windows ───────────────────────→ InvokeModel (Titan)
 Search / Ask ──────────────────────── POST …/search, …/ask (FTS5 + cosine, RRF fusion) ───→ embeddings / Claude
Kiro agent ── stdio MCP (mcp/server.py) ── same backend/core.py and data dir
```

The app opens no network listener of its own. Its HTTP routes are served by the gateway behind the dashboard's
authentication, and every route is restricted to the dashboard owner.

## Limits and notes

- Text appears about one chunk (20 s) behind the speaker. Voxtral transcribes whole clips and does not stream.
- Voxtral does not label speakers.
- Capturing remote voices requires the meeting to run in a browser tab you share, with "Share tab audio" ticked.
  Whether Chrome can also capture a separate desktop app's audio depends on your OS and browser version.
- Keep the app page open while recording. Leaving the page stops the recording and summarizes what was captured.
- `mistral.voxtral-small-24b-2507` is listed by Bedrock as a Legacy model with end of life 2027-03-30. Switch
  `stt_model` in Settings when a successor is available.
- **Get every participant's consent before recording.** Transcripts can contain confidential information: they
  stay on this machine and in your AWS account.

## Development

```bash
python3 -m unittest discover -s tests -v            # unit tests, fake Bedrock, no AWS calls
python3 scripts/live_smoke.py --aws-profile <name>  # real Bedrock end-to-end (Polly-synthesized meeting, a few cents)
node --check ui/index.mjs                            # UI syntax check (no build step)
kirocrew app update meeting-memory --source .        # push local changes into the installed copy
```

See [AGENTS.md](AGENTS.md) for the codebase map and the rules agents follow when changing it.

## Roadmap

- Local models: Whisper through Kiro Crew's local speech-to-text engine, a small local embedding model, and a
  local LLM for summaries.
- Amazon Transcribe streaming as a live, speaker-labelled alternative.
- Calendar-linked meeting titles and attendees.
- Tuning chunk length, overlap and prompts for lower word error rate with KiroCrew Scientist.

## License

[MIT](LICENSE)
