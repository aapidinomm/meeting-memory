"""Unit tests for backend/core.py with a fake Bedrock client (no AWS calls).

Run: python3 -m unittest discover -s tests -v
"""

import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mm_core", str(ROOT / "backend" / "core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def _embed(text, dims=64):
    vec = [0.0] * dims
    for word in text.lower().split():
        word = "".join(c for c in word if c.isalnum())
        if word:
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


class FakeBedrock:
    """Answers converse/invoke_model/get_caller_identity like Bedrock would, and records the calls."""

    def __init__(self, transcripts=None, summary=None, fail_service=None):
        self.transcripts = list(transcripts or [])
        self.summary = summary
        self.calls = []
        self.fail_service = fail_service

    def converse(self, **kw):
        self.calls.append(("converse", kw))
        content = kw["messages"][0]["content"]
        if any("audio" in c for c in content):
            text = self.transcripts.pop(0) if self.transcripts else ""
        elif kw.get("system", [{}])[0].get("text", "").startswith("You write meeting notes"):
            text = "Here you go:\n" + json.dumps(self.summary or {"title": "Planning sync", "tldr": "We planned.",
                                                                    "decisions": ["Ship Friday"],
                                                                    "action_items": [{"owner": "Monidipa", "task": "ship the app", "due": "Friday"}]})
        else:
            text = "Answer citing [1]."
        return {"output": {"message": {"content": [{"text": text}]}}, "usage": {"inputTokens": 10}}

    def invoke_model(self, **kw):
        self.calls.append(("invoke_model", kw))
        body = json.loads(kw["body"])
        return {"body": io.BytesIO(json.dumps({"embedding": _embed(body["inputText"])}).encode())}

    def get_caller_identity(self):
        return {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/Test/x"}


def silent_wav(seconds=1.0):
    return core.wav_bytes([0] * int(16000 * seconds))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.fake = FakeBedrock()
        core.client_factory = lambda service, settings: self.fake
        self.settings = core.save_settings(self.data, {"aws_profile": "test-profile", "aws_region": "us-west-2"})

    def tearDown(self):
        core.client_factory = None
        self.tmp.cleanup()


class SettingsTests(Base):
    def test_profile_is_not_hardcoded(self):
        fresh = core.load_settings(self.data / "elsewhere")
        self.assertEqual(fresh["aws_profile"], "")
        self.assertFalse(core.configured(fresh))

    def test_round_trip_and_partial_update(self):
        core.save_settings(self.data, {"vocabulary": "Monidipa, Kirin\nKirinLib, Kirin"})
        s = core.load_settings(self.data)
        self.assertEqual(s["aws_profile"], "test-profile")
        self.assertEqual(s["vocabulary"], ["Monidipa", "Kirin", "KirinLib"])

    def test_rejects_bad_values(self):
        for patch in ({"aws_region": "west"}, {"chunk_seconds": 1}, {"embed_dims": 300}, {"bogus": 1},
                      {"aws_profile": "bad profile; rm"}, {"stt_model": "x y"}):
            with self.assertRaises(core.AppError, msg=patch):
                core.save_settings(self.data, patch)

    def test_unconfigured_call_fails_closed(self):
        core.client_factory = None
        with self.assertRaises(core.AppError) as ctx:
            core.transcribe_wav(core.load_settings(self.data / "x"), silent_wav())
        self.assertEqual(ctx.exception.code, "not_configured")


class DedupeTests(unittest.TestCase):
    def test_removes_repeated_boundary_words(self):
        self.assertEqual(core.dedupe_overlap("we will ship on Friday", "on Friday, then review."), "then review.")

    def test_keeps_text_without_overlap(self):
        self.assertEqual(core.dedupe_overlap("we will ship", "Next item is hiring."), "Next item is hiring.")

    def test_short_single_word_is_not_treated_as_overlap(self):
        self.assertEqual(core.dedupe_overlap("this is it", "it works"), "it works")

    def test_long_single_word_overlap_is_removed(self):
        self.assertEqual(core.dedupe_overlap("talk about embeddings", "embeddings next"), "next")


class MeetingFlowTests(Base):
    def test_record_summarize_search_ask(self):
        self.fake.transcripts = ["Hello team, we will ship the meeting app on Friday",
                                 "on Friday. Monidipa owns the release notes."]
        m = core.create_meeting(self.data, "")
        core.add_chunk(self.data, self.settings, m["id"], 0, 0, 20000, silent_wav())
        row = core.add_chunk(self.data, self.settings, m["id"], 1, 18500, 21500, silent_wav())
        self.assertEqual(row["text"], "Monidipa owns the release notes.")
        # The audio prompt carries the previous chunk's tail for context.
        audio_prompt = self.fake.calls[1][1]["messages"][0]["content"][1]["text"]
        self.assertIn("ship the meeting app on Friday", audio_prompt)

        meta = core.stop_meeting(self.data, self.settings, m["id"], background=False)
        self.assertEqual(meta["status"], "done")
        got = core.get_meeting(self.data, m["id"])
        self.assertEqual(got["meta"]["title"], "Planning sync")  # model title used when the user gave none
        self.assertIn("**Monidipa**: ship the app (due Friday)", got["summary_md"])
        self.assertEqual(len(got["transcript"]), 2)

        hits = core.search(self.data, self.settings, "release notes")["hits"]
        self.assertTrue(hits and hits[0]["meeting_id"] == m["id"])
        ans = core.ask(self.data, self.settings, "who owns the release notes?")
        self.assertEqual(ans["answer"], "Answer citing [1].")
        self.assertTrue(ans["sources"])

    def test_user_title_is_kept(self):
        m = core.create_meeting(self.data, "Design review")
        core.stop_meeting(self.data, self.settings, m["id"], background=False)
        self.assertEqual(core.read_meta(self.data, m["id"])["title"], "Design review")

    def test_chunk_after_stop_is_refused(self):
        m = core.create_meeting(self.data, "x")
        core.stop_meeting(self.data, self.settings, m["id"], background=False)
        with self.assertRaises(core.AppError) as ctx:
            core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        self.assertEqual(ctx.exception.status, 409)

    def test_duplicate_seq_is_idempotent(self):
        self.fake.transcripts = ["first", "second"]
        m = core.create_meeting(self.data, "x")
        a = core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        b = core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(len(core.read_transcript(self.data, m["id"])), 1)

    def test_rejects_non_wav_and_bad_ids(self):
        m = core.create_meeting(self.data, "x")
        with self.assertRaises(core.AppError):
            core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, b"not audio" * 10)
        for bad in ("../../etc", "20260101T000000-zzzzzz", ""):
            with self.assertRaises(core.AppError):
                core.meeting_dir(self.data, bad)

    def test_finalize_failure_is_recorded_not_raised(self):
        m = core.create_meeting(self.data, "x")
        self.fake.transcripts = ["some words here"]
        core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        self.fake.summary = None
        broken = FakeBedrock()
        broken.converse = lambda **kw: {"output": {"message": {"content": [{"text": "not json"}]}}}
        core.client_factory = lambda s, st: broken
        meta = core.stop_meeting(self.data, self.settings, m["id"], background=False)
        self.assertEqual(meta["status"], "error")
        self.assertIn("JSON", meta["error"])

    def test_edit_summary_and_delete(self):
        m = core.create_meeting(self.data, "x")
        core.stop_meeting(self.data, self.settings, m["id"], background=False)
        core.update_meeting(self.data, m["id"], {"summary_md": "# x\n\nHand-written notes about budgets.\n"})
        hits = core.search(self.data, self.settings, "budgets")["hits"]
        self.assertTrue(any(h["kind"] == "summary" for h in hits))
        core.delete_meeting(self.data, m["id"])
        self.assertEqual(core.list_meetings(self.data), [])
        self.assertEqual(core.search(self.data, self.settings, "budgets")["hits"], [])

    def test_no_speech_marker_becomes_empty(self):
        self.fake.transcripts = ["(no speech)"]
        m = core.create_meeting(self.data, "x")
        row = core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        self.assertEqual(row["text"], "")

    def test_keyword_search_survives_embedding_failure(self):
        self.fake.transcripts = ["the quarterly roadmap review"]
        m = core.create_meeting(self.data, "x")
        core.add_chunk(self.data, self.settings, m["id"], 0, 0, 1000, silent_wav())
        core.stop_meeting(self.data, self.settings, m["id"], background=False)

        def boom(**kw):
            raise RuntimeError("bedrock down")
        self.fake.invoke_model = boom
        res = core.search(self.data, self.settings, "roadmap")
        self.assertTrue(res["hits"])
        self.assertTrue(res["notes"])


class ConnectionTests(Base):
    def test_check_connection_reports_each_model(self):
        out = core.check_connection(self.settings)
        self.assertEqual(out["account"], "123456789012")
        self.assertEqual(set(out["checks"]), {"llm", "embed", "stt"})
        self.assertTrue(out["ok"])


class McpTests(Base):
    def test_tools_list_and_status(self):
        spec2 = importlib.util.spec_from_file_location("mm_server", str(ROOT / "mcp" / "server.py"))
        server = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(server)
        server.DATA_DIR = self.data
        names = [t["name"] for t in server.handle({"id": 1, "method": "tools/list"})["result"]["tools"]]
        self.assertIn("mm_search", names)
        res = server.handle({"id": 2, "method": "tools/call", "params": {"name": "mm_status", "arguments": {}}})
        self.assertTrue(json.loads(res["result"]["content"][0]["text"])["configured"])
        bad = server.handle({"id": 3, "method": "tools/call", "params": {"name": "mm_get_meeting", "arguments": {"meeting_id": "../x"}}})
        self.assertTrue(bad["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
