"""Tests for the persistent, incremental team-state projection.

Contract under test (guard hot path, ``checkpoint_team``):

* first build is a FULL fold (projection established from the whole file);
* later builds fold only complete lines past a byte cursor, and the folded
  state is byte-for-byte equivalent to ``extract_team_state(load_messages(f))``
  — the full replay — at EVERY batch boundary;
* tool_use/tool_result pairing (TaskCreate id re-key, Agent spawn ack,
  Task/notification verdict ordering) stays consistent across batches and
  after restart, because the pairing tables are persisted with the cursor;
* os.replace (prune) / truncation / same-size rewrite discard the projection
  and rebuild from scratch;
* headline scenario: a TaskCreate (and teammate spawn) separated from its
  TaskUpdate (and terminal notifications) by MORE than 5000 records survives
  cache eviction and a real process restart, matching a complete replay.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# More than MAX_CACHED_MESSAGES (5000): the gap the bug hinged on.
GAP_FILLERS = 5005


# ─── message builders ──────────────────────────────────────────────────────

def _tool_use(uid, name, inp, text=None):
    blocks = []
    if text is not None:
        blocks.append({"type": "text", "text": text})
    blocks.append({"type": "tool_use", "id": uid, "name": name, "input": inp})
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}}


def _tool_result(uid, text):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": uid, "content": text}
    ]}}


def _filler(i):
    """Ordinary, non-team message (both content shapes appear in the wild)."""
    return {"type": "user", "message": {"role": "user",
            "content": f"filler conversation line number {i}"}}


def _notif(task_id, status, summary="", result=""):
    # Field order mirrors the real harness (tool-use-id sits between task-id
    # and status); the extractor parses fields independently on purpose.
    return {"type": "queue-operation", "content": (
        f"<task-notification>"
        f"<task-id>{task_id}</task-id>"
        f"<tool-use-id>x-{task_id}</tool-use-id>"
        f"<status>{status}</status>"
        f"<summary>{summary}</summary>"
        f"<result>{result}</result>"
        f"</task-notification>"
    )}


def _idle_notif(agent_id, team_name="fleet"):
    return {"type": "user", "teamName": team_name, "message": {"role": "user",
            "content": (f'<teammate-message teammate_id="{agent_id}">'
                        '{"type": "idle_notification"}'
                        '</teammate-message>')}}


def _agent_spawn(uid, bare, team, name=None, role="worker"):
    return [
        _tool_use(uid, "Agent", {"name": name or bare, "role": role}),
        _tool_result(uid, f"Spawned successfully.\nagent_id: {bare}@{team}\n"
                          f"team_name: {team}\n"),
    ]


def _task_spawn(uid, description, bg=True):
    return _tool_use(uid, "Task", {
        "description": description,
        "subagent_type": "general_purpose_task",
        "prompt": "do work",
        "run_in_background": bg,
    })


def _task_create(uid, real_id, subject, owner="pat", description="details"):
    return [
        _tool_use(uid, "TaskCreate",
                  {"subject": subject, "owner": owner, "description": description}),
        _tool_result(uid, f"Task #{real_id} created successfully: {subject}"),
    ]


def _task_update(uid, real_id, status, owner=None):
    inp = {"taskId": str(real_id), "status": status}
    if owner:
        inp["owner"] = owner
    return _tool_use(uid, "TaskUpdate", inp)


def _line(obj):
    return json.dumps(obj, separators=(",", ":")) + "\n"


def _canon(state):
    """Comparable projection output (entity identity + full member/task state)."""
    return {
        "team_name": state.team_name,
        "message_count": state.message_count,
        "last_coordination_index": state.last_coordination_index,
        "lead_summary": state.lead_summary,
        "tasks": sorted(
            (t.task_id, t.subject, t.status, t.owner, t.description, t.from_create)
            for t in state.tasks
        ),
        "teammates": sorted(
            (x.agent_id, x.name, x.role, x.status, x.model, x.cwd)
            for x in state.teammates
        ),
        "subagents": sorted(
            (s.agent_id, s.description, s.subagent_type, s.status, s.result_summary)
            for s in state.subagents
        ),
    }


def _full_replay(path):
    from cozempic.session import load_messages
    from cozempic.team import extract_team_state
    with patch("cozempic.team.load_team_configs", return_value=[]):
        return _canon(extract_team_state(load_messages(path)))


# ─── accumulator: batched fold == one-shot replay at every split ───────────

class TestBatchFoldEquivalence(unittest.TestCase):
    def _corpus(self):
        msgs = []
        msgs += _task_create("c1", 7, "Ship the widget")                       # 0,1
        msgs += _agent_spawn("a1", "alice", "fleet")                           # 2,3
        msgs.append(_task_spawn("t1", "background job A"))                     # 4
        msgs.append(_filler(0))                                                # 5
        msgs.append(_task_update("u1", 7, "in_progress"))                      # 6
        msgs.append(_tool_use("s1", "SendMessage", {"to": "alice"},            # 7
                              text="lead note: dispatch"))
        # t1 completes before anything else — pass-2 verdict
        msgs.append(_notif("t1", "completed", summary="background job A",      # 8
                           result="job A done"))
        # alice's completion lands BEFORE a later SendMessage below → chronology
        # guard must keep her running in replay AND in the split fold
        msgs.append(_notif("alice", "completed", summary="alice first pass"))  # 9
        msgs += _agent_spawn("a2", "carol", "fleet")                           # 10,11
        msgs += _agent_spawn("a3", "dave", "fleet")                            # 12,13
        msgs.append(_task_spawn("t2", "background job B"))                     # 14
        msgs.append(_tool_use("x1", "TaskStop", {"task_id": "t2"}))          # 15
        msgs.append(_filler(1))                                                # 16
        # notification-then-stop ordering across a batch boundary: notif wins
        msgs.append(_task_spawn("t3", "background job C"))                     # 17
        msgs.append(_notif("t3", "completed", summary="background job C"))     # 18
        msgs.append(_tool_use("x2", "TaskStop", {"task_id": "t3"}))            # 19
        # carol completes with no later activity → terminal
        msgs.append(_notif("carol", "completed", summary="carol done"))        # 20
        # dave idles (genuine carrier with teamName)
        msgs.append(_idle_notif("dave@fleet"))                                 # 21
        # alice re-activated AFTER her completion notification → stays running
        msgs.append(_tool_use("s2", "SendMessage", {"to": "alice"},            # 22
                              text="lead note: re-engage"))
        msgs.append(_filler(2))                                                # 23
        # frank is disbanded but a pass-2 verdict ("running", raw agent text)
        # outranks the TeamDelete in the replay's pass-1→pass-2 ordering
        msgs += _agent_spawn("a4", "frank", "teamz")                           # 24,25
        msgs.append(_notif("frank", "running", summary="frank respawned"))     # 26
        msgs.append(_tool_use("d1", "TeamDelete", {"team_name": "teamz"}))     # 27
        # final task completion far from its create (gap semantics within fold)
        msgs.append(_filler(3))                                                # 28
        msgs.append(_task_update("u2", 7, "completed", owner="pat"))           # 29
        return [(i, m, len(_line(m).encode("utf-8"))) for i, m in enumerate(msgs)]

    def test_every_single_split_matches_replay(self):
        from cozempic.team import (
            _TeamProjectionAccumulator, extract_team_state,
        )
        msgs = self._corpus()
        expected = _canon(extract_team_state(msgs))
        for k in range(0, len(msgs) + 1):
            acc = _TeamProjectionAccumulator()
            acc.fold(msgs[:k])
            # Mid-point snapshot must equal a replay of just that prefix too —
            # the convergence gates must not distort intermediate states.
            self.assertEqual(_canon(acc.snapshot()),
                             _canon(extract_team_state(msgs[:k])),
                             f"prefix {k} snapshot diverged")
            acc.fold(msgs[k:])
            self.assertEqual(_canon(acc.snapshot()), expected,
                             f"split at {k} diverged from full replay")

    def test_roundtrip_through_sidecar_dict(self):
        from cozempic.team import _TeamProjectionAccumulator, extract_team_state
        msgs = self._corpus()
        expected = _canon(extract_team_state(msgs))
        acc = _TeamProjectionAccumulator()
        acc.fold(msgs[:7])
        restored = _TeamProjectionAccumulator.from_dict(
            json.loads(json.dumps(acc.to_dict())))
        restored.fold(msgs[7:])
        self.assertEqual(_canon(restored.snapshot()), expected)


# ─── file-backed projection: full first, incremental after ────────────────

class TestProjectionFileBuilder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / "session.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        from cozempic.team import build_session_team_state
        with patch("cozempic.team.load_team_configs", return_value=[]):
            return build_session_team_state(self.path)

    def _append(self, objs):
        with self.path.open("a", encoding="utf-8") as f:
            for o in objs:
                f.write(_line(o))

    def test_first_full_then_incremental_matches_replay(self):
        from cozempic.team import _team_projection_path
        self._append(_task_create("c1", 7, "Ship the widget"))
        state = self._build()
        self.assertFalse(state.is_empty())
        sidecar = _team_projection_path(self.path)
        self.assertTrue(sidecar.exists(), "first build persists the projection")

        self._append([_task_update("u1", 7, "in_progress"), _filler(0)])
        self._append([_task_update("u2", 7, "completed", owner="pat")])
        state = self._build()

        task = [t for t in state.tasks if t.subject == "Ship the widget"][0]
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.task_id, "7")
        self.assertEqual(_canon(state), _full_replay(self.path))

    def test_call_result_pairing_straddles_batches(self):
        """TaskCreate call in one cycle, its 'Task #N created' result in the
        next, update in a third — the persisted pairing table closes all three."""
        self._append([_tool_use("c1", "TaskCreate",
                                {"subject": "Straddle task", "owner": "pat"})])
        self._build()
        self._append([_tool_result("c1", "Task #3 created successfully: Straddle task")])
        self._build()
        self._append([_task_update("u1", 3, "completed", owner="pat")])
        state = self._build()
        task = [t for t in state.tasks if t.subject == "Straddle task"][0]
        self.assertEqual(task.task_id, "3")
        self.assertEqual(task.status, "completed")
        self.assertEqual(_canon(state), _full_replay(self.path))

    def test_unchanged_file_does_not_rewrite_sidecar(self):
        from cozempic.team import _team_projection_path
        self._append(_task_create("c1", 7, "Ship the widget"))
        self._build()
        sidecar = _team_projection_path(self.path)
        first = sidecar.read_text(encoding="utf-8")
        st1 = sidecar.stat()
        # Second build: file untouched → cursor fast path, no sidecar churn.
        self._build()
        st2 = sidecar.stat()
        self.assertEqual(st2.st_mtime_ns, st1.st_mtime_ns)
        self.assertEqual(sidecar.read_text(encoding="utf-8"), first)

    def test_os_replace_triggers_full_rebuild(self):
        from cozempic.team import _team_projection_path
        self._append(_task_create("c1", 1, "Old generation"))
        self._build()

        # Simulate a prune: brand-new file os.replace'd over the session.
        new_path = self.tmp / "rewritten.jsonl"
        with new_path.open("w", encoding="utf-8") as f:
            for o in _task_create("c9", 2, "Brand new session"):
                f.write(_line(o))
        os.replace(new_path, self.path)

        state = self._build()
        subjects = [t.subject for t in state.tasks]
        self.assertEqual(subjects, ["Brand new session"])
        self.assertEqual(_canon(state), _full_replay(self.path))
        # Rebuilt sidecar is fully consumed (no dangling cursor) and carries
        # no old-generation state.
        sidecar = json.loads(_team_projection_path(self.path)
                             .read_text(encoding="utf-8"))
        self.assertEqual(sidecar["cursor"]["offset"], self.path.stat().st_size)
        self.assertNotIn("Old generation", json.dumps(sidecar))

    def test_truncation_triggers_full_rebuild(self):
        events = (_task_create("c1", 1, "Keep me")
                  + [_filler(i) for i in range(20)]
                  + [_task_update("u1", 1, "completed", owner="pat")])
        self._append(events)
        self._build()

        raw = self.path.read_bytes()
        cut = raw.index(b"\n", raw.index(b"\n") + 1) + 1  # after 2 physical lines
        with self.path.open("r+b") as f:
            f.truncate(cut)

        state = self._build()
        self.assertEqual(_canon(state), _full_replay(self.path))
        task = [t for t in state.tasks if t.subject == "Keep me"][0]
        self.assertEqual(task.status, "pending", "update past truncation point is gone")

    def test_paired_bookkeeping_pruned_unpaired_retained(self):
        """Pairing tables stay bounded: a consumed call/result pair is pruned,
        but an un-acked background spawn is retained across cycles/restarts."""
        from cozempic.team import _team_projection_path
        self._append(_task_create("paired-uid", 1, "Paired task"))
        self._append([_task_spawn("bg-uid", "still running")])
        self._build()
        side = json.loads(_team_projection_path(self.path)
                          .read_text(encoding="utf-8"))["projection"]
        self.assertNotIn("paired-uid", side["tool_use_id_to_name"])
        self.assertNotIn("paired-uid", side["pending_task_ids"])
        self.assertNotIn("paired-uid", side["taskcreate_uid_to_key"])
        self.assertIn("bg-uid", side["pending_task_ids"])
        self.assertIn("bg-uid", side["tool_use_id_to_subagent"])

        # The unpaired spawn is still un-acked after another cycle.
        self._append([_filler(0)])
        self._build()
        side = json.loads(_team_projection_path(self.path)
                          .read_text(encoding="utf-8"))["projection"]
        self.assertIn("bg-uid", side["pending_task_ids"])

        # Once its result lands, bookkeeping is pruned — the ENTITY remains.
        self._append([_tool_result("bg-uid", "agent_id: deadbeef done")])
        state = self._build()
        side = json.loads(_team_projection_path(self.path)
                          .read_text(encoding="utf-8"))["projection"]
        self.assertNotIn("bg-uid", side["pending_task_ids"])
        self.assertNotIn("bg-uid", side["tool_use_id_to_subagent"])
        self.assertTrue(any("still running" in (s.description or "")
                            for s in state.subagents))

    def test_corrupt_sidecar_falls_back_to_full_build(self):
        from cozempic.team import _team_projection_path
        self._append(_task_create("c1", 1, "Recover me"))
        self._build()
        _team_projection_path(self.path).write_text("{not json", encoding="utf-8")
        state = self._build()
        self.assertEqual([t.subject for t in state.tasks], ["Recover me"])
        self.assertEqual(_canon(state), _full_replay(self.path))

    def test_partial_trailing_line_deferred_then_consumed(self):
        self._append(_task_create("c1", 1, "Partial tail"))
        self._build()
        partial = _line(_task_update("u1", 1, "in_progress")).rstrip("\n")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(partial)  # no terminating newline — writer mid-append
        state = self._build()
        task = [t for t in state.tasks if t.subject == "Partial tail"][0]
        self.assertEqual(task.status, "pending", "partial line must not be folded")
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n")
        state = self._build()
        task = [t for t in state.tasks if t.subject == "Partial tail"][0]
        self.assertEqual(task.status, "in_progress")
        self.assertEqual(_canon(state), _full_replay(self.path))

    def test_checkpoint_team_wires_projection(self):
        """The guard entry point retains full state past the eviction window."""
        from cozempic.guard import checkpoint_team
        from cozempic.session import (
            MAX_CACHED_MESSAGES, load_messages_incremental,
        )
        from cozempic.team import extract_team_state
        # Early coordination: one task + one teammate, then >5000 plain records.
        early = (_task_create("c1", 7, "Ship the widget")
                 + _agent_spawn("a1", "alice", "fleet"))
        self._append(early)
        checkpoint_team(session_path=self.path, quiet=True)

        chunk = [_filler(i) for i in range(GAP_FILLERS)]
        for i in range(0, len(chunk), 1000):
            self._append(chunk[i:i + 1000])
            checkpoint_team(session_path=self.path, quiet=True)

        # The ordinary message cache HAS evicted the early records.
        window = load_messages_incremental(self.path)
        self.assertEqual(len(window), MAX_CACHED_MESSAGES)
        with patch("cozempic.team.load_team_configs", return_value=[]):
            windowed = extract_team_state(window)
        self.assertEqual(
            [t.subject for t in windowed.tasks], [],
            "baseline: extraction from the evicted window lost the old task",
        )

        # The projection-based checkpoint still has the complete early state.
        state = checkpoint_team(session_path=self.path, quiet=True)
        self.assertEqual([t.subject for t in state.tasks], ["Ship the widget"])
        self.assertIn("alice@fleet", {t.agent_id for t in state.teammates})
        cp = (self.path.parent / "team-checkpoint.md").read_text(encoding="utf-8")
        self.assertIn("Ship the widget", cp)


# ─── headline: >5000-record gap + real process restart ─────────────────────

# Runs in a FRESH interpreter, so every in-memory cache/accumulator is gone;
# only the sidecar can carry state across the restart.
_RESTART_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, os.environ["COZEMPIC_SRC"])
from pathlib import Path
from unittest.mock import patch

from cozempic import team as team_mod
from cozempic.session import load_messages

tmp = Path(os.environ["COZEMPIC_TMP"])
path = tmp / "session.jsonl"


def line(o):
    return json.dumps(o, separators=(",", ":")) + "\n"


def canon(state):
    return {
        "team_name": state.team_name,
        "message_count": state.message_count,
        "last_coordination_index": state.last_coordination_index,
        "lead_summary": state.lead_summary,
        "tasks": sorted((t.task_id, t.subject, t.status, t.owner, t.description,
                         t.from_create) for t in state.tasks),
        "teammates": sorted((x.agent_id, x.name, x.role, x.status, x.model, x.cwd)
                            for x in state.teammates),
        "subagents": sorted((s.agent_id, s.description, s.subagent_type,
                             s.status, s.result_summary) for s in state.subagents),
    }


def notif(tid, status, summary="", result=""):
    return {"type": "queue-operation", "content": (
        f"<task-notification><task-id>{tid}</task-id>"
        f"<tool-use-id>x-{tid}</tool-use-id><status>{status}</status>"
        f"<summary>{summary}</summary><result>{result}</result></task-notification>")}


def idle(agent_id):
    return {"type": "user", "teamName": "fleet", "message": {"role": "user",
            "content": ('<teammate-message teammate_id="%s">{"type": "idle_notification"}'
                        '</teammate-message>') % agent_id}}


def tupdate(uid, rid, status, owner="pat"):
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": uid, "name": "TaskUpdate",
         "input": {"taskId": str(rid), "status": status, "owner": owner}}]}}


late = [
    tupdate("u-late", 7, "completed"),
    notif("t1", "completed", summary="background job A", result="job A done"),
    notif("alice", "completed", summary="alice finished"),
    idle("dave@fleet"),
]
with path.open("a", encoding="utf-8") as f:
    for o in late:
        f.write(line(o))

with patch.object(team_mod, "load_team_configs", lambda: []):
    proj = canon(team_mod.build_session_team_state(path))
    full = canon(team_mod.extract_team_state(load_messages(path)))

print(json.dumps({"proj": proj, "full": full}, sort_keys=True))
"""


class TestEvictionAndRestart(unittest.TestCase):
    def test_state_after_restart_matches_full_replay(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = tmp / "session.jsonl"

            early = (_task_create("c1", 7, "Ship the widget", owner="pat")
                     + _agent_spawn("a1", "alice", "fleet")
                     + _agent_spawn("a3", "dave", "fleet")
                     + [_task_spawn("t1", "background job A")])

            # Simulate guard cycles: seed early events, then stream >5000
            # fillers through the projection in append batches.
            with path.open("w", encoding="utf-8") as f:
                for o in early:
                    f.write(_line(o))
            env = {**os.environ,
                   "COZEMPIC_SRC": str(SRC_DIR),
                   "COZEMPIC_TMP": td}

            def build_here():
                code = (
                    "import os, sys; sys.path.insert(0, os.environ['COZEMPIC_SRC']);"
                    "from cozempic.team import build_session_team_state;"
                    "from pathlib import Path;"
                    "build_session_team_state(Path(os.environ['COZEMPIC_TMP'])/'session.jsonl')"
                )
                subprocess.run([sys.executable, "-c", code], env=env, check=True,
                               capture_output=True, text=True)

            build_here()
            fillers = [_filler(i) for i in range(GAP_FILLERS)]
            for i in range(0, len(fillers), 800):
                with path.open("a", encoding="utf-8") as f:
                    for o in fillers[i:i + 800]:
                        f.write(_line(o))
                build_here()

            create_line = 0
            last_filler_line = len(early) + GAP_FILLERS - 1
            self.assertGreater(last_filler_line - create_line, 5000,
                               "test premise: create/update gap exceeds 5000")

            # Fresh interpreter: only the sidecar survives the restart.
            proc = subprocess.run(
                [sys.executable, "-c", _RESTART_SCRIPT],
                env=env, check=True, capture_output=True, text=True,
            )
            data = json.loads(proc.stdout.strip().splitlines()[-1])

            self.assertEqual(data["proj"], data["full"],
                             "post-restart projection diverges from full replay")

            proj = data["proj"]
            widget = [t for t in proj["tasks"] if t[1] == "Ship the widget"]
            self.assertEqual(len(widget), 1)
            self.assertEqual(widget[0][0], "7")        # real system id retained
            self.assertEqual(widget[0][2], "completed")
            self.assertEqual(widget[0][3], "pat")      # owner retained
            self.assertEqual(widget[0][4], "details")  # description retained

            by_id = {t[0]: t for t in proj["teammates"]}
            self.assertEqual(by_id["alice@fleet"][3], "completed")
            self.assertEqual(by_id["dave@fleet"][3], "idle")
            sub = {s[0]: s for s in proj["subagents"]}
            self.assertEqual(sub["t1"][3], "completed")


if __name__ == "__main__":
    unittest.main()
