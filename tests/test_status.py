import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from status import apply_event, empty_snapshot, write_snapshot  # noqa: E402


class ApplyEventTests(unittest.TestCase):
    def test_status_idle_clears_progress(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 4, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
        )
        idle = apply_event(
            snap,
            "status",
            {"status": {"exec_info": {"queue_remaining": 0}}},
            now=2,
        )
        self.assertEqual(idle["state"], "idle")
        self.assertEqual(idle["value"], 0)
        self.assertEqual(idle["max"], 0)
        self.assertIsNone(idle["prompt_id"])
        self.assertIsNone(idle["node"])
        self.assertEqual(idle["queue_remaining"], 0)

    def test_status_then_progress_then_idle(self):
        snap = apply_event(
            empty_snapshot(),
            "status",
            {"status": {"exec_info": {"queue_remaining": 1}}},
            now=1,
        )
        self.assertEqual(snap["state"], "running")
        self.assertEqual(snap["last_event"], "status")

        snap = apply_event(
            snap,
            "progress",
            {"value": 3, "max": 20, "prompt_id": "abc", "node": "9"},
            now=2,
        )
        self.assertEqual(snap["state"], "running")
        self.assertEqual(snap["value"], 3)
        self.assertEqual(snap["max"], 20)
        self.assertEqual(snap["last_event"], "progress")

        snap = apply_event(
            snap,
            "status",
            {"status": {"exec_info": {"queue_remaining": 0}}},
            now=3,
        )
        self.assertEqual(snap["state"], "idle")
        self.assertEqual(snap["value"], 0)
        self.assertEqual(snap["max"], 0)

    def test_mid_run_queue_status_keeps_progress(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 8, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
        )
        snap = apply_event(
            snap,
            "status",
            {"status": {"exec_info": {"queue_remaining": 2}}},
            now=2,
        )
        self.assertEqual(snap["state"], "running")
        self.assertEqual(snap["last_event"], "progress")
        self.assertEqual(snap["value"], 8)
        self.assertEqual(snap["max"], 20)
        self.assertEqual(snap["queue_remaining"], 2)

    def test_executing_new_node_clears_sampler(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 20, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
        )
        snap = apply_event(
            snap,
            "executing",
            {"node": "8", "prompt_id": "p"},
            now=2,
        )
        self.assertEqual(snap["state"], "running")
        self.assertEqual(snap["last_event"], "executing")
        self.assertEqual(snap["value"], 0)
        self.assertEqual(snap["max"], 0)
        self.assertEqual(snap["node"], "8")

    def test_executing_none_does_not_force_idle(self):
        snap = apply_event(
            empty_snapshot(),
            "status",
            {"status": {"exec_info": {"queue_remaining": 2}}},
            now=1,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 5, "max": 10, "prompt_id": "p", "node": "3"},
            now=2,
        )
        snap = apply_event(
            snap,
            "executing",
            {"node": None, "prompt_id": "p"},
            now=3,
        )
        self.assertEqual(snap["state"], "running")
        self.assertEqual(snap["queue_remaining"], 2)
        self.assertEqual(snap["value"], 5)
        self.assertEqual(snap["max"], 10)

    def test_unknown_event_ignored(self):
        start = empty_snapshot()
        self.assertIsNone(apply_event(start, "execution_start", {"prompt_id": "p"}))
        self.assertIsNone(apply_event(start, 1, b"preview"))

    def test_does_not_mutate_input(self):
        start = empty_snapshot()
        apply_event(start, "progress", {"value": 1, "max": 2}, now=1)
        self.assertEqual(start["state"], "idle")
        self.assertEqual(start["value"], 0)


class WriteSnapshotTests(unittest.TestCase):
    def test_atomic_write_is_valid_json(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 1, "max": 4, "prompt_id": "p", "node": "1"},
            now=9,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comfyui-status.json"
            write_snapshot(path, snap)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["state"], "running")
            self.assertEqual(loaded["value"], 1)
            self.assertEqual(loaded["max"], 4)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
