import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from status import (  # noqa: E402
    apply_event,
    derive_phase,
    empty_snapshot,
    extract_facts,
    write_snapshot,
    write_session,
    load_session,
)


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
        self.assertIsNone(apply_event(start, "progress_state", {"prompt_id": "p"}))
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


SAMPLE_PROMPT = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "models/sd_xl_base_1.0.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["4", 0],
            "latent_image": ["5", 0],
        },
    },
}

FLUX_PROMPT = {
    "12": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "fp8_e4m3fn"},
    },
    "13": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 768, "height": 1344, "batch_size": 1},
    },
    "25": {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": 99},
    },
    "17": {
        "class_type": "KSampler",
        "inputs": {"seed": 7, "steps": 8, "cfg": 1, "sampler_name": "euler"},
    },
}


class SchemaTwoTests(unittest.TestCase):
    def test_progress_records_step_times(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 1, "max": 20, "prompt_id": "p", "node": "3"},
            now=10,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 2, "max": 20, "prompt_id": "p", "node": "3"},
            now=13,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 3, "max": 20, "prompt_id": "p", "node": "3"},
            now=15.5,
        )
        self.assertEqual(snap["step_times"], [3.0, 2.5])
        self.assertEqual(snap["phase"], "sampling")

    def test_new_sampler_resets_step_times(self):
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 2, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 3, "max": 20, "prompt_id": "p", "node": "3"},
            now=4,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 1, "max": 30, "prompt_id": "p", "node": "9"},
            now=5,
        )
        self.assertEqual(snap["step_times"], [])
        self.assertEqual(snap["node"], "9")

    def test_execution_success_counts_session_once(self):
        snap = apply_event(
            empty_snapshot(now=100),
            "execution_start",
            {"prompt_id": "p"},
            now=100,
        )
        snap = apply_event(
            snap,
            "progress",
            {"value": 4, "max": 20, "prompt_id": "p", "node": "3"},
            now=110,
        )
        snap = apply_event(
            snap,
            "execution_success",
            {"prompt_id": "p"},
            now=148,
        )
        self.assertEqual(snap["session"]["gens"], 1)
        self.assertEqual(snap["session"]["failures"], 0)
        self.assertAlmostEqual(snap["session"]["gpu_sec"], 48)
        self.assertEqual(snap["last_job"]["status"], "ok")
        self.assertAlmostEqual(snap["last_job"]["duration_sec"], 48)

        idle = apply_event(
            snap,
            "status",
            {"status": {"exec_info": {"queue_remaining": 0}}},
            now=149,
        )
        self.assertEqual(idle["session"]["gens"], 1)
        self.assertEqual(idle["last_job"]["status"], "ok")
        self.assertEqual(idle["state"], "idle")

    def test_status_idle_without_success_still_counts(self):
        snap = apply_event(
            empty_snapshot(now=1),
            "progress",
            {"value": 8, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
        )
        idle = apply_event(
            snap,
            "status",
            {"status": {"exec_info": {"queue_remaining": 0}}},
            now=21,
        )
        self.assertEqual(idle["session"]["gens"], 1)
        self.assertAlmostEqual(idle["last_job"]["duration_sec"], 20)
        self.assertIsNone(idle["job_started_at"])

    def test_execution_error_and_interrupt(self):
        snap = apply_event(
            empty_snapshot(now=1),
            "execution_start",
            {"prompt_id": "a"},
            now=1,
        )
        err = apply_event(
            snap,
            "execution_error",
            {
                "prompt_id": "a",
                "node_id": "3",
                "node_type": "KSampler",
                "exception_message": "CUDA out of memory",
            },
            now=5,
        )
        self.assertEqual(err["session"]["failures"], 1)
        self.assertEqual(err["session"]["gens"], 0)
        self.assertEqual(err["last_job"]["status"], "error")
        self.assertEqual(err["last_job"]["error"], "CUDA out of memory")
        self.assertEqual(err["phase"], "error")

        nxt = apply_event(err, "execution_start", {"prompt_id": "b"}, now=10)
        stopped = apply_event(
            nxt,
            "execution_interrupted",
            {"prompt_id": "b", "node_id": "4", "node_type": "VAEDecode"},
            now=12,
        )
        self.assertEqual(stopped["session"]["interrupts"], 1)
        self.assertEqual(stopped["last_job"]["status"], "interrupted")

    def test_extract_facts_sdxl_and_flux(self):
        sdxl = extract_facts(SAMPLE_PROMPT)
        self.assertEqual(sdxl["checkpoint"], "sd_xl_base_1.0.safetensors")
        self.assertEqual(sdxl["width"], 1024)
        self.assertEqual(sdxl["height"], 1024)
        self.assertEqual(sdxl["seed"], 42)
        self.assertEqual(sdxl["steps"], 20)
        self.assertEqual(sdxl["sampler"], "euler")
        self.assertEqual(sdxl["batch"], 1)

        flux = extract_facts(FLUX_PROMPT)
        self.assertEqual(flux["checkpoint"], "flux1-dev.safetensors")
        self.assertEqual(flux["width"], 768)
        self.assertEqual(flux["height"], 1344)
        self.assertEqual(flux["seed"], 7)

    def test_enrich_attaches_facts_and_node_type(self):
        live = {
            "running": [(0, "p", SAMPLE_PROMPT, {
                "extra_pnginfo": {"workflow": {"nodes": [{"id": 3, "title": "Base sampler"}]}}
            }, [])],
            "pending": [(1, "q", {}, {}, [])],
            "vram": {"name": "cuda", "used": 10, "total": 24},
        }
        snap = apply_event(
            empty_snapshot(),
            "progress",
            {"value": 2, "max": 20, "prompt_id": "p", "node": "3"},
            now=1,
            live=live,
        )
        self.assertEqual(snap["facts"]["checkpoint"], "sd_xl_base_1.0.safetensors")
        self.assertEqual(snap["facts"]["seed"], 42)
        self.assertEqual(snap["node_type"], "KSampler")
        self.assertEqual(snap["node_title"], "Base sampler")
        self.assertEqual(snap["queue_running"], 1)
        self.assertEqual(snap["queue_pending"], 1)
        self.assertEqual(snap["vram"]["total"], 24)
        self.assertEqual(snap["phase"], "sampling")

    def test_phase_from_node_type(self):
        snap = apply_event(
            empty_snapshot(),
            "executing",
            {"node": "8", "prompt_id": "p"},
            now=1,
        )
        snap["node_type"] = "VAEDecode"
        self.assertEqual(derive_phase(snap), "decoding")
        snap["node_type"] = "CheckpointLoaderSimple"
        self.assertEqual(derive_phase(snap), "loading")
        snap["node_type"] = "SaveImage"
        self.assertEqual(derive_phase(snap), "saving")
        snap["node_type"] = "CLIPTextEncode"
        self.assertEqual(derive_phase(snap), "working")
        snap["node_type"] = "CLIPLoader"
        self.assertEqual(derive_phase(snap), "loading")
        snap["node_type"] = "DualCLIPLoader"
        self.assertEqual(derive_phase(snap), "loading")

    def test_session_file_resets_on_new_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comfyui-session.json"
            write_session(path, {
                "day": "1999-01-01",
                "gens": 9,
                "failures": 1,
                "interrupts": 0,
                "gpu_sec": 12,
            })
            loaded = load_session(path, now=1_700_000_000)
            self.assertEqual(loaded["gens"], 0)
            self.assertNotEqual(loaded["day"], "1999-01-01")


if __name__ == "__main__":
    unittest.main()
