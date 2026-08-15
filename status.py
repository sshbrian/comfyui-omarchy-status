import json
import os
import time
from pathlib import Path

SCHEMA = 1


def default_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "omarchy" / "comfyui-status.json"
    return Path.home() / ".local" / "state" / "omarchy" / "comfyui-status.json"


def empty_snapshot() -> dict:
    return {
        "schema": SCHEMA,
        "state": "idle",
        "value": 0,
        "max": 0,
        "queue_remaining": 0,
        "prompt_id": None,
        "node": None,
        "updated_at": 0.0,
        "last_event": None,
    }


def _queue_remaining(data) -> int:
    payload = data if isinstance(data, dict) else {}
    status = payload.get("status", payload)
    if not isinstance(status, dict):
        return 0
    exec_info = status.get("exec_info", {})
    if not isinstance(exec_info, dict):
        return 0
    try:
        return max(0, int(exec_info.get("queue_remaining") or 0))
    except (TypeError, ValueError):
        return 0


def _number(value, fallback=0):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    if n != n:  # NaN
        return fallback
    return n


def apply_event(snap, event, data, now=None):
    """Return an updated snapshot, or None if the event is ignored."""
    if event not in ("status", "progress", "executing"):
        return None

    out = empty_snapshot()
    if isinstance(snap, dict):
        out.update(snap)
    out["schema"] = SCHEMA
    if now is None:
        now = time.time()
    out["updated_at"] = now
    payload = data if isinstance(data, dict) else {}

    if event == "status":
        queue = _queue_remaining(payload)
        out["queue_remaining"] = queue
        if queue == 0:
            out["state"] = "idle"
            out["value"] = 0
            out["max"] = 0
            out["prompt_id"] = None
            out["node"] = None
            out["last_event"] = "status"
        else:
            out["state"] = "running"
            # Keep last_event=progress so a mid-run queue change does not
            # hide the sampler bar.
            if out.get("last_event") != "progress":
                out["last_event"] = "status"
        return out

    if event == "progress":
        out["state"] = "running"
        out["value"] = _number(payload.get("value"), 0)
        out["max"] = _number(payload.get("max"), 0)
        out["prompt_id"] = payload.get("prompt_id")
        out["node"] = payload.get("node")
        out["last_event"] = "progress"
        return out

    # executing
    node = payload.get("node")
    if payload.get("prompt_id") is not None:
        out["prompt_id"] = payload.get("prompt_id")
    if node is None:
        # Prompt finished this node list; idle comes from status.
        out["last_event"] = "executing"
        return out

    if node != out.get("node"):
        out["value"] = 0
        out["max"] = 0
        out["node"] = node
    out["state"] = "running"
    out["last_event"] = "executing"
    return out


def write_snapshot(path, snap) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    body = json.dumps(snap, separators=(",", ":"), ensure_ascii=True)
    tmp.write_text(body + "\n", encoding="utf-8")
    os.replace(tmp, dest)
