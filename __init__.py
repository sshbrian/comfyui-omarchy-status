"""Publish ComfyUI queue/progress events to a JSON file for the Omarchy bar."""

import importlib.util
import logging
import threading
import time
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_log = logging.getLogger("comfyui_omarchy_status")
_REFRESH_SEC = 2.0


def _load_status():
    path = Path(__file__).resolve().parent / "status.py"
    spec = importlib.util.spec_from_file_location("omarchy_comfy_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(snap):
    if not isinstance(snap, dict):
        return ()
    return (
        snap.get("state"),
        snap.get("phase"),
        snap.get("value"),
        snap.get("max"),
        snap.get("queue_remaining"),
        snap.get("queue_running"),
        snap.get("queue_pending"),
        snap.get("prompt_id"),
        snap.get("node"),
        snap.get("node_type"),
        snap.get("node_title"),
        snap.get("last_event"),
        tuple(snap.get("step_times") or ()),
        tuple(sorted((snap.get("facts") or {}).items())),
        tuple(sorted((snap.get("vram") or {}).items())),
        tuple(sorted((snap.get("session") or {}).items())),
        None if not snap.get("last_job") else tuple(sorted(
            (k, v) for k, v in snap["last_job"].items() if k != "facts"
        )),
    )


def _patch():
    status = _load_status()
    try:
        from server import PromptServer
    except Exception as exc:
        _log.warning("PromptServer unavailable; status file disabled: %s", exc)
        return

    orig = PromptServer.send_sync
    lock = threading.Lock()
    path = status.default_path()
    session_path = status.default_session_path()
    snap = status.empty_snapshot()
    snap["session"] = status.load_session(session_path)
    holder = {"server": None, "snap": snap}

    def persist(nxt, write_session=False):
        status.write_snapshot(path, nxt)
        if write_session:
            try:
                status.write_session(session_path, nxt.get("session") or status.empty_session())
            except Exception:
                _log.exception("failed to persist Omarchy session totals")

    def send_sync(self, event, data, sid=None):
        holder["server"] = self
        if isinstance(event, str):
            try:
                with lock:
                    current = holder["snap"]
                    nxt = status.apply_event(current, event, data)
                    if nxt is not None:
                        prev_session = current.get("session")
                        holder["snap"] = nxt
                        persist(nxt, write_session=nxt.get("session") != prev_session)
            except Exception:
                _log.exception("failed to update Omarchy status file")
        return orig(self, event, data, sid)

    def refresh_loop():
        while True:
            time.sleep(_REFRESH_SEC)
            server = holder["server"]
            if server is None:
                server = getattr(PromptServer, "instance", None)
            if server is None:
                continue
            try:
                live = status.live_from_server(server)
                with lock:
                    current = holder["snap"]
                    nxt = status.refresh_live(current, live)
                    if _visible(nxt) == _visible(current):
                        continue
                    holder["snap"] = nxt
                    persist(nxt)
            except Exception:
                _log.exception("failed to refresh Omarchy status file")

    PromptServer.send_sync = send_sync
    try:
        persist(holder["snap"])
    except Exception:
        _log.exception("failed to write initial Omarchy status file")
    else:
        _log.info("Omarchy status file: %s", path)

    thread = threading.Thread(target=refresh_loop, name="omarchy-status-refresh", daemon=True)
    thread.start()


_patch()
