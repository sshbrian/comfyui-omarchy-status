"""Publish ComfyUI queue/progress events to a JSON file for the Omarchy bar."""

import importlib.util
import logging
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

_log = logging.getLogger("comfyui_omarchy_status")


def _load_status():
    path = Path(__file__).resolve().parent / "status.py"
    spec = importlib.util.spec_from_file_location("omarchy_comfy_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch():
    status = _load_status()
    try:
        from server import PromptServer
    except Exception as exc:
        _log.warning("PromptServer unavailable; status file disabled: %s", exc)
        return

    orig = PromptServer.send_sync
    snap = status.empty_snapshot()
    path = status.default_path()

    def send_sync(self, event, data, sid=None):
        nonlocal snap
        if isinstance(event, str):
            try:
                nxt = status.apply_event(snap, event, data)
                if nxt is not None:
                    snap = nxt
                    status.write_snapshot(path, snap)
            except Exception:
                _log.exception("failed to update Omarchy status file")
        return orig(self, event, data, sid)

    PromptServer.send_sync = send_sync
    try:
        status.write_snapshot(path, snap)
    except Exception:
        _log.exception("failed to write initial Omarchy status file")
    else:
        _log.info("Omarchy status file: %s", path)


_patch()
