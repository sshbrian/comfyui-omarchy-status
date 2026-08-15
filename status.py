import json
import os
import time
from pathlib import Path

SCHEMA = 2
STEP_TIMES_MAX = 48
ERROR_MAX_LEN = 200

HANDLED_EVENTS = frozenset(
    {
        "status",
        "progress",
        "executing",
        "execution_start",
        "execution_success",
        "execution_error",
        "execution_interrupted",
    }
)

_LOADER_MARKERS = (
    "loader",
    "load_",
    "checkpoint",
    "unet",
    "lora",
    "clip",
    "dualcliploader",
)
_DECODE_MARKERS = ("vaedecode", "vae decode", "decode")
_SAVE_MARKERS = ("saveimage", "save_image", "previewimage", "save video", "vhs_videocombine")
_SAMPLE_MARKERS = ("ksampler", "sampler", "sample", "unipc", "randomnoise")
_LATENT_EMPTY_MARKERS = ("emptylatent", "empty_latent", "emptysd3", "emptyflux", "emptyhunyuan")


def default_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home) / "omarchy" / "comfyui-status.json"
    return Path.home() / ".local" / "state" / "omarchy" / "comfyui-status.json"


def default_session_path() -> Path:
    return default_path().with_name("comfyui-session.json")


def empty_facts() -> dict:
    return {
        "checkpoint": None,
        "width": None,
        "height": None,
        "seed": None,
        "steps": None,
        "sampler": None,
        "cfg": None,
        "batch": None,
    }


def empty_vram() -> dict:
    return {"name": None, "used": 0, "total": 0}


def today_str(now=None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))


def empty_session(now=None) -> dict:
    return {
        "day": today_str(now),
        "gens": 0,
        "failures": 0,
        "interrupts": 0,
        "gpu_sec": 0.0,
    }


def empty_snapshot(now=None) -> dict:
    return {
        "schema": SCHEMA,
        "state": "idle",
        "phase": "idle",
        "value": 0,
        "max": 0,
        "queue_remaining": 0,
        "queue_running": 0,
        "queue_pending": 0,
        "prompt_id": None,
        "node": None,
        "node_type": None,
        "node_title": None,
        "updated_at": 0.0,
        "last_event": None,
        "step_times": [],
        "step_at": None,
        "job_started_at": None,
        "facts": empty_facts(),
        "last_job": None,
        "vram": empty_vram(),
        "session": empty_session(now),
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


def _as_int(value):
    n = _number(value, None)
    if n is None:
        return None
    try:
        return int(n)
    except (TypeError, ValueError, OverflowError):
        return None


def _clip_error(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > ERROR_MAX_LEN:
        return text[: ERROR_MAX_LEN - 1] + "…"
    return text


def _widget(node, key):
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    value = inputs.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    return None


def _basename(value):
    text = str(value).strip()
    if not text:
        return None
    return text.replace("\\", "/").rstrip("/").split("/")[-1]


def extract_facts(prompt) -> dict:
    facts = empty_facts()
    if not isinstance(prompt, dict):
        return facts

    checkpoint = None
    unet = None
    size = None
    size_prio = 0
    seed = None
    seed_prio = 0
    steps = None
    sampler = None
    cfg = None
    batch = None

    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        lowered = class_type.lower()

        ckpt = _widget(node, "ckpt_name") or _widget(node, "ckpt_path")
        if ckpt:
            checkpoint = _basename(ckpt)
        model_name = _widget(node, "unet_name") or _widget(node, "model_name")
        if model_name and ("unet" in lowered or class_type in ("UNETLoader", "UnetLoaderGGUF")):
            unet = _basename(model_name)
        elif model_name and checkpoint is None:
            checkpoint = _basename(model_name)

        width = _as_int(_widget(node, "width"))
        height = _as_int(_widget(node, "height"))
        if width and height and width > 0 and height > 0:
            prio = 2 if any(mark in lowered.replace(" ", "") for mark in _LATENT_EMPTY_MARKERS) else 1
            if prio >= size_prio:
                size = (width, height)
                size_prio = prio

        node_seed = _widget(node, "seed")
        node_noise = _widget(node, "noise_seed")
        if node_seed is not None:
            prio = 2 if "sampler" in lowered else 1
            if prio >= seed_prio:
                seed = _as_int(node_seed)
                seed_prio = prio
        elif node_noise is not None and seed_prio < 2:
            seed = _as_int(node_noise)
            seed_prio = 1

        node_steps = _as_int(_widget(node, "steps"))
        if node_steps is not None:
            steps = node_steps
        node_sampler = _widget(node, "sampler_name")
        if node_sampler:
            sampler = str(node_sampler)
        node_cfg = _widget(node, "cfg")
        if node_cfg is not None:
            cfg = _number(node_cfg, None)
        node_batch = _as_int(_widget(node, "batch_size"))
        if node_batch is not None:
            batch = node_batch

    facts["checkpoint"] = unet or checkpoint
    if size:
        facts["width"], facts["height"] = size
    facts["seed"] = seed
    facts["steps"] = steps
    facts["sampler"] = sampler
    facts["cfg"] = cfg
    facts["batch"] = batch
    return facts


def node_class_type(prompt, node_id):
    if not isinstance(prompt, dict) or node_id is None:
        return None
    node = prompt.get(str(node_id))
    if node is None:
        node = prompt.get(node_id)
    if isinstance(node, dict):
        class_type = node.get("class_type")
        return str(class_type) if class_type else None
    return None


def node_title(extra_data, node_id):
    if node_id is None or not isinstance(extra_data, dict):
        return None
    png = extra_data.get("extra_pnginfo")
    workflow = png.get("workflow") if isinstance(png, dict) else None
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        return None
    want = str(node_id)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("id")) != want:
            continue
        title = node.get("title")
        if title:
            return str(title)
        return None
    return None


def _blob(*parts):
    return " ".join(str(part).lower() for part in parts if part)


def derive_phase(snap) -> str:
    data = snap if isinstance(snap, dict) else {}
    last = data.get("last_event")
    if last == "execution_error" and data.get("job_started_at") is None:
        return "error"
    if last == "execution_interrupted" and data.get("job_started_at") is None:
        return "interrupted"
    if data.get("state") != "running":
        return "idle"

    if last == "progress" and _number(data.get("max"), 0) > 0:
        return "sampling"

    blob = _blob(data.get("node_type"), data.get("node_title"))
    if any(mark in blob for mark in _SAMPLE_MARKERS):
        return "sampling"
    if any(mark in blob for mark in _DECODE_MARKERS):
        return "decoding"
    if any(mark in blob for mark in _SAVE_MARKERS):
        return "saving"
    if any(mark in blob for mark in _LOADER_MARKERS):
        return "loading"
    if last in ("status", "execution_start") and not data.get("node"):
        return "queued"
    return "working"


def normalize_session(session, now=None) -> dict:
    now = time.time() if now is None else now
    day = today_str(now)
    if not isinstance(session, dict) or session.get("day") != day:
        return empty_session(now)
    return {
        "day": day,
        "gens": max(0, int(_number(session.get("gens"), 0))),
        "failures": max(0, int(_number(session.get("failures"), 0))),
        "interrupts": max(0, int(_number(session.get("interrupts"), 0))),
        "gpu_sec": max(0.0, float(_number(session.get("gpu_sec"), 0))),
    }


def load_session(path=None, now=None) -> dict:
    dest = Path(path) if path is not None else default_session_path()
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return empty_session(now)
    return normalize_session(data, now)


def write_session(path, session) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    body = json.dumps(session, separators=(",", ":"), ensure_ascii=True)
    tmp.write_text(body + "\n", encoding="utf-8")
    os.replace(tmp, dest)


def _copy_snap(snap, now=None) -> dict:
    out = empty_snapshot(now)
    if not isinstance(snap, dict):
        return out
    out.update(snap)
    out["schema"] = SCHEMA
    out["step_times"] = list(out.get("step_times") or [])
    facts = empty_facts()
    if isinstance(out.get("facts"), dict):
        facts.update(out["facts"])
    out["facts"] = facts
    out["session"] = normalize_session(out.get("session"), now)
    if isinstance(out.get("last_job"), dict):
        out["last_job"] = dict(out["last_job"])
    else:
        out["last_job"] = None
    vram = empty_vram()
    if isinstance(out.get("vram"), dict):
        vram.update(out["vram"])
    out["vram"] = vram
    return out


def _clear_current(out) -> None:
    out["value"] = 0
    out["max"] = 0
    out["prompt_id"] = None
    out["node"] = None
    out["node_type"] = None
    out["node_title"] = None
    out["step_times"] = []
    out["step_at"] = None
    out["job_started_at"] = None
    out["facts"] = empty_facts()


def _finish_job(out, prev, now, status, error=None) -> None:
    started = prev.get("job_started_at")
    if started is None and not prev.get("prompt_id"):
        return
    duration = now - started if started else 0
    out["last_job"] = {
        "prompt_id": prev.get("prompt_id"),
        "status": status,
        "duration_sec": round(max(0.0, duration), 2),
        "ended_at": now,
        "node": prev.get("node"),
        "node_type": prev.get("node_type"),
        "error": _clip_error(error),
        "facts": dict(prev.get("facts") or empty_facts()),
    }
    session = normalize_session(prev.get("session"), now)
    session["gpu_sec"] = round(session.get("gpu_sec", 0) + max(0.0, duration), 2)
    if status == "ok":
        session["gens"] = session.get("gens", 0) + 1
    elif status == "error":
        session["failures"] = session.get("failures", 0) + 1
    elif status == "interrupted":
        session["interrupts"] = session.get("interrupts", 0) + 1
    out["session"] = session
    out["job_started_at"] = None


def _ensure_job(out, prev, prompt_id, now) -> None:
    prev_id = prev.get("prompt_id")
    if prev.get("job_started_at") and prev_id and prompt_id and prev_id != prompt_id:
        _finish_job(out, prev, now, "ok")
    if out.get("job_started_at") is None:
        out["job_started_at"] = now
    if prompt_id:
        out["prompt_id"] = prompt_id


def _record_step(out, prev, payload, now) -> None:
    prev_value = _number(prev.get("value"), 0)
    value = _number(payload.get("value"), 0)
    same = (
        prev.get("prompt_id") == payload.get("prompt_id")
        and prev.get("node") == payload.get("node")
        and _number(prev.get("max"), 0) == _number(payload.get("max"), 0)
    )
    times = list(prev.get("step_times") or [])
    if same and value > prev_value:
        started = prev.get("step_at")
        if started is None:
            started = prev.get("updated_at")
        dt = now - _number(started, now)
        if 0 < dt < 600:
            times.append(round(dt, 4))
            times = times[-STEP_TIMES_MAX:]
    elif not same:
        times = []
    out["step_times"] = times
    out["step_at"] = now


def _queue_item_prompt_id(item):
    try:
        return item[1]
    except (TypeError, IndexError):
        return None


def _queue_item_prompt(item):
    try:
        return item[2]
    except (TypeError, IndexError):
        return None


def _queue_item_extra(item):
    try:
        return item[3]
    except (TypeError, IndexError):
        return None


def find_queue_item(running, pending, prompt_id):
    for group in (running, pending):
        if not isinstance(group, (list, tuple)):
            continue
        for item in group:
            if prompt_id is not None and _queue_item_prompt_id(item) == prompt_id:
                return item
    if isinstance(running, (list, tuple)) and running:
        return running[0]
    return None


def read_vram() -> dict:
    try:
        import comfy.model_management as mm
    except Exception:
        return empty_vram()
    try:
        device = mm.get_torch_device()
        total = mm.get_total_memory(device)
        free = mm.get_free_memory(device)
        name = mm.get_torch_device_name(device)
        return {
            "name": str(name) if name else None,
            "used": int(max(0, _number(total, 0) - _number(free, 0))),
            "total": int(_number(total, 0)),
        }
    except Exception:
        return empty_vram()


def live_from_server(server):
    if server is None:
        return None
    queue = getattr(server, "prompt_queue", None)
    if queue is None:
        return None
    try:
        running, pending = queue.get_current_queue_volatile()
    except Exception:
        return None
    return {
        "running": running or [],
        "pending": pending or [],
        "vram": read_vram(),
    }


def enrich_live(snap, live) -> dict:
    out = snap if isinstance(snap, dict) else empty_snapshot()
    if not isinstance(live, dict):
        return out

    running = live.get("running") or []
    pending = live.get("pending") or []
    if isinstance(running, (list, tuple)) or isinstance(pending, (list, tuple)):
        run_n = len(running) if isinstance(running, (list, tuple)) else 0
        pend_n = len(pending) if isinstance(pending, (list, tuple)) else 0
        out["queue_running"] = run_n
        out["queue_pending"] = pend_n
        if out.get("last_event") != "status":
            out["queue_remaining"] = run_n + pend_n

    vram = live.get("vram")
    if isinstance(vram, dict) and _number(vram.get("total"), 0) > 0:
        merged = empty_vram()
        merged.update(vram)
        out["vram"] = merged

    item = find_queue_item(running, pending, out.get("prompt_id"))
    prompt = _queue_item_prompt(item)
    extra = _queue_item_extra(item)
    if prompt:
        facts = extract_facts(prompt)
        if any(facts.get(key) is not None for key in facts):
            out["facts"] = facts
        node_id = out.get("node")
        if node_id is not None:
            class_type = node_class_type(prompt, node_id)
            if class_type:
                out["node_type"] = class_type
            title = node_title(extra, node_id)
            if title:
                out["node_title"] = title
    return out


def refresh_live(snap, live, now=None):
    if now is None:
        now = time.time()
    out = _copy_snap(snap, now)
    out = enrich_live(out, live)
    out["phase"] = derive_phase(out)
    return out


def apply_event(snap, event, data, now=None, live=None):
    """Return an updated snapshot, or None if the event is ignored."""
    if event not in HANDLED_EVENTS:
        return None

    if now is None:
        now = time.time()
    prev = snap if isinstance(snap, dict) else empty_snapshot(now)
    out = _copy_snap(prev, now)
    out["updated_at"] = now
    payload = data if isinstance(data, dict) else {}

    if event == "status":
        queue = _queue_remaining(payload)
        out["queue_remaining"] = queue
        if queue == 0:
            if prev.get("job_started_at") is not None:
                _finish_job(out, prev, now, "ok")
            out["state"] = "idle"
            _clear_current(out)
            out["last_event"] = "status"
        else:
            out["state"] = "running"
            if out.get("last_event") != "progress":
                out["last_event"] = "status"
        return _finalize(out, live)

    if event == "progress":
        _ensure_job(out, prev, payload.get("prompt_id"), now)
        _record_step(out, prev, payload, now)
        out["state"] = "running"
        out["value"] = _number(payload.get("value"), 0)
        out["max"] = _number(payload.get("max"), 0)
        out["prompt_id"] = payload.get("prompt_id")
        out["node"] = payload.get("node")
        out["last_event"] = "progress"
        return _finalize(out, live)

    if event == "executing":
        node = payload.get("node")
        prompt_id = payload.get("prompt_id")
        if prompt_id is not None:
            _ensure_job(out, prev, prompt_id, now)
            out["prompt_id"] = prompt_id
        if node is None:
            out["last_event"] = "executing"
            return _finalize(out, live)
        if node != out.get("node"):
            out["value"] = 0
            out["max"] = 0
            out["node"] = node
            out["node_type"] = None
            out["node_title"] = None
            out["step_times"] = []
            out["step_at"] = None
        out["state"] = "running"
        out["last_event"] = "executing"
        return _finalize(out, live)

    if event == "execution_start":
        prompt_id = payload.get("prompt_id")
        _ensure_job(out, prev, prompt_id, now)
        out["state"] = "running"
        out["prompt_id"] = prompt_id
        out["value"] = 0
        out["max"] = 0
        out["node"] = None
        out["node_type"] = None
        out["node_title"] = None
        out["step_times"] = []
        out["step_at"] = None
        out["last_event"] = "execution_start"
        return _finalize(out, live)

    if event == "execution_success":
        _finish_job(out, prev, now, "ok")
        out["last_event"] = "execution_success"
        out["value"] = 0
        out["max"] = 0
        out["node"] = None
        out["node_type"] = None
        out["node_title"] = None
        out["step_times"] = []
        out["step_at"] = None
        out["facts"] = empty_facts()
        return _finalize(out, live)

    if event == "execution_error":
        _finish_job(out, prev, now, "error", payload.get("exception_message"))
        out["last_event"] = "execution_error"
        out["node"] = payload.get("node_id", out.get("node"))
        out["node_type"] = payload.get("node_type") or out.get("node_type")
        return _finalize(out, live)

    # execution_interrupted
    _finish_job(out, prev, now, "interrupted")
    out["last_event"] = "execution_interrupted"
    out["node"] = payload.get("node_id", out.get("node"))
    out["node_type"] = payload.get("node_type") or out.get("node_type")
    return _finalize(out, live)


def _finalize(out, live):
    if live:
        out = enrich_live(out, live)
    out["phase"] = derive_phase(out)
    return out


def write_snapshot(path, snap) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    body = json.dumps(snap, separators=(",", ":"), ensure_ascii=True)
    tmp.write_text(body + "\n", encoding="utf-8")
    os.replace(tmp, dest)
