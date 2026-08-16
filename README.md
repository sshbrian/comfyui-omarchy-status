# comfyui-omarchy-status

A ComfyUI custom node package that writes generation status to a JSON file so the sibling Omarchy bar plugin (`omarchy-comfyui-status-plugin`) can show Idle, a progress bar, it/s, and the click-to-open dashboard.

It does not add any graph nodes. On load it wraps `PromptServer.send_sync` and snapshots queue, progress, executing, and job lifecycle events. A small refresh thread fills in VRAM, queue split, and prompt facts (checkpoint, size, seed).

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/sshbrian/comfyui-omarchy-status.git
```

Restart ComfyUI. The file appears at:

```
${XDG_STATE_HOME:-~/.local/state}/omarchy/comfyui-status.json
```

Today's session totals are also persisted at `comfyui-session.json` next to that file so a Comfy restart does not wipe the day.

This package is required for the sampler bar, it/s, ETA, sparkline, facts, last job, and session totals on the click-to-open dashboard. Without it the Omarchy widget can still show Offline / Idle / Working… from `GET /prompt`.

## Snapshot

Schema 2:

```json
{
  "schema": 2,
  "state": "idle",
  "phase": "idle",
  "value": 0,
  "max": 0,
  "queue_remaining": 0,
  "queue_running": 0,
  "queue_pending": 0,
  "prompt_id": null,
  "node": null,
  "node_type": null,
  "node_title": null,
  "updated_at": 0,
  "last_event": null,
  "step_times": [],
  "facts": {
    "checkpoint": null,
    "width": null,
    "height": null,
    "seed": null,
    "steps": null,
    "sampler": null,
    "cfg": null,
    "batch": null
  },
  "last_job": null,
  "vram": { "name": null, "used": 0, "total": 0 },
  "session": {
    "day": "2026-08-16",
    "gens": 0,
    "failures": 0,
    "interrupts": 0,
    "gpu_sec": 0
  }
}
```

`state` is `idle` or `running`. `phase` is a high-level activity: `idle`, `queued`, `loading`, `sampling`, `decoding`, `saving`, `working`, `error`, or `interrupted`. The bar derives Offline itself when ComfyUI does not answer HTTP.

`step_times` is this sampler's recent step durations in seconds (the Omarchy panel draws the sparkline and ETA from it). `last_job` and `session` are this-machine totals for today, not a full history.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
