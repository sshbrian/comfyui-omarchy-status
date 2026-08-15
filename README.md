# comfyui-omarchy-status

A ComfyUI custom node package that writes generation status to a JSON file so the sibling Omarchy bar plugin (`omarchy-comfyui-status-plugin`) can show Idle, a progress bar, and it/s.

It does not add any graph nodes. On load it wraps `PromptServer.send_sync` and snapshots `status`, `progress`, and `executing` events.

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <this-repo> comfyui-omarchy-status
```

Restart ComfyUI. The file appears at:

```
${XDG_STATE_HOME:-~/.local/state}/omarchy/comfyui-status.json
```

This package is required for the sampler bar and it/s. Without it the Omarchy widget can still show Offline / Idle / Working… from `GET /prompt`.

## Snapshot

```json
{
  "schema": 1,
  "state": "idle",
  "value": 0,
  "max": 0,
  "queue_remaining": 0,
  "prompt_id": null,
  "node": null,
  "updated_at": 0,
  "last_event": null
}
```

`state` is `idle` or `running`. The bar derives Offline itself when ComfyUI does not answer HTTP.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```
