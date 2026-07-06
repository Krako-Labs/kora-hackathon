# KORA Router (AMD Developer Hackathon, Track 1)

A token-efficient routing agent. KORA decides, before any remote inference, how
each unit of work should be handled: resolved by cheap deterministic or local
computation, or escalated to a remote model only when nothing cheaper is
confident. Remote token usage is a direct, measurable consequence of that
routing.

## Status

This is the launch-day skeleton. The pipeline runs end to end with a placeholder
routing policy that escalates every task to the remote model, so the container
is runnable and testable before the task set is published. The real routing
logic (deterministic rules first, a local small model for cheap-but-non-trivial
work, remote only when needed) is added once the task I/O format and the allowed
models are known.

## Layout

```
kora_router/
  main.py             entry point: load tasks, route each, write results
  router.py           Route / RouteDecision / Router
  local_model.py      local backend
  fireworks_client.py remote backend (OpenAI-compatible Fireworks endpoint)
Dockerfile            python:3.11-slim image, single runtime dependency
requirements.txt      openai client (local backend deps added on launch day)
```

## Setup

```
pip install -r requirements.txt
```

Remote calls use the OpenAI-compatible Fireworks endpoint. Supply credentials
and the model id at run time via environment variables:

```
export FIREWORKS_API_KEY=...
export REMOTE_MODEL=accounts/fireworks/models/<model>
```

## Usage

```
python -m kora_router.main --tasks tasks.json --out results.json
```

Arguments:

- `--tasks` path to the task JSON (required). Accepts either a list of tasks or
  an object with a `tasks` key.
- `--out` output path (default `results.json`).
- `--remote-model` Fireworks model id. Falls back to the `REMOTE_MODEL`
  environment variable.

The output records, per task, the chosen route and the remote token count, plus
a summary with total remote tokens, remote call count, and route counts.

## Docker

```
docker build -t kora-router .
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e REMOTE_MODEL=accounts/fireworks/models/<model> \
  -v "$PWD":/data \
  kora-router --tasks /data/tasks.json --out /data/results.json
```

## License

MIT. See LICENSE.
