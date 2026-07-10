# KORA: Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II, Track 1** · Krako Labs

Most routing tools decide which model should handle a task. KORA asks a different question first: **does this task need a remote call at all?**

## How it works

Every task flows through three tiers. The cheapest correct path wins, and the remote model is the exception path, not the default.

```
Task
 |
 v
[1] Deterministic rules      -> resolved: answer returned, 0 remote tokens
 |   provably correct computation, no model at all
 |   answer-blind, conservative, never guesses
 v
[2] Local model (CPU)        -> resolved: gate passed, 0 remote tokens
 |   quantized instruct model shipped inside the container
 |   an output validation gate checks every answer;
 |   empty, runaway, or malformed output escalates instead of shipping
 v
[3] Remote model             -> counted: tokens score against the run
     Fireworks endpoint, model chosen from ALLOWED_MODELS
     reached only when nothing cheaper is trusted
```

Design principles:

- **Accuracy-gate-first.** A deterministic rule fires only when its output is certain to be correct. The local tier is not trusted blindly either: every local answer passes a form-level validation gate, and anything suspicious escalates. The router never trades accuracy for tokens.
- **Answer-blind routing.** Every routing decision is made from the request text alone, never from a peeked ground truth or a per-dataset lookup.
- **No estimator overhead.** Unlike cascade approaches that run a quality-estimator model to decide, the front-door is a direct check. There is no extra model call in the hot path.
- **Self-contained container.** The local model weights are baked into the image at build time. Nothing is downloaded at run time.

## Scoring-harness contract

The container runs with no arguments:

- reads tasks from `/input/tasks.json` as `[{"task_id": ..., "prompt": ...}, ...]`
- writes answers to `/output/results.json` as `[{"task_id": ..., "answer": ...}, ...]`
- reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the environment
- routes all remote calls through `FIREWORKS_BASE_URL`, with the model chosen strictly from `ALLOWED_MODELS`
- a single failing task never crashes the run: it is recorded with an empty answer and the container still writes a complete results file and exits 0

## Repository layout

```
kora_router/
  main.py              entry point and front-door routing policy
  router.py            three-tier routing core, validation gate, output hygiene
  deterministic.py     safe arithmetic evaluator (rule tier)
  local_model.py       llama.cpp CPU backend for the in-container model
  fireworks_client.py  OpenAI-compatible client for the Fireworks endpoint
eval/
  run_eval.py          offline eval of the deterministic tier (no remote calls)
  run_practice.py      end-to-end practice run through the full pipeline
Dockerfile
requirements.txt
```

## Building the image

The local model weights are not committed to the repository. Download the GGUF into `models/` before building:

```bash
mkdir -p models
wget -O models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

docker build -t kora-router .
```

## Running locally

Against a task file, with the harness contract:

```bash
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=... \
  ghcr.io/hkalbertkim/kora-router:latest
```

The run prints a routing summary to stdout, for example:

```
wrote /output/results.json: 8 tasks, 273 remote tokens, 1 remote calls,
routes={'local': 7, 'remote': 1}
```

An optional debug sidecar (`--debug-out` or `KORA_DEBUG_OUT`) records the route, reason, and per-task prompt/completion token split for every task.

## Local validation before any submission

- `eval/run_eval.py` checks the deterministic tier offline: deterministic precision, math coverage, and over-routing (deflecting a task that should escalate) with a target of zero.
- `eval/run_practice.py` runs the published practice tasks through the full pipeline (deterministic, local with validation gate, remote escalation) and reports routes and remote token spend.

No performance number is claimed until it is measured; leaderboard figures come from the AMD automated scoring run.

## License

See [LICENSE](LICENSE).
