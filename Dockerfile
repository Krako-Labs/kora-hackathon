# KORA token-efficient routing agent, container image.
#
# The routing core is pure Python with a single runtime dependency (the
# OpenAI-compatible client used for remote Fireworks calls), so the base image
# stays slim. The local small-model layer (e.g. Gemma weights + runtime) is
# added on launch day once the allowed models and scoring environment are known.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kora_router ./kora_router

# FIREWORKS_API_KEY and REMOTE_MODEL are supplied at run time via env vars.
# Example:
#   docker run --rm -e FIREWORKS_API_KEY=... -e REMOTE_MODEL=accounts/fireworks/models/... \
#     -v "$PWD":/data kora-router \
#     python -m kora_router.main --tasks /data/tasks.json --out /data/results.json

ENTRYPOINT ["python", "-m", "kora_router.main"]
