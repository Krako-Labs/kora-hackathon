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

# Default remote model preference. select_model only honors this when its
# basename matches an entry in the harness-injected ALLOWED_MODELS allow-list,
# so no model outside the allow-list can ever be selected. Keeping the
# preference pinned makes local validation and scored runs use the same model.
ENV REMOTE_MODEL=minimax-m3

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are supplied at
# run time by the scoring harness. The container reads /input/tasks.json and
# writes /output/results.json by default.

ENTRYPOINT ["python", "-m", "kora_router.main"]
