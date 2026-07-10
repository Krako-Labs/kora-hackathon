# KORA token-efficient routing agent, container image.
#
# The routing core is pure Python. Two model paths are baked in:
#   - remote: OpenAI-compatible client for Fireworks calls (counted tokens)
#   - local : quantized instruct model on CPU via llama.cpp (zero tokens)
# The local weights ship inside the image so the container is fully
# self-contained: no downloads at run time.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY kora_router ./kora_router
COPY models/Llama-3.2-3B-Instruct-Q4_K_M.gguf ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

ENV KORA_LOCAL_MODEL=/app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Default remote model preference. select_model only honors this when its
# basename matches an entry in the harness-injected ALLOWED_MODELS allow-list,
# so no model outside the allow-list can ever be selected. Keeping the
# preference pinned makes local validation and scored runs use the same model.
ENV REMOTE_MODEL=minimax-m3

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are supplied at
# run time by the scoring harness. The container reads /input/tasks.json and
# writes /output/results.json by default.

ENTRYPOINT ["python", "-m", "kora_router.main"]
