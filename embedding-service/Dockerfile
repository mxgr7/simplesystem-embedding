# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "fastapi>=0.115.0" \
        "uvicorn>=0.30.0" \
        "httpx>=0.28.0" \
        "redis>=5.0.0" \
        "numpy>=1.26.0" \
        "jinja2>=3.1.5" \
        "pyyaml>=6.0" \
        "prometheus-client>=0.20.0" \
        "prometheus-fastapi-instrumentator>=7.0.0"

# Flat-import layout: same convention as search-api / playground-app.
# `uvicorn main:app` works because WORKDIR is the service directory.
COPY embedding-service ./embedding-service

WORKDIR /app/embedding-service

EXPOSE 8082

# --limit-concurrency 256 is the connection-level airbag. App-level
# inflight cap (MAX_INFLIGHT) is the precise 429 layer; this catches
# runaways below it.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8082", \
     "--limit-concurrency", "256"]
