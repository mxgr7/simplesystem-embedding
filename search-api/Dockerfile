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
        "python-dotenv>=1.0.1" \
        "pymilvus>=2.4.0" \
        "numpy>=1.26.0" \
        "prometheus-fastapi-instrumentator>=7.0.0" \
        "pyyaml>=6.0"

COPY search-api ./search-api

WORKDIR /app/search-api

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
