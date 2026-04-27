# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install only the playground's runtime deps — skip torch/lightning/transformers/
# mlflow/hydra/faiss that the root pyproject pulls for training but the app never
# imports.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "fastapi>=0.115.0" \
        "uvicorn>=0.30.0" \
        "httpx>=0.28.0" \
        "jinja2>=3.1.5" \
        "python-dotenv>=1.0.1" \
        "pymilvus>=2.4.0" \
        "duckdb>=1.5.0" \
        "pandas>=2.2.3" \
        "pyarrow>=19.0.1" \
        "prometheus-fastapi-instrumentator>=7.0.0"

COPY playground-app ./playground-app

WORKDIR /app/playground-app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
