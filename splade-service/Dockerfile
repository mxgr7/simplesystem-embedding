# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS frontend

ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
WORKDIR /app
RUN uv pip install --system --no-cache \
      "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "httpx>=0.28,<1" \
      "redis>=5,<7" "numpy>=1.26,<3" "jinja2>=3.1,<4" \
      "prometheus-client>=0.20,<1" \
      "prometheus-fastapi-instrumentator>=7,<8"
COPY . /app
EXPOSE 8137
# A high HTTP airbag above app-level encoder admission. This bounds cache hits
# and waiting requests without rejecting priority queries behind document work.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8137", "--limit-concurrency", "512", "--no-access-log"]

FROM frontend AS backend
ENV PYTHONPATH=/runtime
EXPOSE 8138
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8138", "--no-access-log"]

# Portable self-contained CPU image for hosts with adequate Docker storage.
# The dev compose uses `backend` and bind-mounts these dependencies from /data.
FROM frontend AS backend-bundled
RUN uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.8.0"
RUN uv pip install --system --no-cache \
      "transformers>=4.49,<6" \
      "huggingface-hub>=0.24,<2" "tokenizers>=0.21,<1"
EXPOSE 8138
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8138", "--no-access-log"]

# Self-contained CUDA backend for the dedicated bulk-indexing GPU.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS backend-cuda
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
WORKDIR /app
RUN uv pip install --system --no-cache \
      "fastapi==0.135.1" "uvicorn==0.41.0" "numpy==2.4.3" \
      "transformers==5.3.0" "huggingface-hub==1.6.0" \
      "tokenizers==0.22.2" "protobuf==6.33.5" \
      "sentencepiece==0.2.1"
COPY . /app
EXPOSE 8138
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8138", "--no-access-log"]
