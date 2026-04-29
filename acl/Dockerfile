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
        "pyyaml>=6.0" \
        "prometheus-fastapi-instrumentator>=7.0.0"

COPY acl ./acl

WORKDIR /app/acl

# App on 8081 (per spec §3 + packet A1). Metrics on the same port for
# now via the in-app /metrics route; A1 spec calls for a separate
# uvicorn instance on 9090 — operators can split via a second process
# command if needed (see README).
EXPOSE 8081

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
