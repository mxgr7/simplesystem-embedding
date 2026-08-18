# syntax=docker/dockerfile:1.7
#
# `service` is the DEPLOYED target. It carries no torch: the CUDA runtime is
# bind-mounted from /data at PYTHONPATH=/runtime, exactly as splade-service's
# `backend` target does on the same box. That is not a stylistic choice --
# `semantic-search.prod.nextgen` has Docker root on `/`, which is 88% full with
# ~6 GB free, and `pytorch/pytorch:...-cuda12.8-cudnn9-runtime` unpacks to
# 9-11 GB. Pulling it would fill the root filesystem of a host that is also
# running the entire semantic stack. /data has 802 GB.
#
# See README.md §Runtime provisioning for the one-time `uv pip install --target`
# that populates /data/ce-service/runtime.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS service

ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
WORKDIR /app
RUN uv pip install --system --no-cache \
      "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "orjson>=3.10,<4" \
      "numpy>=1.26,<3" \
      "prometheus-client>=0.20,<1" \
      "prometheus-fastapi-instrumentator>=7,<8"
ENV PYTHONPATH=/runtime
COPY . /app
EXPOSE 8140
# --limit-concurrency is a second, cruder bound than MAX_INFLIGHT: it caps the
# ASGI server's own queue so a burst is refused at the socket rather than
# accumulating handlers waiting on a GPU that can only run one forward.
CMD ["uvicorn", "ceserve.app:app", "--host", "0.0.0.0", "--port", "8140", \
     "--limit-concurrency", "16", "--no-access-log"]

# Self-contained CPU image: CI, local development, and the fp32 parity cell of
# the score-agreement check. CE_ALLOW_CPU=1 is required to actually serve from
# it -- see scorer.resolve_device.
FROM service AS service-bundled
RUN uv pip install --system --no-cache \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.8.0"
RUN uv pip install --system --no-cache \
      "transformers>=5,<6" "tokenizers>=0.22,<1" "safetensors>=0.4,<1"

# Self-contained CUDA image, for a host that has the disk. NOT the prod T4 --
# see the header. Kept so a dedicated g4dn.xlarge is a compose-file change
# rather than a Dockerfile change.
#
# No triton, and therefore no torch.compile: it needs a C compiler the slim
# images do not have, and on splade that combination made /encode-packed return
# 500 while /health, /readyz and /embed all stayed green. `bench_ce_t4.py` also
# measured compile as a LOSS for this student (57.4 ms compiled+fixed against
# 32.7 ms plain pad-to-longest), so there is nothing to buy back.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS service-cuda
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1
WORKDIR /app
RUN uv pip install --system --no-cache \
      "fastapi==0.135.1" "uvicorn==0.41.0" "orjson==3.11.8" "numpy==2.4.3" \
      "transformers==5.3.0" "tokenizers==0.22.2" "safetensors==0.6.2" \
      "prometheus-client==0.24.1" \
      "prometheus-fastapi-instrumentator==7.1.0"
COPY . /app
EXPOSE 8140
CMD ["uvicorn", "ceserve.app:app", "--host", "0.0.0.0", "--port", "8140", \
     "--limit-concurrency", "16", "--no-access-log"]
