#!/usr/bin/env bash
# Deploy the Soup cross-encoder rerank server to a Vast.ai instance.
#
# Usage:  HF_TOKEN=<token> ./vast_cross_encoder.sh [options] <vast-offer-id>

set -euo pipefail

CKPT="${CKPT:-mxgr/simplesystem-cross-encoder:soup.ckpt}"
SERVE_DTYPE="${SERVE_DTYPE:-bf16}"
SERVE_MAX_BATCH="${SERVE_MAX_BATCH:-128}"
SERVE_COMPILE="${SERVE_COMPILE:-0}"

usage() {
  cat <<USAGE
Usage: HF_TOKEN=<token> $0 [options] <vast-offer-id>

  --ckpt SPEC       path or HF spec for the Soup CE checkpoint.
                    (default ${CKPT})
  --dtype DTYPE     autocast dtype: bf16|fp16|fp32|auto. bf16 ~2x batch
                    capacity at S=512 vs fp32, no measurable accuracy
                    delta on this 330M model. (default ${SERVE_DTYPE})
  --max-batch N     encoder forward chunk size. 128 is the bench'd 4090
                    sweet spot; re-bench for other GPUs. (default ${SERVE_MAX_BATCH})
  --compile {0,1}   torch.compile the encoder. 0 sets TORCHDYNAMO_DISABLE=1.
                    Compile showed 0% gain on the bench'd 4090; toggle
                    on to A/B on other GPUs. (default ${SERVE_COMPILE})

Pass-through env (forwarded if set): LGBM, TEMPERATURE, ENSEMBLE_W.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)      CKPT=$2; shift 2;;
    --dtype)     SERVE_DTYPE=$2; shift 2;;
    --max-batch) SERVE_MAX_BATCH=$2; shift 2;;
    --compile)   SERVE_COMPILE=$2; shift 2;;
    -h|--help)   usage; exit 0;;
    -*)          echo "unknown flag: $1" >&2; usage >&2; exit 2;;
    *)           break;;
  esac
done

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
OFFER_ID=$1

: "${HF_TOKEN:?HF_TOKEN must be set (used to pull the CE checkpoint from HF)}"

ENV_ARGS="-e HF_TOKEN=${HF_TOKEN} -e CKPT=${CKPT} -e SERVE_DTYPE=${SERVE_DTYPE} -e SERVE_MAX_BATCH=${SERVE_MAX_BATCH} -e SERVE_COMPILE=${SERVE_COMPILE}"
[[ "$SERVE_COMPILE" != "1" ]] && ENV_ARGS="$ENV_ARGS -e TORCHDYNAMO_DISABLE=1"
[[ -n "${LGBM:-}" ]]          && ENV_ARGS="$ENV_ARGS -e LGBM=${LGBM}"
[[ -n "${TEMPERATURE:-}" ]]   && ENV_ARGS="$ENV_ARGS -e TEMPERATURE=${TEMPERATURE}"
[[ -n "${ENSEMBLE_W:-}" ]]    && ENV_ARGS="$ENV_ARGS -e ENSEMBLE_W=${ENSEMBLE_W}"
ENV_ARGS="$ENV_ARGS -p 8080:8080"

STARTUP=$(cat <<'EOF'
set -e

# install cloudflared
apt-get update
mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | tee /etc/apt/sources.list.d/cloudflared.list
apt-get update
apt-get install -y cloudflared

# start cloudflare tunnel in background
nohup cloudflared tunnel run \
  --token eyJhIjoiOTI3NmU4Njk1M2Y3MmZjNWY4YTJlODdhN2Y2OWM1MDUiLCJ0IjoiNGI4ZTU5NjktZTU4MC00ZTJlLWI1MGUtODIzYmVkNzhkYmRiIiwicyI6Ik9UUTJaVFppTVdZdE5UQmhOaTAwTnpBM0xXSTJaRGd0TmpsaE1HTmpORGd5TUdOayJ9 \
  >/var/log/cloudflared.log 2>&1 &

mkdir -p /opt

# node_exporter on :9100 (cpu/ram/disk/net)
curl -sL https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz \
  | tar xz -C /opt
nohup /opt/node_exporter-1.8.2.linux-amd64/node_exporter \
  --web.listen-address=:9100 \
  >/var/log/node_exporter.log 2>&1 &

# nvidia_gpu_exporter on :9835 (gpu/vram/temp/power)
curl -sL https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.3.1/nvidia_gpu_exporter_1.3.1_linux_x86_64.tar.gz \
  | tar xz -C /opt
nohup /opt/nvidia_gpu_exporter \
  --web.listen-address=:9835 \
  >/var/log/gpu_exporter.log 2>&1 &

# launch the cross-encoder rerank FastAPI app (matches the image's CMD)
cd /app
exec uv run uvicorn cross_encoder_serve.server:app --host 0.0.0.0 --port 8080
EOF
)

vastai create instance "$OFFER_ID" \
  --image ghcr.io/mxgr7/simplesystem-embedding/cross-encoder-serve:sha-6c941eb \
  --env "$ENV_ARGS" \
  --disk 50 \
  --entrypoint /bin/bash \
  --args -c "$STARTUP"
