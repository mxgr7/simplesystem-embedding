#!/usr/bin/env bash
# Deploy the Soup cross-encoder rerank server to a Vast.ai instance.
#
# Usage:
#   HF_TOKEN=<token> ./vast_cross_encoder.sh <vast-offer-id>
#
# CKPT defaults to mxgr/simplesystem-cross-encoder:soup.ckpt (HF Hub).
# Override for a different revision or a local/mounted path:
#   CKPT=mxgr/simplesystem-cross-encoder@<rev>:soup.ckpt
#   CKPT=/app/checkpoints/soup.ckpt
#
# LGBM defaults to the booster baked into the image at
# /app/artifacts/lgbm_soup.txt; override via env if you want to disable
# (LGBM=) or pull a different one.
#
# Server runs eager (no torch.compile). Compile gave 0% throughput
# improvement on this model+GPU class while costing ~20s/new-shape and
# ~25% of the per-request batch budget — see TORCHDYNAMO_DISABLE in the
# image entrypoint.

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set (used to pull the CE checkpoint from HF)}"

CKPT="${CKPT:-mxgr/simplesystem-cross-encoder:soup.ckpt}"

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

vastai create instance "$1" \
  --image ghcr.io/mxgr7/simplesystem-embedding/cross-encoder-serve:sha-b9f1592 \
  --env "-e CKPT=${CKPT} -e HF_TOKEN=${HF_TOKEN} -e TORCHDYNAMO_DISABLE=1 -p 8080:8080 -p 9100:9100 -p 9835:9835" \
  --disk 50 \
  --entrypoint /bin/bash \
  --args -c "$STARTUP"
