#!/usr/bin/env bash
# Deploy the TEI embedding server (mxgr/simplesystem-embedding) to a
# Vast.ai instance. Multi-GPU: launches one TEI per GPU and fronts them
# with nginx round-robin on :3000.
#
# Usage:  HF_TOKEN=<token> ./vast_embedding.sh [--arch NAME] [--max-batch-tokens N] <vast-offer-id>

set -euo pipefail

MAX_BATCH_TOKENS=1048576
ARCH=ampere

usage() {
  cat <<USAGE
Usage: HF_TOKEN=<token> $0 [--arch NAME] [--max-batch-tokens N] <vast-offer-id>

  --arch NAME           GPU architecture; picks the matching TEI image tag.
                        One of: turing, ampere, ampere86, ada, hopper.
                          turing   -> T4, RTX 20xx
                          ampere   -> A100, A30                  (default)
                          ampere86 -> A10, RTX 30xx
                          ada      -> L4, L40, RTX 40xx
                          hopper   -> H100, H200
  --max-batch-tokens N  tokens packed per GPU forward pass; all other
                        TEI limits scale from this. (default ${MAX_BATCH_TOKENS})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)             ARCH=$2; shift 2;;
    --max-batch-tokens) MAX_BATCH_TOKENS=$2; shift 2;;
    -h|--help)          usage; exit 0;;
    -*)                 echo "unknown flag: $1" >&2; usage >&2; exit 2;;
    *)                  break;;
  esac
done

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
OFFER_ID=$1

case "$ARCH" in
  turing)   IMAGE_TAG=turing-1.9 ;;
  ampere)   IMAGE_TAG=1.9 ;;
  ampere86) IMAGE_TAG=86-1.9 ;;
  ada)      IMAGE_TAG=89-1.9 ;;
  hopper)   IMAGE_TAG=hopper-1.9 ;;
  *)        echo "unknown --arch: $ARCH" >&2; usage >&2; exit 2;;
esac

# Derivations assume ~256 tok/req floor and ~2.5KB/req on the wire.
MAX_BATCH_REQUESTS=$(( MAX_BATCH_TOKENS / 256 ))
MAX_CLIENT_BATCH_SIZE=$MAX_BATCH_REQUESTS
MAX_CONCURRENT_REQUESTS=$(( MAX_BATCH_REQUESTS * 2 ))
PAYLOAD_LIMIT=$(( MAX_CLIENT_BATCH_SIZE * 2500 ))

: "${HF_TOKEN:?HF_TOKEN must be set (used to pull the embedding model from HF)}"

STARTUP=$(cat <<'EOF'
set -e

# install cloudflared + nginx
apt-get update
mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | tee /etc/apt/sources.list.d/cloudflared.list
apt-get update
apt-get install -y cloudflared nginx

# start cloudflare tunnel in background
nohup cloudflared tunnel run \
  --token eyJhIjoiOTI3NmU4Njk1M2Y3MmZjNWY4YTJlODdhN2Y2OWM1MDUiLCJ0IjoiMmJmYTIxNzQtNTUyNC00ZWRkLWJmOTMtMTllNTg4ZmIwYTIxIiwicyI6Ill6ZGtZekEzTnpJdFltTTJaaTAwT1RjNUxXSXdOemN0TkdaalpUSTRaR0V3TjJOaSJ9 \
  >/var/log/cloudflared.log 2>&1 &

mkdir -p /opt

# node_exporter on :9100 (cpu/ram/disk/net)
curl -sL https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz \
  | tar xz -C /opt
nohup /opt/node_exporter-1.8.2.linux-amd64/node_exporter \
  --web.listen-address=:9100 \
  >/var/log/node_exporter.log 2>&1 &

# nvidia_gpu_exporter on :9835 (covers all GPUs from one process)
curl -sL https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.3.1/nvidia_gpu_exporter_1.3.1_linux_x86_64.tar.gz \
  | tar xz -C /opt
nohup /opt/nvidia_gpu_exporter \
  --web.listen-address=:9835 \
  >/var/log/gpu_exporter.log 2>&1 &

# detect GPUs / CPUs and split tokenization workers across GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
TOTAL_CPUS=$(nproc)
WORKERS_PER_GPU=$(( TOTAL_CPUS / NUM_GPUS ))
[ "$WORKERS_PER_GPU" -lt 8 ] && WORKERS_PER_GPU=8
echo "tei: $NUM_GPUS gpu(s), $TOTAL_CPUS cpu(s), $WORKERS_PER_GPU tokenization workers/gpu"

# launch one TEI per GPU on 3001..300N
# logs stream to main stdout (visible via `vastai logs`) with a [tei<i>] prefix
# and are tee'd to /var/log/tei.<i>.log for later inspection
launch_tei() {
  local i=$1
  local PORT=$((3001 + i))
  (
    CUDA_VISIBLE_DEVICES=$i /entrypoint.sh \
      --model-id mxgr/simplesystem-embedding \
      --pooling mean \
      --port $PORT \
      --dtype float16 \
      --max-batch-tokens $MAX_BATCH_TOKENS \
      --max-concurrent-requests $MAX_CONCURRENT_REQUESTS \
      --max-client-batch-size $MAX_CLIENT_BATCH_SIZE \
      --max-batch-requests $MAX_BATCH_REQUESTS \
      --tokenization-workers $WORKERS_PER_GPU \
      --auto-truncate \
      --payload-limit $PAYLOAD_LIMIT \
      2>&1 | stdbuf -oL sed -u "s/^/[tei$i] /" | tee -a /var/log/tei.$i.log
  ) &
}

# Launch tei0 alone first so it populates the HF cache without contention.
# If we launch all N at once they race on the cache lock; only the winner
# downloads model.safetensors, the losers fall through to pytorch_model.bin
# (which doesn't exist in this repo) and exit. Once tei0 reports healthy,
# the on-disk cache is complete and tei1..N start instantly without re-downloading.
echo "starting tei0 first to warm HF cache..."
launch_tei 0
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:3001/health" >/dev/null && break
  sleep 5
done

# fan out the rest now that the cache is hot
for i in $(seq 1 $((NUM_GPUS - 1))); do
  launch_tei $i
done

# nginx round-robin on :3000 -> backends
UPSTREAMS=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  UPSTREAMS="${UPSTREAMS}    server 127.0.0.1:$((3001 + i));\n"
done
rm -f /etc/nginx/sites-enabled/default
cat >/etc/nginx/sites-available/tei <<NGINX
upstream tei {
$(printf "%b" "$UPSTREAMS")
}
server {
  listen 3000;
  client_max_body_size 100m;
  proxy_read_timeout  600s;
  proxy_send_timeout  600s;
  location = /_status { return 200 "nginx-ok\n"; }
  location / {
    proxy_pass http://tei;
    proxy_set_header Host \$host;
    proxy_http_version 1.1;
  }
}
NGINX
ln -sf /etc/nginx/sites-available/tei /etc/nginx/sites-enabled/tei
nginx -t

# wait until every backend is healthy before fronting them
for i in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((3001 + i))
  echo "waiting for tei backend on :$PORT ..."
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 5
  done
done

exec nginx -g 'daemon off;'
EOF
)

vastai create instance "$OFFER_ID" \
  --image ghcr.io/huggingface/text-embeddings-inference:${IMAGE_TAG} \
  --env "-e HF_TOKEN=${HF_TOKEN} -e MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS} -e MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS} -e MAX_CLIENT_BATCH_SIZE=${MAX_CLIENT_BATCH_SIZE} -e MAX_BATCH_REQUESTS=${MAX_BATCH_REQUESTS} -e PAYLOAD_LIMIT=${PAYLOAD_LIMIT} -p 3000:3000" \
  --disk 40 \
  --entrypoint /bin/bash \
  --args -c "$STARTUP"
