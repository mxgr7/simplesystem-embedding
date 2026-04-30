#!/usr/bin/env bash
# Deploy the TEI embedding server (mxgr/simplesystem-embedding) to a
# Vast.ai instance. Multi-GPU: launches one TEI per GPU and fronts them
# with nginx round-robin on :3000.
#
# Usage:  HF_TOKEN=<token> ./vast_embedding.sh <vast-offer-id>

set -euo pipefail
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
for i in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((3001 + i))
  (
    CUDA_VISIBLE_DEVICES=$i /entrypoint.sh \
      --model-id mxgr/simplesystem-embedding \
      --pooling mean \
      --port $PORT \
      --dtype float16 \
      --max-batch-tokens 1048576 \
      --max-concurrent-requests 8192 \
      --max-client-batch-size 4096 \
      --max-batch-requests 4096 \
      --tokenization-workers $WORKERS_PER_GPU \
      --auto-truncate \
      --payload-limit 10000000 \
      2>&1 | stdbuf -oL sed -u "s/^/[tei$i] /" | tee -a /var/log/tei.$i.log
  ) &
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

vastai create instance "$1" \
  --image ghcr.io/huggingface/text-embeddings-inference:1.9 \
  --env "-e HF_TOKEN=${HF_TOKEN} -p 3000:3000" \
  --disk 40 \
  --entrypoint /bin/bash \
  --args -c "$STARTUP"
