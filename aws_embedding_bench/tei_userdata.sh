#!/usr/bin/env bash
# EC2 user-data that runs TEI for the embedding-model benchmark. Port of
# the Vast.ai vast_embedding.sh launcher. Variables substituted in by the
# Pulumi program (Bash-style ${...} placeholders, not Pulumi outputs):
#   __HF_TOKEN__       — HuggingFace token for model pull
#   __TEI_IMAGE_TAG__  — TEI Docker tag matching the instance's GPU arch
#                        (e.g. turing-1.9, 86-1.9, 89-1.9, cpu-1.9)
#   __INSTANCE_KIND__  — "gpu" or "cpu"

set -euo pipefail
exec > >(tee -a /var/log/tei-userdata.log) 2>&1

# Drop the -x trace (would log the HF token); use an explicit log() instead.
log() { printf '[%s | tei-userdata] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
T0=$(date +%s)
elapsed() { echo "$(( $(date +%s) - T0 ))s"; }

HF_TOKEN="__HF_TOKEN__"
TEI_IMAGE_TAG="__TEI_IMAGE_TAG__"
INSTANCE_KIND="__INSTANCE_KIND__"
log "starting (image_tag=$TEI_IMAGE_TAG kind=$INSTANCE_KIND host=$(hostname))"

# AWS Deep Learning Base AMI (Ubuntu 22.04) ships with nvidia drivers + docker
# + nvidia-container-toolkit pre-installed. The plain c7i (CPU) box needs
# Docker manually.
if [ "$INSTANCE_KIND" = "cpu" ]; then
  log "CPU instance: installing docker + nginx (apt-get)"
  apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io nginx curl
  systemctl enable --now docker
fi

# Always need nginx for the multi-GPU round-robin (no-op when NUM_GPUS=1).
if ! command -v nginx >/dev/null 2>&1; then
  log "installing nginx"
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q nginx
fi

# Detect GPU count. CPU instance => 0.
if [ "$INSTANCE_KIND" = "gpu" ]; then
  log "GPU detection:"
  nvidia-smi -L 2>&1 | sed 's/^/    /' || log "  nvidia-smi failed!"
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NUM_GPUS=0
fi
TOTAL_CPUS=$(nproc)
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
log "hardware: kind=$INSTANCE_KIND gpus=$NUM_GPUS cpus=$TOTAL_CPUS mem=$(( MEM_KB / 1024 / 1024 ))GiB"

# Phase 1: probe the GPU's max-batch-tokens ceiling. We try a small descending
# ladder; first value that boots TEI healthy wins. The HF cache volume is
# shared, so subsequent attempts re-use the downloaded model (cold first try
# ~5min, warm retries ~30s). For CPU we skip the probe and use a fixed value.
PROBE_LADDER=(524288 262144 131072)
MAX_BATCH_TOKENS=""

IMAGE="ghcr.io/huggingface/text-embeddings-inference:${TEI_IMAGE_TAG}"
log "docker pull $IMAGE (elapsed $(elapsed))"
docker pull "$IMAGE" 2>&1 | sed 's/^/    /'
log "docker pull done (elapsed $(elapsed))"

mkdir -p /var/cache/tei

dump_container_log() {
  local name="$1"
  log "--- last 50 lines of $name ---"
  docker logs --tail 50 "$name" 2>&1 | sed 's/^/    /' || true
  log "--- end $name ---"
}

# Launch one TEI container for GPU index $1 with max-batch-tokens $2.
# Derives the other TEI limits from max-batch-tokens (same formulas as
# vast_embedding.sh). Returns 0 on healthy, 1 on OOM, 2 on other failure.
launch_tei() {
  local i=$1
  local tokens=$2
  local port=$((3001 + i))
  local container_name="tei-$i"
  local max_batch_requests=$(( tokens / 256 ))
  local max_concurrent_requests=$(( max_batch_requests * 2 ))
  local max_client_batch_size=$max_batch_requests
  local payload_limit=$(( max_client_batch_size * 2500 ))
  local gpu_args=() env_args=()
  if [ "$INSTANCE_KIND" = "gpu" ]; then
    gpu_args=(--gpus "device=$i")
    env_args=(-e "CUDA_VISIBLE_DEVICES=$i")
  fi
  local workers_per_proc=$(( TOTAL_CPUS / (NUM_GPUS > 0 ? NUM_GPUS : 1) ))
  [ "$workers_per_proc" -lt 8 ] && workers_per_proc=8

  log "starting $container_name on :$port  max_batch_tokens=$tokens  max_batch_requests=$max_batch_requests"
  docker rm -f "$container_name" >/dev/null 2>&1 || true
  docker run -d --restart unless-stopped --name "$container_name" \
    "${gpu_args[@]}" "${env_args[@]}" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -p "127.0.0.1:${port}:80" \
    -v /var/cache/tei:/data \
    "$IMAGE" \
    --model-id mxgr/simplesystem-embedding \
    --pooling mean --dtype float16 --auto-truncate \
    --max-batch-tokens "$tokens" \
    --max-concurrent-requests "$max_concurrent_requests" \
    --max-client-batch-size "$max_client_batch_size" \
    --max-batch-requests "$max_batch_requests" \
    --tokenization-workers "$workers_per_proc" \
    --payload-limit "$payload_limit" >/dev/null

  # Poll /health; on container exit, classify as OOM (1) vs other (2).
  local probe_t0=$(date +%s)
  for attempt in $(seq 1 60); do  # 60 * 5s = 300s max wait
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
      log "  $container_name healthy after $(( $(date +%s) - probe_t0 ))s"
      return 0
    fi
    if ! docker inspect "$container_name" -f '{{.State.Running}}' 2>/dev/null | grep -q true; then
      if docker logs "$container_name" 2>&1 | grep -qiE 'out.of.memory|cuda_error_out_of_memory'; then
        log "  $container_name OOMed at max_batch_tokens=$tokens after $(( $(date +%s) - probe_t0 ))s"
        return 1
      else
        log "  $container_name exited (non-OOM) after $(( $(date +%s) - probe_t0 ))s; logs:"
        docker logs --tail 30 "$container_name" 2>&1 | sed 's/^/    /' || true
        return 2
      fi
    fi
    if [ $((attempt % 12)) -eq 0 ]; then
      log "  ... $container_name still warming (attempt $attempt, $(( $(date +%s) - probe_t0 ))s); tail:"
      docker logs --tail 3 "$container_name" 2>&1 | sed 's/^/      /' || true
    fi
    sleep 5
  done
  log "  $container_name never became healthy within 300s; assuming hung"
  dump_container_log "$container_name"
  return 2
}

# Number of worker processes. For CPU we still launch 1.
WORKERS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))

# Phase 1 — probe max_batch_tokens (GPU only). For CPU just use 16K.
if [ "$INSTANCE_KIND" = "cpu" ]; then
  MAX_BATCH_TOKENS=16384
  log "CPU mode: skipping probe, using max_batch_tokens=$MAX_BATCH_TOKENS"
  launch_tei 0 "$MAX_BATCH_TOKENS" || dump_container_log tei-0
else
  log "phase 1: probing max-batch-tokens ceiling on GPU 0 (ladder: ${PROBE_LADDER[*]})"
  for tokens in "${PROBE_LADDER[@]}"; do
    rc=0
    launch_tei 0 "$tokens" || rc=$?
    if [ "$rc" = 0 ]; then
      MAX_BATCH_TOKENS=$tokens
      log "phase 1 WINNER: max_batch_tokens=$MAX_BATCH_TOKENS"
      break
    elif [ "$rc" = 1 ]; then
      log "phase 1: $tokens OOMed, trying next"
      continue
    else
      log "phase 1: $tokens failed non-OOM — aborting probe"
      break
    fi
  done
  if [ -z "$MAX_BATCH_TOKENS" ]; then
    log "phase 1 FALLBACK: all probes failed; trying 65536 as last resort"
    MAX_BATCH_TOKENS=65536
    launch_tei 0 "$MAX_BATCH_TOKENS" || dump_container_log tei-0
  fi
fi

# Surface the winning value to the orchestrator.
echo "$MAX_BATCH_TOKENS" > /var/run/tei-max-batch-tokens

# Fan out to remaining GPUs (if any) with the same winning value.
if [ "$WORKERS" -gt 1 ]; then
  log "fanning out workers 1..$((WORKERS - 1)) at max_batch_tokens=$MAX_BATCH_TOKENS"
  for i in $(seq 1 $((WORKERS - 1))); do
    launch_tei "$i" "$MAX_BATCH_TOKENS" || dump_container_log "tei-$i"
  done
fi

# nginx round-robin on :3000 -> backends
log "configuring nginx round-robin on :3000 -> $WORKERS backend(s)"
UPSTREAMS=""
for i in $(seq 0 $((WORKERS - 1))); do
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

# Wait until every backend is healthy before fronting them.
for i in $(seq 0 $((WORKERS - 1))); do
  port=$((3001 + i))
  log "waiting for tei backend on :$port"
  back_t0=$(date +%s)
  back_ok=""
  for attempt in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
      back_ok=yes
      log "  tei$i healthy after ${attempt} probes ($(( $(date +%s) - back_t0 ))s)"
      break
    fi
    sleep 5
  done
  if [ -z "$back_ok" ]; then
    log "  tei$i NEVER healthy after $(( $(date +%s) - back_t0 ))s"
    dump_container_log "tei-$i"
  fi
done

log "reloading nginx"
systemctl reload nginx || systemctl restart nginx

# Sentinel file the bench wrapper polls for via SSH to know we're up.
echo "ok" > /var/run/tei-ready
log "TEI server fully up (total elapsed $(elapsed)); listening on :3000"
log "docker ps:"
docker ps --format '    {{.Names}} {{.Status}} {{.Ports}}' || true
