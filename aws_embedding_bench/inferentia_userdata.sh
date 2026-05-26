#!/usr/bin/env bash
# EC2 user-data for Inferentia bench boxes (inf1.* and inf2.*).
#
# Strategy: skip in-place compile. Model is pre-compiled on a separate box
# (see compile.sh + /tei-models/) and SCP'd in by bench.sh after this script
# finishes installing the runtime deps.
#
# Pulumi substitutes these markers before passing as user-data:
#   __NEURON_VENV_GLOB__  /opt/aws_neuron_venv_pytorch_* (inf1)
#                         /opt/aws_neuronx_venv_pytorch_* (inf2)
#   __NUM_CORES__         number of NeuronCores on the chip
#                         inf1.* = 4 (per chip), inf2.xlarge = 2
#
# Hand-off sequence (same as before):
#   1. user-data installs runtime deps (fastapi/uvicorn + nginx), configures
#      one systemd unit per NeuronCore + an nginx upstream covering all of
#      them on :3000, writes /var/run/inf-deps-ready, exits
#   2. bench.sh polls /var/run/inf-deps-ready, then SCPs the .pt +
#      tokenizer/config files into /opt/neuron-model/ and neuron_server.py
#      into /opt/srv/, then `systemctl start neuron-tei-{0..N-1}` over SSH
#   3. nginx already running; /health returns 200; bench.sh runs loadgen

set -uo pipefail
exec > >(tee -a /var/log/inferentia-userdata.log) 2>&1

log() { printf '[%s | inf-ud] %s\n' "$(date -u +%FT%TZ)" "$*"; }
T0=$(date +%s); el() { echo "$(( $(date +%s) - T0 ))s"; }
DEPS_SENTINEL=/var/run/inf-deps-ready
NEURON_VENV_GLOB='__NEURON_VENV_GLOB__'
NUM_CORES=__NUM_CORES__
log "boot on $(hostname); kernel $(uname -r)"
log "venv-glob=$NEURON_VENV_GLOB num-cores=$NUM_CORES"

VENV=$(ls -d $NEURON_VENV_GLOB/ 2>/dev/null \
        | grep -Ev 'nxd_(training|inference)/?$' \
        | sort -V | tail -1 | sed 's:/$::')
if [ -z "$VENV" ]; then
  echo "fatal: no venv matched $NEURON_VENV_GLOB (wrong AMI?)" > "$DEPS_SENTINEL"
  exit 1
fi
log "venv: $VENV"
ls -la /dev/neuron* 2>&1 | sed 's/^/    /' || true

# Pick which Neuron package is shipped: inf2 venv contains torch_neuronx
# (with x); inf1 venv contains torch_neuron (without).
case "$VENV" in
  *aws_neuronx_venv_pytorch_*) NEURON_IMPORT="torch_neuronx" ;;
  *aws_neuron_venv_pytorch_*)  NEURON_IMPORT="torch_neuron"  ;;
  *) NEURON_IMPORT="torch_neuronx" ;;
esac
log "expected neuron package: $NEURON_IMPORT"

INSTALL_LOG=/var/log/neuron-install.log
# torch_neuron(x) import-time init shells out to a binary in $VENV/bin
# (libneuronpjrt-path on inf2; similar on inf1). Just invoking $VENV/bin/python
# doesn't put $VENV/bin on PATH — only `activate` does — so export explicitly.
export PATH="$VENV/bin:$PATH"

log "verifying torch + $NEURON_IMPORT importable in DLAMI venv"
if ! "$VENV/bin/python" -c "
import torch
neuron = __import__('$NEURON_IMPORT')
print('torch', torch.__version__, '$NEURON_IMPORT', neuron.__version__)
" >"$INSTALL_LOG" 2>&1; then
  log "$NEURON_IMPORT import FAILED — tail $INSTALL_LOG:"
  tail -20 "$INSTALL_LOG" | sed 's/^/    /' || true
  { echo "fatal: $NEURON_IMPORT import failed in DLAMI venv"; tail -20 "$INSTALL_LOG"; } > "$DEPS_SENTINEL"
  exit 1
fi
tail -1 "$INSTALL_LOG" | sed 's/^/    /'

log "pip install fastapi + uvicorn + pydantic (el $(el))"
if ! "$VENV/bin/pip" install --only-binary :all: \
      fastapi 'uvicorn[standard]' pydantic \
      >>"$INSTALL_LOG" 2>&1; then
  log "pip install FAILED — tail $INSTALL_LOG:"
  tail -40 "$INSTALL_LOG" | sed 's/^/    /' || true
  { echo "fatal: pip install failed"; tail -40 "$INSTALL_LOG"; } > "$DEPS_SENTINEL"
  exit 1
fi
log "pip install done (el $(el))"

# Make sure transformers is installed (DLAMI may or may not ship it).
if ! "$VENV/bin/python" -c 'import transformers; print("transformers", transformers.__version__)' \
      >>"$INSTALL_LOG" 2>&1; then
  log "transformers missing — installing"
  "$VENV/bin/pip" install --only-binary :all: 'transformers>=4.36,<4.44' 'tokenizers>=0.15,<1' \
        >>"$INSTALL_LOG" 2>&1 || true
fi
tail -3 "$INSTALL_LOG" | sed 's/^/    /'

mkdir -p /opt/srv /opt/neuron-model
chown -R ubuntu:ubuntu /opt/srv /opt/neuron-model

# --- install + configure nginx as a round-robin fan-out on :3000 ----------- #
log "apt-get update (el $(el)) — DLAMI ships a stale apt cache that 404s on nginx-common"
DEBIAN_FRONTEND=noninteractive apt-get update -q >>"$INSTALL_LOG" 2>&1 || true
log "installing nginx (el $(el))"
DEBIAN_FRONTEND=noninteractive apt-get install -y -q nginx >>"$INSTALL_LOG" 2>&1 \
  || { log "nginx install FAILED"; tail -20 "$INSTALL_LOG" | sed 's/^/    /';
       echo "fatal: nginx install failed" > "$DEPS_SENTINEL"; exit 1; }

# Build the nginx upstream block dynamically — one server line per core.
UPSTREAM_LINES=""
for CORE in $(seq 0 $((NUM_CORES - 1))); do
  PORT=$((3001 + CORE))
  UPSTREAM_LINES+="    server 127.0.0.1:${PORT};
"
done

cat >/etc/nginx/sites-available/neuron-tei <<NGINX
upstream neuron_tei {
${UPSTREAM_LINES}    keepalive 32;
}
server {
    listen 3000;
    location / {
        proxy_pass http://neuron_tei;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 120s;
        proxy_buffering off;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/neuron-tei /etc/nginx/sites-enabled/neuron-tei
rm -f /etc/nginx/sites-enabled/default
nginx -t >>"$INSTALL_LOG" 2>&1 \
  || { log "nginx -t FAILED"; tail -20 "$INSTALL_LOG" | sed 's/^/    /';
       echo "fatal: nginx config invalid" > "$DEPS_SENTINEL"; exit 1; }
systemctl restart nginx
log "nginx configured: :3000 -> 127.0.0.1:3001..$((3000 + NUM_CORES))"

# --- one systemd unit per NeuronCore --------------------------------------- #
# Each writes NEURON_RT_VISIBLE_CORES to claim exactly its assigned core.
for CORE in $(seq 0 $((NUM_CORES - 1))); do
  PORT=$((3001 + CORE))
  cat >/etc/systemd/system/neuron-tei-${CORE}.service <<UNIT
[Unit]
Description=Neuron TEI worker on NeuronCore ${CORE} (port ${PORT})
After=network.target
[Service]
Environment=PATH=$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=NEURON_MODEL_DIR=/opt/neuron-model
Environment=NEURON_MAX_SEQ_LEN=256
Environment=NEURON_FIXED_BATCH=8
Environment=NEURON_RT_VISIBLE_CORES=${CORE}
WorkingDirectory=/opt/srv
ExecStart=$VENV/bin/uvicorn neuron_server:app --host 127.0.0.1 --port ${PORT} --workers 1
Restart=on-failure
StandardOutput=journal
StandardError=journal
[Install]
WantedBy=multi-user.target
UNIT
done
systemctl daemon-reload

log "deps installed (NUM_CORES=$NUM_CORES); awaiting model files via SCP (total $(el))"
echo "ok" > "$DEPS_SENTINEL"
exit 0
