#!/usr/bin/env bash
# EC2 user-data for the Inferentia compile box (inf1 or inf2). Minimal: verify
# the DLAMI venv has torch + the right torch-neuron(x) + transformers
# importable, then signal ready. compile.sh SCPs the source model +
# compile_neuron.py and runs the trace manually after this completes.
#
# compile.sh sed-substitutes __NEURON_VENV_GLOB__ before passing this as
# user-data — inf1 uses /opt/aws_neuron_venv_pytorch_* (no x), inf2 uses
# /opt/aws_neuronx_venv_pytorch_*.

set -uo pipefail
exec > >(tee -a /var/log/compile-userdata.log) 2>&1

log() { printf '[%s | compile-ud] %s\n' "$(date -u +%FT%TZ)" "$*"; }
T0=$(date +%s); el() { echo "$(( $(date +%s) - T0 ))s"; }
DEPS_SENTINEL=/var/run/compile-deps-ready
NEURON_VENV_GLOB='__NEURON_VENV_GLOB__'
log "boot on $(hostname); kernel $(uname -r)"
log "venv glob: $NEURON_VENV_GLOB"

VENV=$(ls -d $NEURON_VENV_GLOB/ 2>/dev/null \
        | grep -Ev 'nxd_(training|inference)/?$' \
        | sort -V | tail -1 | sed 's:/$::')
if [ -z "$VENV" ]; then
  echo "fatal: no venv matched $NEURON_VENV_GLOB (wrong AMI?)" > "$DEPS_SENTINEL"
  exit 1
fi
log "venv: $VENV"
ls -la /dev/neuron* 2>&1 | sed 's/^/    /' || true

# PATH fix: torch_neuronx import-time init shells out to libneuronpjrt-path,
# which lives in $VENV/bin (not on PATH unless the venv is activated).
export PATH="$VENV/bin:$PATH"
INSTALL_LOG=/var/log/compile-install.log

# Pick which torch-neuron package we expect to find: inf2 venv contains
# torch_neuronx (with x); inf1 venv contains torch_neuron (without x).
case "$VENV" in
  *aws_neuronx_venv_pytorch_*) NEURON_IMPORT="torch_neuronx" ;;
  *aws_neuron_venv_pytorch_*)  NEURON_IMPORT="torch_neuron"  ;;
  *) NEURON_IMPORT="torch_neuronx" ;;  # safe default
esac
log "expected neuron package: $NEURON_IMPORT"

log "installing transformers + tokenizers (DLAMI may or may not ship them)"
if ! "$VENV/bin/pip" install --only-binary :all: \
      'transformers>=4.36,<4.44' 'tokenizers>=0.15,<1' \
      >"$INSTALL_LOG" 2>&1; then
  log "transformers install FAILED — tail $INSTALL_LOG:"
  tail -30 "$INSTALL_LOG" | sed 's/^/    /' || true
  { echo "fatal: transformers install failed"; tail -30 "$INSTALL_LOG"; } > "$DEPS_SENTINEL"
  exit 1
fi
log "transformers install done (el $(el))"

log "verifying torch + $NEURON_IMPORT + transformers importable"
if ! "$VENV/bin/python" -c "
import torch
neuron = __import__('$NEURON_IMPORT')
import transformers
print(f'torch={torch.__version__} $NEURON_IMPORT={neuron.__version__} transformers={transformers.__version__}')
" >>"$INSTALL_LOG" 2>&1; then
  log "import FAILED — tail $INSTALL_LOG:"
  tail -30 "$INSTALL_LOG" | sed 's/^/    /' || true
  { echo "fatal: dep import failed"; tail -30 "$INSTALL_LOG"; } > "$DEPS_SENTINEL"
  exit 1
fi
tail -1 "$INSTALL_LOG" | sed 's/^/    /'

# Stage workspace so SCP from compile.sh (as 'ubuntu') doesn't need sudo.
mkdir -p /opt/source-model
chown -R ubuntu:ubuntu /opt/source-model

log "compile-deps ready (total $(el))"
echo "ok" > "$DEPS_SENTINEL"
exit 0
