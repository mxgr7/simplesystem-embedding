#!/usr/bin/env bash
# Orchestrates one TEI throughput benchmark on an AWS instance type:
#   1. pulumi up         — provisions tei server + co-located load-gen box
#   2. SSH-poll          — waits for TEI to report healthy on :3000
#   3. run loadgen       — over SSH to the co-located c7i.large
#   4. pulumi destroy    — always runs (trap), even on failure
#
# Usage:
#   ./bench.sh g6.xlarge                 # one instance type
#   ./bench.sh --sweep                   # the full ladder
#   ./bench.sh --sweep g6.xlarge g5.xlarge  # subset of the ladder
#
# Reads /workspace/.env for AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# HF_TOKEN. Writes results to ./results.csv (one row per instance).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --------------------------------------------------------------------------- #
# Logging helpers — ISO-8601 UTC timestamps + a global start-of-run epoch so
# every line shows wall-clock and elapsed-since-launch. Everything goes to
# stderr so command substitutions stay clean.
# --------------------------------------------------------------------------- #
RUN_T0=$(date +%s)
log()  { printf '[%s | +%4ds] %s\n' \
           "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( $(date +%s) - RUN_T0 ))" "$*" >&2; }
warn() { log "WARN: $*"; }
die()  { log "FATAL: $*"; exit 1; }

# --------------------------------------------------------------------------- #
# Load env (AWS creds + HF token)
# --------------------------------------------------------------------------- #
log "loading env from /workspace/.env"
if [ -f /workspace/.env ]; then
  set -a; source /workspace/.env; set +a
else
  warn "/workspace/.env not found — relying on inherited env"
fi
: "${AWS_ACCESS_KEY_ID:?missing AWS_ACCESS_KEY_ID (expected in /workspace/.env)}"
: "${AWS_SECRET_ACCESS_KEY:?missing AWS_SECRET_ACCESS_KEY (expected in /workspace/.env)}"
: "${HF_TOKEN:?missing HF_TOKEN (expected in /workspace/.env)}"
: "${AWS_REGION:=eu-central-1}"
export AWS_REGION
log "env ok: aws_region=$AWS_REGION  aws_key_id=${AWS_ACCESS_KEY_ID:0:6}***  hf_token=${HF_TOKEN:0:6}***"

# Local-file Pulumi backend so the operator doesn't need a Pulumi Cloud account.
# Pulumi won't auto-create the backend directory — must mkdir up-front.
mkdir -p "${HERE}/.pulumi-state"
export PULUMI_BACKEND_URL="file://${HERE}/.pulumi-state"
export PULUMI_CONFIG_PASSPHRASE="${PULUMI_CONFIG_PASSPHRASE:-bench}"
log "pulumi backend: $PULUMI_BACKEND_URL"

LADDER=(c7i.4xlarge g4dn.xlarge g6.xlarge g5.xlarge g6e.xlarge g6.12xlarge inf1.xlarge inf1.2xlarge inf2.xlarge)

# Pre-compiled Neuron model artifacts (SCP'd to inf1/inf2 boxes after their
# user-data finishes installing runtime deps). Built by compile.sh — one .pt
# per chip generation, since the Neuron compiler is target-specific. The
# .neff is embedded inside the .pt by neuron.trace, no separate .neff file.
INF_MODEL_DIR="/tei-models"
INF_ST_DIR="useful-cub-58-st"   # tokenizer + 1_Pooling/2_Dense/3_Normalize
# Map: instance-type -> artifact basename (without .pt). inf1.xlarge and
# inf1.2xlarge share the chip so they share the artifact.
declare -A INF_PT_BY_INSTANCE=(
  [inf1.xlarge]="useful-cub-58-st.neuron-inf1-bs8-sl256"
  [inf1.2xlarge]="useful-cub-58-st.neuron-inf1-bs8-sl256"
  [inf2.xlarge]="useful-cub-58-st.neuron-inf2-bs8-sl256"
)

# --------------------------------------------------------------------------- #
# Arg parsing
# --------------------------------------------------------------------------- #
INSTANCES=()
if [ "$#" -eq 0 ]; then
  echo "usage: $0 <instance-type> | --sweep [instance...]" >&2
  exit 2
fi
if [ "$1" = "--sweep" ]; then
  shift
  if [ "$#" -gt 0 ]; then INSTANCES=("$@"); else INSTANCES=("${LADDER[@]}"); fi
else
  INSTANCES=("$1")
fi
log "instance plan: ${INSTANCES[*]}"

# --------------------------------------------------------------------------- #
# SSH key — one throwaway pair for the whole sweep
# --------------------------------------------------------------------------- #
KEY_FILE="${HERE}/bench-key"
if [ ! -f "$KEY_FILE" ]; then
  log "generating throwaway ed25519 keypair at $KEY_FILE"
  ssh-keygen -t ed25519 -N "" -f "$KEY_FILE" -C "aws-embedding-bench" >/dev/null
else
  log "reusing existing keypair at $KEY_FILE"
fi
PUB_KEY_MATERIAL="$(cat "${KEY_FILE}.pub")"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=10 -o LogLevel=ERROR -i "$KEY_FILE")

# --------------------------------------------------------------------------- #
# Pulumi CLI sanity check. The CLI auto-creates the venv declared in
# Pulumi.yaml (virtualenv: venv) on first `up`, so no manual setup needed.
# --------------------------------------------------------------------------- #
if ! command -v pulumi >/dev/null 2>&1; then
  die "pulumi CLI not found; install with: curl -fsSL https://get.pulumi.com | sh"
fi
if ! command -v uv >/dev/null 2>&1; then
  die "uv not found; install with: curl -fsSL https://astral.sh/uv/install.sh | sh"
fi
log "pulumi CLI: $(pulumi version)"
log "uv:         $(uv --version)"

# --------------------------------------------------------------------------- #
# Quota pre-flight — warn (don't block) if missing.
# --------------------------------------------------------------------------- #
quota_check() {
  local name="$1" code="$2" required="$3"
  local q
  q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code "$code" \
      --region "$AWS_REGION" --query 'Quota.Value' --output text 2>/dev/null || echo "?")
  log "quota check: $name ($code) = $q (need >=$required)"
  if [ "$q" != "?" ] && python3 -c "import sys; sys.exit(0 if float('$q') < float('$required') else 1)"; then
    warn "quota looks insufficient; pulumi up may fail"
  fi
}
quota_check "Running On-Demand G and VT instances" "L-DB2E81BA" 48
quota_check "Running On-Demand Inf instances" "L-1945791B" 4

# --------------------------------------------------------------------------- #
# Per-instance benchmark loop
# --------------------------------------------------------------------------- #
run_one() {
  local instance="$1"
  local stack="bench-${instance//./-}"
  local instance_t0=$(date +%s)
  echo >&2
  log "================================================================="
  log "  $instance  ->  stack $stack"
  log "================================================================="

  pushd "$HERE" >/dev/null
  log "pulumi stack init/select $stack"
  # Every pulumi command below uses --stack explicitly. The local-file
  # backend writes the "currently selected stack" to a shared workspace
  # file; if two parallel processes both call `stack select`, the second
  # write wins and the first process's subsequent commands silently
  # operate on the wrong stack. --stack everywhere sidesteps that race.
  pulumi stack init "$stack" 2>/dev/null || true
  log "writing pulumi config (region, instanceType, keypair, hfToken)"
  pulumi config set --stack "$stack" aws:region "$AWS_REGION"
  pulumi config set --stack "$stack" bench:instanceType "$instance"
  pulumi config set --stack "$stack" bench:publicKeyMaterial "$PUB_KEY_MATERIAL"
  pulumi config set --stack "$stack" --plaintext bench:hfToken "$HF_TOKEN"

  # Always tear down on exit (incl. failure) so we don't leak GPU instances.
  # `stack` is a `local` in this function; once the function exits the var
  # is gone, so the trap captures it explicitly via TRAP_STACK (global).
  TRAP_STACK="$stack"
  # The INT/TERM handler exits explicitly. Without `exit`, bash runs the
  # handler body and then *resumes* the next statement — meaning a Ctrl-C
  # mid-bench would destroy the stack and then keep polling /health for
  # 25 minutes against the now-deleted server. EXIT runs naturally; no
  # exit needed there.
  trap '
    log "trap fired — tearing down ${TRAP_STACK:-<unset>} ..."
    [ -n "${TRAP_STACK:-}" ] && pulumi destroy --yes --skip-preview --stack "$TRAP_STACK" || true
    [ -n "${TRAP_STACK:-}" ] && pulumi stack rm "$TRAP_STACK" --yes 2>/dev/null || true
  ' EXIT
  trap '
    log "INT/TERM caught — tearing down ${TRAP_STACK:-<unset>} and exiting"
    [ -n "${TRAP_STACK:-}" ] && pulumi destroy --yes --skip-preview --stack "$TRAP_STACK" || true
    [ -n "${TRAP_STACK:-}" ] && pulumi stack rm "$TRAP_STACK" --yes 2>/dev/null || true
    exit 130
  ' INT TERM

  log "pulumi up — provisioning $instance + loadgen c7i.large"
  pulumi up --stack "$stack" --yes --skip-preview

  local server_ip loadgen_ip server_pub
  server_ip=$(pulumi stack output --stack "$stack" server_private_ip)
  loadgen_ip=$(pulumi stack output --stack "$stack" loadgen_public_ip)
  server_pub=$(pulumi stack output --stack "$stack" server_public_ip)
  log "pulumi outputs:  server_private=$server_ip  server_public=$server_pub  loadgen_public=$loadgen_ip"

  # --- inf1/inf2 only: SCP pre-compiled model + start the neuron-tei services --- #
  # The userdata script only installs deps and stages the systemd units; it
  # leaves /opt/neuron-model empty and writes /var/run/inf-deps-ready as the
  # hand-off signal. We poll that, copy the artifacts, then start the units.
  # Number of services depends on chip generation (inf1=4 cores, inf2.xlarge=2).
  case "$instance" in
    inf1.*) local _inf_num_cores=4 ;;
    inf2.*) local _inf_num_cores=2 ;;
    *)      local _inf_num_cores=0 ;;
  esac
  if [ "$_inf_num_cores" -gt 0 ]; then
    log "inf: waiting for /var/run/inf-deps-ready (deps install, ~3-5min)"
    local inf_wait_t0=$(date +%s)
    local inf_attempt=0
    local inf_deps_ready=""
    for _ in $(seq 1 60); do
      inf_attempt=$((inf_attempt + 1))
      if ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
           "test -f /var/run/inf-deps-ready" 2>/dev/null; then
        inf_deps_ready=$(ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
                           "cat /var/run/inf-deps-ready" 2>/dev/null || echo "")
        log "inf: deps-ready after ${inf_attempt} probe(s) ($(( $(date +%s) - inf_wait_t0 ))s): $inf_deps_ready"
        break
      fi
      [ $((inf_attempt % 6)) -eq 0 ] && \
        log "  ... inf deps still installing (attempt $inf_attempt, $(( $(date +%s) - inf_wait_t0 ))s)"
      sleep 10
    done

    if [ -z "$inf_deps_ready" ] || [[ "$inf_deps_ready" == fatal:* ]]; then
      warn "inf: deps install failed or never completed (sentinel: '$inf_deps_ready')"
      log "fetching user-data + install logs for postmortem"
      {
        echo "=== /var/run/inf-deps-ready ==="; echo "$inf_deps_ready"
        echo "=== /var/log/inferentia-userdata.log (tail -100) ==="
        ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
          "sudo tail -100 /var/log/inferentia-userdata.log 2>/dev/null" || true
        echo "=== /var/log/neuron-install.log (tail -60) ==="
        ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
          "sudo tail -60 /var/log/neuron-install.log 2>/dev/null" || true
      } > "${HERE}/last_inf_install_failed.log" 2>&1 || true
      # Fall through — the /health poll below will time out and the row
      # gets recorded as "never healthy".
    else
      local pt_base="${INF_PT_BY_INSTANCE[$instance]:-}"
      [ -z "$pt_base" ] && die "no artifact mapped for $instance in INF_PT_BY_INSTANCE"
      local pt_src="$INF_MODEL_DIR/${pt_base}.pt"
      local st_src="$INF_MODEL_DIR/$INF_ST_DIR"
      if [ ! -f "$pt_src" ] || [ ! -d "$st_src" ]; then
        warn "inf: model artifacts missing on operator box ($pt_src or $st_src) — skipping upload"
      else
        log "inf: SCPing $pt_src ($(du -h "$pt_src" | cut -f1)) -> $server_pub:/opt/neuron-model/model.pt"
        scp "${SSH_OPTS[@]}" "$pt_src" "ubuntu@$server_pub:/opt/neuron-model/model.pt"
        log "inf: SCPing tokenizer + ST structural files from $st_src/"
        scp -r "${SSH_OPTS[@]}" \
            "$st_src/tokenizer.json" "$st_src/tokenizer_config.json" \
            "$st_src/config.json" "$st_src/sentence_bert_config.json" \
            "$st_src/modules.json" \
            "$st_src/1_Pooling" "$st_src/2_Dense" "$st_src/3_Normalize" \
            "ubuntu@$server_pub:/opt/neuron-model/"
        log "inf: SCPing neuron_server.py to /opt/srv/"
        scp "${SSH_OPTS[@]}" "${HERE}/neuron_server.py" \
            "ubuntu@$server_pub:/opt/srv/neuron_server.py"
        # Build "neuron-tei-0.service neuron-tei-1.service ..." list dynamically.
        local _svc_list=""
        for c in $(seq 0 $((_inf_num_cores - 1))); do
          _svc_list+="neuron-tei-${c}.service "
        done
        log "inf: starting $_svc_list (one per NeuronCore)"
        ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
            "sudo systemctl start $_svc_list" || \
            warn "systemctl start returned non-zero (services may still come up)"
      fi
    fi
  fi
  # --- end inf-only block --- #

  # ----- Wait for load-gen box to finish its user-data ----- #
  log "waiting for loadgen user-data (polls every 10s, timeout 10min) ..."
  local wait_t0=$(date +%s)
  local attempt=0
  local loadgen_ready=""
  for _ in $(seq 1 60); do
    attempt=$((attempt + 1))
    if ssh "${SSH_OPTS[@]}" "ubuntu@$loadgen_ip" \
         "test -f /var/run/loadgen-ready" 2>/dev/null; then
      loadgen_ready=yes
      log "loadgen ready after ${attempt} attempt(s) ($(( $(date +%s) - wait_t0 ))s)"
      break
    fi
    [ $((attempt % 6)) -eq 0 ] && \
      log "  ... loadgen still booting (attempt $attempt, $(( $(date +%s) - wait_t0 ))s elapsed)"
    sleep 10
  done
  [ -z "$loadgen_ready" ] && warn "loadgen never reported ready — SSH probes likely to fail"

  # ----- Wait for the TEI box to be healthy (or compile_failed) ----- #
  log "waiting for TEI server (polls every 10s, timeout 25min for model pull/compile) ..."
  wait_t0=$(date +%s)
  attempt=0
  local ready=""
  for _ in $(seq 1 150); do
    attempt=$((attempt + 1))
    if ssh "${SSH_OPTS[@]}" "ubuntu@$loadgen_ip" \
         "curl -sf http://$server_ip:3000/health" >/dev/null 2>&1; then
      ready="healthy"
      log "TEI healthy after ${attempt} attempt(s) ($(( $(date +%s) - wait_t0 ))s)"
      break
    fi
    if [ $((attempt % 6)) -eq 0 ]; then
      log "  ... still waiting on TEI (attempt $attempt, $(( $(date +%s) - wait_t0 ))s elapsed); tailing server logs:"
      case "$instance" in
        inf1.*|inf2.*)
          ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
            "sudo tail -5 /var/log/inferentia-userdata.log 2>/dev/null; sudo journalctl -u 'neuron-tei-*' -n 15 --no-pager 2>/dev/null" >&2 || true ;;
        *)
          ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
            "sudo tail -5 /var/log/tei-userdata.log 2>/dev/null" >&2 || true ;;
      esac
    fi
    sleep 10
  done

  # Per-process CSV destinations. Env vars let bench_parallel.sh point each
  # parallel run at its own files so they don't race on the same outputs.
  local RESULTS_DST="${BENCH_RESULTS_CSV:-${HERE}/results.csv}"
  local MATRIX_DST="${BENCH_MATRIX_CSV:-${HERE}/results_matrix.csv}"
  # Per-process tmp paths — /tmp/bench-row.csv would race across parallel runs.
  local TMP_ROW="/tmp/bench-row.$instance.$$.csv"
  local TMP_MAT="/tmp/bench-matrix.$instance.$$.csv"

  # Placeholder rows on failure use the v2 schema:
  # instance,max_batch_tokens,batch,conc,target_tokens,emb/s,req/s,p50,p99,err,dur,warm,ts
  local FAIL_ROW="$instance,0,0,0,256,0.0,0.0,nan,nan,1,0,0,$(date +%s)"

  if [ "$ready" != "healthy" ]; then
    warn "$instance: TEI never reported healthy after 25min — recording failure row"
    log "fetching last 50 lines of server logs for postmortem:"
    case "$instance" in
      inf1.*|inf2.*)
        ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
          "sudo tail -50 /var/log/inferentia-userdata.log 2>/dev/null; sudo journalctl -u 'neuron-tei-*' -n 80 --no-pager 2>/dev/null" >&2 || true ;;
      *)
        ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
          "sudo tail -50 /var/log/tei-userdata.log 2>/dev/null" >&2 || true ;;
    esac
    echo "$FAIL_ROW" >> "$RESULTS_DST"
  else
    # Pull the phase-1 discovered max-batch-tokens (written by tei_userdata.sh).
    # For inf1/inf2 there's no probe — the compiled .pt fixes (batch, seq_len),
    # so record FIXED_BATCH * MAX_SEQ_LEN as max_batch_tokens for the CSV.
    local max_batch_tokens=0
    case "$instance" in
      inf1.*|inf2.*)
        max_batch_tokens=2048   # FIXED_BATCH=8 * MAX_SEQ_LEN=256
        ;;
      *)
        max_batch_tokens=$(ssh "${SSH_OPTS[@]}" "ubuntu@$server_pub" \
          "sudo cat /var/run/tei-max-batch-tokens 2>/dev/null" 2>/dev/null || echo 0)
        [ -z "$max_batch_tokens" ] && max_batch_tokens=0
        ;;
    esac
    log "max_batch_tokens=$max_batch_tokens on $instance"

    log "scp'ing loadgen.py to $loadgen_ip"
    scp "${SSH_OPTS[@]}" "${HERE}/loadgen.py" \
        "ubuntu@$loadgen_ip:/opt/loadgen/loadgen.py"
    log "running loadgen matrix on $loadgen_ip against http://$server_ip:3000 ..."
    ssh -tt "${SSH_OPTS[@]}" "ubuntu@$loadgen_ip" \
      "python3 -u /opt/loadgen/loadgen.py \
         --url http://$server_ip:3000 \
         --instance-type $instance \
         --max-batch-tokens $max_batch_tokens \
         --results-csv /home/ubuntu/results.csv \
         --matrix-csv /home/ubuntu/results_matrix.csv" 2>&1 \
      | while IFS= read -r line; do log "[loadgen] $line"; done

    log "fetching results.csv + results_matrix.csv back to operator machine"
    scp "${SSH_OPTS[@]}" "ubuntu@$loadgen_ip:/home/ubuntu/results.csv"        "$TMP_ROW"
    scp "${SSH_OPTS[@]}" "ubuntu@$loadgen_ip:/home/ubuntu/results_matrix.csv" "$TMP_MAT"
    for pair in "$RESULTS_DST:$TMP_ROW" "$MATRIX_DST:$TMP_MAT"; do
      DST="${pair%%:*}"
      SRC_FILE="${pair##*:}"
      if [ ! -f "$DST" ]; then
        cp "$SRC_FILE" "$DST"
      else
        tail -n +2 "$SRC_FILE" >> "$DST"
      fi
    done
    rm -f "$TMP_ROW" "$TMP_MAT"
    log "appended summary row to $RESULTS_DST:"
    tail -1 "$RESULTS_DST" | sed 's/^/    /' >&2
  fi

  # Explicit teardown (the trap runs too, but doing it here lets the loop
  # continue cleanly).
  log "tearing down $stack (instance wall-clock: $(( $(date +%s) - instance_t0 ))s)"
  pulumi destroy --yes --skip-preview --stack "$stack"
  pulumi stack rm --stack "$stack" --yes
  trap - EXIT INT TERM
  log "teardown complete for $instance"
  popd >/dev/null
}

for instance in "${INSTANCES[@]}"; do
  run_one "$instance"
done

# Sanity check: no bench-tagged instances left running.
echo >&2
log "final leak check ..."
left=$(aws ec2 describe-instances --region "$AWS_REGION" \
   --filters "Name=tag:Project,Values=aws-embedding-bench" \
             "Name=instance-state-name,Values=running,pending" \
   --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$left" ]; then
  warn "bench instances still running: $left"
  exit 1
fi
log "all clean. total wall-clock: $(( $(date +%s) - RUN_T0 ))s"
echo >&2
log "results:"
column -ts, "${HERE}/results.csv" 2>/dev/null || cat "${HERE}/results.csv"
