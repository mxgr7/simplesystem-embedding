#!/usr/bin/env bash
# Launch ./bench.sh for every LADDER instance type concurrently.
# Each runs in its own process, writes to its own log + results CSV
# (avoiding concurrent-write races), then we merge at the end.
#
# Usage:  ./bench_parallel.sh           # full LADDER
#         ./bench_parallel.sh g6.xlarge g5.xlarge  # subset
#
# Wall-clock = max(per-instance time), typically ~40 min vs ~4h sequential.
# AWS quota requirement: ~64 G-family vCPU + ~28 standard vCPU when running
# the full LADDER.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- log helper (same format as bench.sh) ---
RUN_T0=$(date +%s)
log() { printf '[%s | +%4ds | parallel] %s\n' \
         "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(( $(date +%s) - RUN_T0 ))" "$*" >&2; }

# Default LADDER mirrors bench.sh (kept in sync manually).
DEFAULT_LADDER=(c7i.4xlarge g4dn.xlarge g6.xlarge g5.xlarge g6e.xlarge g6.12xlarge)
if [ "$#" -gt 0 ]; then
  LADDER=("$@")
else
  LADDER=("${DEFAULT_LADDER[@]}")
fi
log "launching ${#LADDER[@]} parallel bench.sh runs for: ${LADDER[*]}"

# --- per-instance launch ---
PIDS=()
for inst in "${LADDER[@]}"; do
  safe=${inst//./_}
  LOG="${HERE}/parallel/sweep.${safe}.log"
  RES="${HERE}/parallel/results.${safe}.csv"
  MAT="${HERE}/parallel/matrix.${safe}.csv"
  mkdir -p "${HERE}/parallel"
  # bench.sh accepts a single instance type as positional arg. We override
  # RESULTS_CSV/MATRIX_CSV via env so each instance writes to its own files.
  log "  starting $inst -> log=$(basename "$LOG") results=$(basename "$RES")"
  BENCH_RESULTS_CSV="$RES" BENCH_MATRIX_CSV="$MAT" \
    ./bench.sh "$inst" > "$LOG" 2>&1 &
  PIDS+=($!)
  # Tiny stagger so pulumi-init for stacks doesn't race on the same lock file
  # in the local backend during the first few seconds.
  sleep 3
done

log "all ${#PIDS[@]} runs in-flight (pids: ${PIDS[*]}). Waiting for completion..."

# --- wait for all + collect exit codes ---
FAIL=0
for i in "${!PIDS[@]}"; do
  inst="${LADDER[i]}"
  pid="${PIDS[i]}"
  if wait "$pid"; then
    log "  $inst: completed cleanly"
  else
    rc=$?
    log "  $inst: FAILED with exit $rc"
    FAIL=$((FAIL + 1))
  fi
done

# --- merge per-instance CSVs into the final results.csv + results_matrix.csv ---
log "merging per-instance CSVs"
{
  # header from the first file that has one
  for f in "${HERE}/parallel/"results.*.csv; do
    [ -f "$f" ] || continue
    head -1 "$f"; break
  done
  for f in "${HERE}/parallel/"results.*.csv; do
    [ -f "$f" ] || continue
    tail -n +2 "$f"
  done
} > "${HERE}/results.csv"
{
  for f in "${HERE}/parallel/"matrix.*.csv; do
    [ -f "$f" ] || continue
    head -1 "$f"; break
  done
  for f in "${HERE}/parallel/"matrix.*.csv; do
    [ -f "$f" ] || continue
    tail -n +2 "$f"
  done
} > "${HERE}/results_matrix.csv"
log "merged: $(wc -l < "${HERE}/results.csv") summary rows, $(wc -l < "${HERE}/results_matrix.csv") matrix rows"

if [ "$FAIL" -gt 0 ]; then
  log "$FAIL instance(s) failed — check parallel/sweep.*.log for details"
  exit 1
fi
log "parallel sweep complete; total wall-clock: $(( $(date +%s) - RUN_T0 ))s"
