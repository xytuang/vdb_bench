#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_monitors.sh  —  run from node0
#
# Orchestrates the full benchmark across 5 rounds:
#   1. Take segment snapshot (pre-insert)
#   2. Insert 10M vectors (calls insert_and_search.py --single-round N)
#   3. Launch per-node monitors via SSH
#   4. Run vectordbbench
#   5. Take segment snapshot (post-benchmark)
#   6. Collect all logs from remote nodes back to node0
#   7. Repeat
#
# Prerequisites:
#   - SSH key-based access from node0 to node2/node3/node4/node5
#   - monitor_node.sh present locally (will be scp'd to each node)
#   - Python env with pymilvus on node0
#
# Usage:
#   chmod +x run_monitors.sh
#   ./run_monitors.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Node aliases (from /etc/hosts) ────────────────────────────────────────────
QUERYNODE="node2"
STREAMINGNODE="node3"
DATANODE="node4"
MINIO="node5"

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR="/mydata/vdb_bench/distributed_setup/results"
MONITOR_SCRIPT_LOCAL="$(dirname "$0")/monitor_node.sh"
INSERT_SCRIPT="/mydata/vdb_bench/distributed_setup/insert_and_search.py"
SEGMENT_SCRIPT="/mydata/vdb_bench/distributed_setup/segment_snapshot.py"
VDBBENCH_RESULTS_DIR="/mydata/vdb_bench/bench/lib/python3.12/site-packages/vectordb_bench/results/Milvus"
PYTHON="/mydata/vdb_bench/bench/bin/python3"

# ── Benchmark settings ────────────────────────────────────────────────────────
NUM_ROUNDS=5
MONITOR_INTERVAL=10      # seconds between samples in monitor_node.sh
VDBBENCH_DURATION=300    # seconds — must match your vectordbbench search duration

# ── Helpers ───────────────────────────────────────────────────────────────────
log()       { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
separator() { echo ""; echo "══════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════"; }

deploy_monitor_script() {
    log "Deploying monitor_node.sh to all nodes..."
    for node in $QUERYNODE $STREAMINGNODE $DATANODE $MINIO; do
        scp -q "$MONITOR_SCRIPT_LOCAL" "${node}:~/monitor_node.sh"
        ssh "$node" "chmod +x ~/monitor_node.sh"
        log "  ok: $node"
    done
}

start_monitor() {
    local node="$1" role="$2" round="$3" duration="$4"
    log "  Starting monitor: ${node} (${role})"
    ssh "$node" \
        "nohup ~/monitor_node.sh ${role} ${round} ${MONITOR_INTERVAL} ${duration} \
         > /tmp/monitor_${role}_round${round}_stdout.log 2>&1 &"
}

collect_logs() {
    local round="$1"
    local round_dir="${RESULTS_DIR}/round${round}_logs"
    mkdir -p "$round_dir"
    log "Collecting logs for round ${round} → ${round_dir}/"

    for entry in \
        "${QUERYNODE}:querynode" \
        "${STREAMINGNODE}:streamingnode" \
        "${DATANODE}:datanode" \
        "${MINIO}:minio"
    do
        local node="${entry%%:*}" role="${entry##*:}"
        local src="${node}:/tmp/monitor_${role}_round${round}.log"
        local dst="${round_dir}/monitor_${role}_round${round}.log"
        scp -q "$src" "$dst" \
            && log "  ok: ${src}" \
            || log "  WARN: failed to collect ${src}"
    done

    # Pull segment snapshots into round dir too
    cp "${RESULTS_DIR}/segment_logs/segments_round${round}_pre.log"  "$round_dir/" 2>/dev/null || true
    cp "${RESULTS_DIR}/segment_logs/segments_round${round}_post.log" "$round_dir/" 2>/dev/null || true
}

network_latency_check() {
    separator "NETWORK LATENCY CHECK"
    for pair in \
        "${QUERYNODE}:${STREAMINGNODE}" \
        "${QUERYNODE}:${MINIO}" \
        "${STREAMINGNODE}:${DATANODE}"
    do
        local src="${pair%%:*}" dst="${pair##*:}"
        local result
        result=$(ssh "$src" "ping -c 5 -q ${dst} 2>/dev/null | tail -1" 2>/dev/null || echo "unreachable")
        log "  ${src} -> ${dst}: ${result}"
    done
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
separator "PRE-FLIGHT"
mkdir -p "${RESULTS_DIR}/segment_logs"
deploy_monitor_script
network_latency_check

# ── Round loop ────────────────────────────────────────────────────────────────
for ROUND in $(seq 1 $NUM_ROUNDS); do
    separator "ROUND ${ROUND} / ${NUM_ROUNDS}"
    CUMULATIVE_M=$(( ROUND * 10 ))

    # 1. Insert 10M vectors
    log "[1/5] Inserting 10M vectors..."
    $PYTHON "$INSERT_SCRIPT" --single-round "$ROUND"

    # 2. Segment snapshot — log lag as diagnostic, do not wait
    log "[2/5] Segment snapshot (pre-benchmark)..."
    $PYTHON "$SEGMENT_SCRIPT" --round "$ROUND" --phase pre

    # 3. Start monitors on all nodes
    log "[3/5] Starting monitors..."
    start_monitor "$QUERYNODE"     "querynode"     "$ROUND" "$VDBBENCH_DURATION"
    start_monitor "$STREAMINGNODE" "streamingnode" "$ROUND" "$VDBBENCH_DURATION"
    start_monitor "$DATANODE"      "datanode"      "$ROUND" "$VDBBENCH_DURATION"
    start_monitor "$MINIO"         "minio"         "$ROUND" "$VDBBENCH_DURATION"
    sleep 3   # let monitors initialize before benchmark starts

    # 4. Run vectordbbench — detect new result JSON by diffing directory
    log "[4/5] Running vectordbbench..."
    # Use find instead of ls glob so empty dir doesn't cause set -e to abort
    BEFORE_FILES=$(find "${VDBBENCH_RESULTS_DIR}" -name "*.json" 2>/dev/null | sort)

    vectordbbench milvusdiskann --config-file "../config.yaml"

    AFTER_FILES=$(find "${VDBBENCH_RESULTS_DIR}" -name "*.json" 2>/dev/null | sort)
    NEW_JSON=$(comm -13 <(echo "$BEFORE_FILES") <(echo "$AFTER_FILES") | head -1 || true)

    if [ -n "$NEW_JSON" ]; then
        DEST="${RESULTS_DIR}/round${ROUND}_${CUMULATIVE_M}M_$(basename "$NEW_JSON")"
        mv "$NEW_JSON" "$DEST"
        log "  Benchmark result -> ${DEST}"
    else
        log "  WARN: no new vectordbbench JSON detected"
    fi

    # 5. Segment snapshot after benchmark
    log "[5/5] Segment snapshot (post-benchmark)..."
    $PYTHON "$SEGMENT_SCRIPT" --round "$ROUND" --phase post

    # Collect logs — monitors ran concurrently with vectordbbench and are
    #    already done (they run for VDBBENCH_DURATION seconds). Give a small
    #    buffer in case of clock skew then collect.
    log "[+] Collecting monitor logs..."
    sleep 15
    collect_logs "$ROUND"

    separator "ROUND ${ROUND} DONE — ${CUMULATIVE_M}M vectors total"
done

# ── Final summary ─────────────────────────────────────────────────────────────
separator "ALL ROUNDS COMPLETE"
log "Results in: ${RESULTS_DIR}"
echo ""
find "$RESULTS_DIR" -type f | sort | sed "s|${RESULTS_DIR}/||" | while read -r f; do
    log "  $f"
done
