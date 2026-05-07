#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# monitor_node.sh  —  per-node metric collector
#
# Usage (called automatically by run_monitors.sh on node0):
#   ./monitor_node.sh <role> <round> <interval_seconds> <duration_seconds>
#
# Logs to: /tmp/monitor_<role>_round<round>.log
# ─────────────────────────────────────────────────────────────────────────────

ROLE="${1:-unknown}"
ROUND="${2:-0}"
INTERVAL="${3:-2}"
DURATION="${4:-600}"

LOGFILE="/tmp/monitor_${ROLE}_round${ROUND}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

separator() {
    echo "" >> "$LOGFILE"
    echo "──────────────────────────────────────────" >> "$LOGFILE"
    echo "$1" >> "$LOGFILE"
    echo "──────────────────────────────────────────" >> "$LOGFILE"
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "" > "$LOGFILE"   # truncate/create
separator "NODE MONITOR — role=${ROLE}  round=${ROUND}  $(hostname)  $(date)"
log "interval=${INTERVAL}s  duration=${DURATION}s"
log "Kernel: $(uname -r)"
log "Uptime: $(uptime)"

# ── One-time static info ──────────────────────────────────────────────────────
separator "STATIC: CPU"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|MHz" >> "$LOGFILE"

separator "STATIC: MEMORY"
free -h >> "$LOGFILE"

separator "STATIC: DISK"
df -h >> "$LOGFILE"
lsblk >> "$LOGFILE"

separator "STATIC: NETWORK INTERFACES"
ip -br addr >> "$LOGFILE"

# ── CPU governor ──────────────────────────────────────────────────────────────
separator "CPU SCALING GOVERNOR"
for cpu in /sys/devices/system/cpu/cpu{0,1,2,3}/cpufreq/scaling_governor; do
    [ -f "$cpu" ] && echo "$cpu: $(cat $cpu)" >> "$LOGFILE"
done

# ── Role-specific one-time checks ─────────────────────────────────────────────
case "$ROLE" in
    querynode)
        separator "DISKANN LOCAL CACHE"
        CACHE_PATH="/mydata/milvus/cache/milvus/cache/cache/5/local_chunk"
        if [ -d "$CACHE_PATH" ]; then
            du -sh "$CACHE_PATH" >> "$LOGFILE" 2>&1
            ls -lh "$CACHE_PATH" | head -30 >> "$LOGFILE" 2>&1
        else
            echo "Cache path not found: $CACHE_PATH" >> "$LOGFILE"
        fi
        ;;
    streamingnode)
        separator "GROWING MMAP"
        MMAP_PATH="/mydata/milvus/cache/milvus/cache/cache/5/growing_mmap"
        if [ -d "$MMAP_PATH" ]; then
            du -sh "$MMAP_PATH" >> "$LOGFILE" 2>&1
            ls -lh "$MMAP_PATH" | head -30 >> "$LOGFILE" 2>&1
        else
            echo "mmap path not found: $MMAP_PATH" >> "$LOGFILE"
        fi
        ;;
    datanode)
        separator "MILVUS DATA DIR"
        du -sh /mydata/milvus/ >> "$LOGFILE" 2>&1
        ;;
    minio)
        separator "MINIO DISK"
        df -h /mydata >> "$LOGFILE" 2>&1
        du -sh /mydata/minio/ >> "$LOGFILE" 2>&1
        ;;
esac

# ── Detect primary disk device for iostat ─────────────────────────────────────
# Find the device backing /mydata (strip partition number)
NVME_DEVS=$(ls /dev/nvme*n1 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')
[ -z "$NVME_DEVS" ] && NVME_DEVS="nvme0n1"

# ── Time-series loop ──────────────────────────────────────────────────────────
separator "TIME-SERIES METRICS (every ${INTERVAL}s)"

END_TIME=$(( $(date +%s) + DURATION ))
SAMPLE=0

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    SAMPLE=$(( SAMPLE + 1 ))
    TS="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "" >> "$LOGFILE"
    echo "=== SAMPLE ${SAMPLE} @ ${TS} ===" >> "$LOGFILE"

    # CPU usage
    echo "--- CPU (top snapshot) ---" >> "$LOGFILE"
    top -b -n 1 | head -5 >> "$LOGFILE"

    # Per-core usage (non-blocking — reads /proc/stat delta would need two reads;
    # use mpstat with 1s sample only if interval allows it)
    echo "--- CPU utilization % ---" >> "$LOGFILE"
    mpstat 1 1 2>/dev/null | grep -v "^$" | tail -3 >> "$LOGFILE" || \
        vmstat 1 1 >> "$LOGFILE"

    # Memory
    echo "--- MEMORY ---" >> "$LOGFILE"
    free -h >> "$LOGFILE"

    # Disk I/O (non-blocking snapshot since last sample)
    echo "--- DISK I/O (${NVME_DEVS}) ---" >> "$LOGFILE"
    iostat -x $NVME_DEVS 2>/dev/null | grep -v "^$" | grep -v "^Linux" >> "$LOGFILE" || \
        iostat -x 2>/dev/null | grep -v "^$" | grep -v "^Linux" | head -10 >> "$LOGFILE"

    # Role-specific per-sample metrics
    case "$ROLE" in
        querynode)
            echo "--- DISKANN CACHE SIZE ---" >> "$LOGFILE"
            du -sh /mydata/milvus/cache/milvus/cache/cache/5/local_chunk/ 2>/dev/null >> "$LOGFILE"

            echo "--- OPEN FILES (milvus process) ---" >> "$LOGFILE"
            MILVUS_PID=$(pgrep -f "milvus" | head -1)
            if [ -n "$MILVUS_PID" ]; then
                ls /proc/$MILVUS_PID/fd 2>/dev/null | wc -l | \
                    xargs -I{} echo "open file descriptors: {}" >> "$LOGFILE"
            fi
            ;;

        streamingnode)
            echo "--- GROWING MMAP SIZE ---" >> "$LOGFILE"
            du -sh /mydata/milvus/cache/milvus/cache/cache/5/growing_mmap/ 2>/dev/null >> "$LOGFILE"

            echo "--- MEMORY BREAKDOWN (milvus process) ---" >> "$LOGFILE"
            MILVUS_PID=$(pgrep -f "milvus" | head -1)
            if [ -n "$MILVUS_PID" ]; then
                cat /proc/$MILVUS_PID/status 2>/dev/null | \
                    grep -E "VmRSS|VmSize|VmPeak|Threads" >> "$LOGFILE"
            fi
            ;;

        minio)
            echo "--- MINIO DISK USAGE ---" >> "$LOGFILE"
            df -h /mydata >> "$LOGFILE"
            ;;
    esac

    # Network throughput (bytes in/out on primary interface)
    echo "--- NETWORK (rx/tx bytes delta) ---" >> "$LOGFILE"
    PRIMARY_IF=$(ip route | awk '/default/{print $5; exit}')
    if [ -n "$PRIMARY_IF" ]; then
        RX1=$(cat /sys/class/net/$PRIMARY_IF/statistics/rx_bytes 2>/dev/null)
        TX1=$(cat /sys/class/net/$PRIMARY_IF/statistics/tx_bytes 2>/dev/null)
        sleep "$INTERVAL"
        RX2=$(cat /sys/class/net/$PRIMARY_IF/statistics/rx_bytes 2>/dev/null)
        TX2=$(cat /sys/class/net/$PRIMARY_IF/statistics/tx_bytes 2>/dev/null)
        RX_RATE=$(( (RX2 - RX1) / INTERVAL / 1024 ))
        TX_RATE=$(( (TX2 - TX1) / INTERVAL / 1024 ))
        echo "  ${PRIMARY_IF}: RX ${RX_RATE} KB/s  TX ${TX_RATE} KB/s" >> "$LOGFILE"
    else
        sleep "$INTERVAL"
    fi
done

separator "MONITORING COMPLETE — $(date)"
echo "Log written to: $LOGFILE"
