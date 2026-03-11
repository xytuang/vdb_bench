#!/bin/bash
set -euo pipefail

CONFIG_FILE="config.yaml"
CONCURRENCIES=("1" "2" "4" "8" "16" "32" "64" "128")

# Helper: update num_concurrency in the yaml to a single value, and set
# the search flags appropriately.
# Uses sed to do an in-place replacement of the num_concurrency line.

set_concurrency() {
    local value="$1"
    sed -i "s/^  num_concurrency:.*/  num_concurrency: \"${value}\"/" "$CONFIG_FILE"
}

set_search_mode() {
    local serial="$1"    # true or false
    local concurrent="$2" # true or false
    sed -i "s/^  search_serial:.*/  search_serial: ${serial}/" "$CONFIG_FILE"
    sed -i "s/^  search_concurrent:.*/  search_concurrent: ${concurrent}/" "$CONFIG_FILE"
}

# Helper function to restart Milvus and wait until it is healthy
restart_milvus() {
    echo ">>> Restarting Milvus to clear chunk cache..."
    sudo docker-compose down
    sudo docker-compose up -d

    echo ">>> Waiting for Milvus to become healthy..."
    local retries=0
    local max_retries=20
    until sudo docker-compose ps | grep -q "healthy" || [ $retries -ge $max_retries ]; do
        sleep 5
        retries=$((retries + 1))
        echo "    ...waiting ($retries/$max_retries)"
    done

    if [ $retries -ge $max_retries ]; then
        echo "ERROR: Milvus did not become healthy in time. Aborting."
        exit 1
    fi

    echo ">>> Milvus is up."
}

# Helper function to drop the OS page cache
drop_page_cache() {
    echo ">>> Dropping page cache..."
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    echo ">>> Page cache dropped."
}

echo "========================================"
echo " Phase 1: Serial latency test"
echo "========================================"

restart_milvus
drop_page_cache

set_search_mode true false   # serial=true, concurrent=false
# num_concurrency is irrelevant for serial, but set it cleanly anyway
set_concurrency "1"

echo ">>> Running serial search..."
python3 load_index.py
sleep 30
NUM_PER_BATCH=10000 vectordbbench milvusdiskann --config-file "$CONFIG_FILE"
echo ">>> Serial latency test complete."

echo ""
echo "========================================"
echo " Phase 2: Concurrent throughput sweep"
echo "========================================"

set_search_mode false true   # serial=false, concurrent=true

for concurrency in "${CONCURRENCIES[@]}"; do
    echo ""
    echo "--- Concurrency: ${concurrency} ---"

    restart_milvus
    drop_page_cache
    python3 load_index.py
    sleep 30
    # Patch the yaml so only this single concurrency level is tested
    set_concurrency "$concurrency"

    echo ">>> Running concurrent search at concurrency=${concurrency}..."
    NUM_PER_BATCH=10000 vectordbbench milvusdiskann --config-file "$CONFIG_FILE"
    echo ">>> Concurrency ${concurrency} complete."
done

set_search_mode true true
set_concurrency "1,2,4,8,16,32,64,128"
