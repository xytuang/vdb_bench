#!/bin/bash
set -euo pipefail

CONFIG_FILE="config.yaml"

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

echo ""
echo "========================================"
echo " Start cache sweep test"
echo "========================================"

for run in {1..3}; do
    echo ""
    echo "--- Run: ${run} ---"

    restart_milvus
    drop_page_cache
    python3 load_index.py
    sleep 30
    # Patch the yaml so only this single concurrency level is tested
    # Warmup for 5 minutes
    NUM_PER_BATCH=10000 vectordbbench milvusdiskann --config-file "$CONFIG_FILE"
    NUM_PER_BATCH=10000 vectordbbench milvusdiskann --config-file "$CONFIG_FILE"
    echo ">>> Run ${run} complete."
done

