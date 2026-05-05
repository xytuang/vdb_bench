#!/usr/bin/env python3
"""
insert_and_search.py

Inserts 10M vectors per round into the pre-existing VDBBench collection.
Can be run standalone (all 5 rounds) or called per-round by run_monitors.sh.

Usage:
    # Run all 5 rounds (standalone, no external orchestration)
    python3 insert_and_search.py

    # Run a single round (called by run_monitors.sh)
    python3 insert_and_search.py --single-round 1
    python3 insert_and_search.py --single-round 2
    ...
"""

import time
import numpy as np
from pymilvus import MilvusClient
import os
import sys
import argparse


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_PATH      = "/mydata/SPTAG/datasets/SPACEV1B/"
HOST              = "10.10.1.8"
PORT              = 31053
COLLECTION_NAME   = "VDBBench"
VECTORS_PER_ROUND = 10_000_000
NUM_ROUNDS        = 5
INSERT_BATCH_SIZE = 160_000


# ─────────────────────────────────────────────
# SPACEV1B reader
# ─────────────────────────────────────────────
class SPACEV1BReader:
    DIM = 100  # SPACEV1B is always 100-dimensional int8

    def __init__(self, dataset_path):
        self.vectors_file = os.path.join(
            dataset_path, "vectors.bin/new_vectors_merged.bin"
        )
        self.dim = self.DIM

        # Derive count from file size — merged file has no header
        file_size = os.path.getsize(self.vectors_file)
        self.num_vectors = file_size // self.dim
        print(f"Dataset: {self.num_vectors:,} vectors of dimension {self.dim}")

    def read_vectors(self, start_idx: int, count: int) -> np.ndarray:
        """Read `count` int8 vectors at `start_idx`, return as float32."""
        with open(self.vectors_file, "rb") as f:
            f.seek(start_idx * self.dim)  # no header in merged file
            data = f.read(count * self.dim)
        vectors = np.frombuffer(data, dtype=np.int8).reshape(count, self.dim)
        return vectors.astype(np.float32)


# ─────────────────────────────────────────────
# Insert
# ─────────────────────────────────────────────
def insert_vectors(client: MilvusClient, reader: SPACEV1BReader,
                   global_start_idx: int, num_vectors: int):
    """Insert num_vectors vectors starting at global_start_idx in the dataset."""
    end_idx = global_start_idx + num_vectors - 1
    print(f"\nInserting {num_vectors:,} vectors "
          f"(dataset offsets {global_start_idx:,} – {end_idx:,}) ...")

    num_batches    = (num_vectors + INSERT_BATCH_SIZE - 1) // INSERT_BATCH_SIZE
    start_time     = time.time()
    inserted_total = 0

    for batch_idx in range(num_batches):
        batch_start = global_start_idx + batch_idx * INSERT_BATCH_SIZE
        batch_count = min(INSERT_BATCH_SIZE,
                          global_start_idx + num_vectors - batch_start)

        vectors = reader.read_vectors(batch_start, batch_count)

        data = [
            {
                "pk": batch_start + i,
                "id": batch_start + i,
                "vector": vectors[i].tolist(),
            }
            for i in range(batch_count)
        ]

        client.insert(collection_name=COLLECTION_NAME, data=data)
        inserted_total += batch_count

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            rate    = inserted_total / elapsed
            pct     = inserted_total * 100.0 / num_vectors
            print(f"  [{pct:5.1f}%] {inserted_total:,} / {num_vectors:,}  "
                  f"({rate:,.0f} vec/s)")

    elapsed = time.time() - start_time
    print(f"Insert complete: {inserted_total:,} vectors in {elapsed:.1f}s  "
          f"({inserted_total / elapsed:,.0f} vec/s)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-round", type=int, default=None, metavar="N",
        help="Run only round N (1-based). If omitted, runs all rounds."
    )
    args = parser.parse_args()

    # Determine which rounds to run
    if args.single_round is not None:
        r = args.single_round
        if not (1 <= r <= NUM_ROUNDS):
            print(f"ERROR: --single-round must be between 1 and {NUM_ROUNDS}")
            sys.exit(1)
        rounds = [r]
    else:
        rounds = list(range(1, NUM_ROUNDS + 1))

    print("=" * 60)
    print("SPACEV1B INSERT")
    print("=" * 60)
    print(f"Collection   : {COLLECTION_NAME}")
    print(f"Rounds to run: {rounds}")
    print(f"Vectors/round: {VECTORS_PER_ROUND:,}")
    print(f"Batch size   : {INSERT_BATCH_SIZE:,}")
    print("=" * 60)

    # Connect
    client = MilvusClient(uri=f"http://{HOST}:{PORT}")
    if not client.has_collection(COLLECTION_NAME):
        print(f"ERROR: collection '{COLLECTION_NAME}' does not exist.")
        sys.exit(1)
    print(f"Connected to Milvus at {HOST}:{PORT}")

    # Dataset reader
    reader = SPACEV1BReader(DATASET_PATH)
    total_needed = NUM_ROUNDS * VECTORS_PER_ROUND
    if reader.num_vectors < total_needed:
        print(f"ERROR: dataset has {reader.num_vectors:,} vectors, "
              f"need {total_needed:,}")
        sys.exit(1)

    # Run rounds
    for round_num in rounds:
        dataset_offset = (round_num - 1) * VECTORS_PER_ROUND
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} — inserting at offset {dataset_offset:,}")
        print(f"{'='*60}")
        insert_vectors(client, reader, dataset_offset, VECTORS_PER_ROUND)

    print("\nDone.")


if __name__ == "__main__":
    main()
