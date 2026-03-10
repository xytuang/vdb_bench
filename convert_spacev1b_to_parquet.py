"""
convert_spacev1b_to_parquet.py

Converts the raw SPACEV1B binary dataset (from Microsoft SPTAG) into the
three Parquet files required by VectorDBBench:

    train.parquet      – training vectors  (id: int64, emb: list[float32])
    test.parquet       – query vectors     (id: int64, emb: list[float32])
    neighbors.parquet  – ground-truth k-NN (id: int64, neighbors_id: list[int32])

Binary file format (vectors.bin / query.bin):
    [int32 count][int32 dim][count × dim × int8 values]

Binary file format (truth.bin):
    [int32 num_queries][int32 top_k][num_queries × top_k × int32 neighbor_ids]
    Optionally followed by num_queries × top_k × float32 distances (ignored here).

Usage:
    python convert_spacev1b_to_parquet.py          # use defaults below
    python convert_spacev1b_to_parquet.py --help   # show all options
"""

import os
import struct
import time
import multiprocessing
import subprocess
import argparse

import numpy as np
import pandas as pd

RAW_DATA_DIR = "/mydata/SPTAG/datasets/SPACEV1B"
PARQUET_DATA_DIR = "/mydata/spacev1b"

RAW_DATASET_PATH = "vectors.bin/vectors_merged.bin"
RAW_QUERY_FILE = "query.bin"
RAW_TRUTH_FILE = "truth.bin"

# VectorDBBench requires exactly these three file names
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
NEIGHBORS_FILE = "neighbors.parquet"

# How many training vectors to include (full dataset is ~1 billion;
DEFAULT_NUM_TRAIN_VECTORS = 50_000_000
DEFAULT_BATCH_SIZE = 10_000


def raw_data_exists(raw_dir: str) -> bool:
    for rel in [RAW_DATASET_PATH, RAW_QUERY_FILE, RAW_TRUTH_FILE]:
        if not os.path.isfile(os.path.join(raw_dir, rel)):
            print(f"  Missing raw file: {os.path.join(raw_dir, rel)}")
            return False
    return True


def parquet_data_exists(parquet_dir: str) -> bool:
    for fname in [TRAIN_FILE, TEST_FILE, NEIGHBORS_FILE]:
        if not os.path.isfile(os.path.join(parquet_dir, fname)):
            return False
    return True


def download_data():
    subprocess.run(["./download_data.sh"], check=True)


class SPACEV1BReader:
    """
    Reader for SPACEV1B dataset binary files.

    File layout
    -----------
    vectors.bin / query.bin:
        4 bytes  – int32  count  (number of vectors)
        4 bytes  – int32  dim    (vector dimensionality)
        count × dim bytes – int8 vector components

    truth.bin:
        4 bytes  – int32  num_queries
        4 bytes  – int32  top_k
        num_queries × top_k × 4 bytes – int32 neighbor indices
        (optional: num_queries × top_k × 4 bytes – float32 distances)
    """

    HEADER_BYTES = 8  # two int32 values

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.vectors_file = os.path.join(dataset_path, RAW_DATASET_PATH)
        self.query_file   = os.path.join(dataset_path, RAW_QUERY_FILE)
        self.truth_file   = os.path.join(dataset_path, RAW_TRUTH_FILE)

        self.num_vectors, self.dim        = self._read_header(self.vectors_file)
        self.num_queries, self.query_dim  = self._read_header(self.query_file)
        self.num_truth,   self.top_k      = self._read_header(self.truth_file)

        print("Dataset info:")
        print(f"  Vectors : {self.num_vectors:,} × {self.dim}-d  (int8)")
        print(f"  Queries : {self.num_queries:,} × {self.query_dim}-d  (int8)")
        print(f"  Truth   : {self.num_truth:,} queries, top-{self.top_k} neighbors each")

        if self.dim != self.query_dim:
            raise ValueError(
                f"Dimension mismatch: vectors={self.dim}, queries={self.query_dim}"
            )
        if self.num_truth != self.num_queries:
            raise ValueError(
                f"Truth row count ({self.num_truth}) != query count ({self.num_queries})"
            )

    @staticmethod
    def _read_header(filepath: str):
        """Return (count, dim) from the first 8 bytes of a binary file."""
        with open(filepath, "rb") as f:
            count = struct.unpack("<i", f.read(4))[0]
            dim   = struct.unpack("<i", f.read(4))[0]
        return count, dim

    def read_vectors(self, start_idx: int, count: int) -> np.ndarray:
        """
        Read *count* vectors starting at *start_idx*.
        Returns float32 array of shape (count, dim).
        """
        with open(self.vectors_file, "rb") as f:
            offset = self.HEADER_BYTES + start_idx * self.dim
            f.seek(offset)
            data = f.read(count * self.dim)

        vectors = np.frombuffer(data, dtype=np.int8).reshape(count, self.dim)
        return vectors.astype(np.float32)

    def read_queries(self) -> np.ndarray:
        """
        Read all query vectors.
        Returns float32 array of shape (num_queries, dim).
        """
        with open(self.query_file, "rb") as f:
            f.seek(self.HEADER_BYTES)
            data = f.read(self.num_queries * self.query_dim)

        queries = np.frombuffer(data, dtype=np.int8).reshape(
            self.num_queries, self.query_dim
        )
        return queries.astype(np.float32)

    def read_truth(self) -> np.ndarray:
        with open(self.truth_file, "rb") as f:
            f.seek(self.HEADER_BYTES)
            data = f.read(self.num_truth * self.top_k * 4)  # int32 = 4 bytes

        truth = np.frombuffer(data, dtype=np.int32).reshape(self.num_truth, self.top_k)
        return truth


def create_train_file(
    raw_dir: str,
    parquet_dir: str,
    num_vectors: int = DEFAULT_NUM_TRAIN_VECTORS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Read up to *num_vectors* training vectors in batches and write
    train.parquet in the VectorDBBench schema:

        id     : int64
        emb    : list[float32]
    """
    reader = SPACEV1BReader(raw_dir)
    num_vectors = min(num_vectors, reader.num_vectors)
    num_batches = (num_vectors + batch_size - 1) // batch_size

    out_path = os.path.join(parquet_dir, TRAIN_FILE)
    start_time = time.time()

    # Collect all rows then write once.  For very large runs you could
    # write multiple parquet part-files and list them in VDBBench config.
    all_rows = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        count = min(batch_size, num_vectors - start_idx)

        vectors = reader.read_vectors(start_idx, count)

        for i in range(count):
            all_rows.append({
                "id":  start_idx + i,
                "emb": vectors[i].tolist(),
            })

        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            loaded  = min((batch_idx + 1) * batch_size, num_vectors)
            rate    = loaded / elapsed if elapsed > 0 else float("inf")
            print(
                f"  {loaded:,} / {num_vectors:,} vectors "
                f"({loaded * 100.0 / num_vectors:.1f}%)  –  "
                f"{rate:,.0f} vec/s"
            )

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)

    elapsed = time.time() - start_time
    print(
        f"train.parquet written ({len(df):,} vectors) in "
        f"{elapsed:.1f}s  ({num_vectors / elapsed:,.0f} vec/s)"
    )


def create_test_file(raw_dir: str, parquet_dir: str) -> None:
    """
    Read all query vectors and write test.parquet:

        id     : int64
        emb    : list[float32]
    """
    reader  = SPACEV1BReader(raw_dir)
    queries = reader.read_queries()

    rows = [
        {"id": i, "emb": queries[i].tolist()}
        for i in range(len(queries))
    ]

    out_path = os.path.join(parquet_dir, TEST_FILE)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"test.parquet written ({len(rows):,} queries)")


def create_neighbors_file(raw_dir: str, parquet_dir: str) -> None:
    """
    Read pre-computed ground-truth and write neighbors.parquet:

        id           : int64          – query index
        neighbors_id : list[int32]    – ordered list of neighbor indices
    """
    reader = SPACEV1BReader(raw_dir)
    truth  = reader.read_truth()   # shape: (num_queries, top_k)

    rows = [
        {"id": i, "neighbors_id": truth[i].tolist()}
        for i in range(len(truth))
    ]

    out_path = os.path.join(parquet_dir, NEIGHBORS_FILE)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"neighbors.parquet written ({len(rows):,} rows, top-{truth.shape[1]})")


def convert_to_parquet(
    raw_dir: str,
    parquet_dir: str,
    num_train: int,
    batch_size: int,
) -> None:
    """Run each writer in its own process (they are I/O-bound, not CPU-bound)."""

    tasks = []

    train_path     = os.path.join(parquet_dir, TRAIN_FILE)
    test_path      = os.path.join(parquet_dir, TEST_FILE)
    neighbors_path = os.path.join(parquet_dir, NEIGHBORS_FILE)

    if not os.path.isfile(train_path):
        tasks.append(multiprocessing.Process(
            target=create_train_file,
            args=(raw_dir, parquet_dir, num_train, batch_size),
            name="train",
        ))
    else:
        print(f"  Skipping train.parquet (already exists)")

    if not os.path.isfile(test_path):
        tasks.append(multiprocessing.Process(
            target=create_test_file,
            args=(raw_dir, parquet_dir),
            name="test",
        ))
    else:
        print(f"  Skipping test.parquet (already exists)")

    if not os.path.isfile(neighbors_path):
        tasks.append(multiprocessing.Process(
            target=create_neighbors_file,
            args=(raw_dir, parquet_dir),
            name="neighbors",
        ))
    else:
        print(f"  Skipping neighbors.parquet (already exists)")

    for p in tasks:
        print(f"  Starting process: {p.name}")
        p.start()

    for p in tasks:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Process '{p.name}' failed with exit code {p.exitcode}")

    print("\nAll parquet files ready.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SPACEV1B raw binary files to VectorDBBench parquet format."
    )
    parser.add_argument("--raw-dir",    default=RAW_DATA_DIR,
                        help="Directory containing the raw SPACEV1B files")
    parser.add_argument("--out-dir",    default=PARQUET_DATA_DIR,
                        help="Output directory for parquet files")
    parser.add_argument("--num-train",  type=int, default=DEFAULT_NUM_TRAIN_VECTORS,
                        help="Number of training vectors to include (default: 50M)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Vectors read per batch (default: 10 000)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir    = args.raw_dir
    parquet_dir = args.out_dir
    num_train  = args.num_train
    batch_size = args.batch_size

    print(f"Raw data dir   : {raw_dir}")
    print(f"Parquet out dir: {parquet_dir}")
    print(f"Train vectors  : {num_train:,}")
    print()

    # 1. Ensure raw data is present
    if not raw_data_exists(raw_dir):
        print("Raw data not found – attempting download …")
        download_data()
        if not raw_data_exists(raw_dir):
            raise FileNotFoundError(
                f"Raw SPACEV1B files still missing under '{raw_dir}'. "
                "Please download them manually."
            )

    # 2. Create output directory if needed
    os.makedirs(parquet_dir, exist_ok=True)

    # 3. Convert
    if parquet_data_exists(parquet_dir):
        print("All parquet files already exist – nothing to do.")
        return

    convert_to_parquet(raw_dir, parquet_dir, num_train, batch_size)


if __name__ == "__main__":
    main()
