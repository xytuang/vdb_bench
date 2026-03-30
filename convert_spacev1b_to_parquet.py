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

RAM-friendly training conversion
---------------------------------
50 M × 100-d float32 vectors ≈ 20 GB in a Python list-of-lists before Parquet
serialisation, which easily exhausts RAM on typical workstations.

The approach here:
  1. Split the work into NUM_TRAIN_PARTS equal slices.
  2. Write each slice to a *part file* (train_part_0.parquet … train_part_4.parquet)
     in separate processes so they run concurrently but each only holds 1/N of
     the data at a time.
  3. Once all part files exist, merge them *sequentially* – reading one part at a
     time and appending to the final train.parquet via PyArrow's ParquetWriter so
     the merge itself also stays RAM-efficient.
  4. Delete the part files after a successful merge.

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
import pyarrow as pa
import pyarrow.parquet as pq

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
DEFAULT_NUM_TRAIN_VECTORS = 500_000_000
DEFAULT_BATCH_SIZE = 10_000
# Number of equal slices the training data is split into before merging.
# Each part process holds at most (num_train / NUM_TRAIN_PARTS) vectors in RAM.
NUM_TRAIN_PARTS = 5


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
        """
        Read the pre-computed ground-truth neighbor indices.
        Returns int32 array of shape (num_queries, top_k).
        """
        with open(self.truth_file, "rb") as f:
            f.seek(self.HEADER_BYTES)
            data = f.read(self.num_truth * self.top_k * 4)  # int32 = 4 bytes

        truth = np.frombuffer(data, dtype=np.int32).reshape(self.num_truth, self.top_k)
        return truth


def _train_part_path(parquet_dir: str, part_idx: int) -> str:
    return os.path.join(parquet_dir, f"train_part_{part_idx}.parquet")


def _write_train_part(
    raw_dir: str,
    parquet_dir: str,
    part_idx: int,
    global_start: int,
    part_vectors: int,
    batch_size: int,
) -> None:
    """
    Worker function (runs in its own process).

    Reads *part_vectors* vectors from the binary file starting at
    *global_start*, and writes them to train_part_<part_idx>.parquet.
    Only 1/NUM_TRAIN_PARTS of the total data lives in RAM at any moment.
    """
    out_path = _train_part_path(parquet_dir, part_idx)
    if os.path.isfile(out_path):
        print(f"  [part {part_idx}] already exists, skipping")
        return

    reader     = SPACEV1BReader(raw_dir)
    num_batches = (part_vectors + batch_size - 1) // batch_size
    start_time = time.time()

    # PyArrow writer lets us append row-groups without holding all rows in RAM
    schema = pa.schema([
        pa.field("id",  pa.int64()),
        pa.field("emb", pa.list_(pa.float32())),
    ])
    writer = pq.ParquetWriter(out_path, schema)

    for batch_idx in range(num_batches):
        batch_start = global_start + batch_idx * batch_size
        count       = min(batch_size, global_start + part_vectors - batch_start)

        vectors = reader.read_vectors(batch_start, count)

        ids = list(range(batch_start, batch_start + count))
        emb = [vectors[i].tolist() for i in range(count)]

        table = pa.table({"id": ids, "emb": emb}, schema=schema)
        writer.write_table(table)

        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed  = time.time() - start_time
            progress = min((batch_idx + 1) * batch_size, part_vectors)
            rate     = progress / elapsed if elapsed > 0 else float("inf")
            print(
                f"  [part {part_idx}] {progress:,} / {part_vectors:,} "
                f"({progress * 100.0 / part_vectors:.1f}%)  –  {rate:,.0f} vec/s"
            )

    writer.close()
    elapsed = time.time() - start_time
    print(
        f"[part {part_idx}] done – {part_vectors:,} vectors in "
        f"{elapsed:.1f}s  ({part_vectors / elapsed:,.0f} vec/s)"
    )


def _merge_train_parts(parquet_dir: str, num_parts: int) -> None:
    """
    Merge part files into the final train.parquet *one part at a time*
    so peak RAM never exceeds a single part's worth of data.
    Deletes each part file immediately after it has been appended.
    """
    out_path = os.path.join(parquet_dir, TRAIN_FILE)
    print(f"\nMerging {num_parts} part files → {out_path}")

    # Infer schema from part 0 to initialise the writer
    first_part = _train_part_path(parquet_dir, 0)
    schema = pq.read_schema(first_part)
    writer = pq.ParquetWriter(out_path, schema)

    total_rows = 0
    for part_idx in range(num_parts):
        part_path = _train_part_path(parquet_dir, part_idx)
        print(f"  Appending part {part_idx} …", end=" ", flush=True)

        part_table = pq.read_table(part_path)
        writer.write_table(part_table)
        total_rows += len(part_table)

        # Free RAM immediately before reading the next part
        del part_table

        print(f"done  (running total: {total_rows:,} rows)")

        # Remove part file now that it's safely merged
        os.remove(part_path)
        print(f"  Deleted {part_path}")

    writer.close()
    print(f"train.parquet written – {total_rows:,} vectors total")


def create_train_file(
    raw_dir: str,
    parquet_dir: str,
    num_vectors: int = DEFAULT_NUM_TRAIN_VECTORS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_parts: int = NUM_TRAIN_PARTS,
) -> None:
    """
    Split *num_vectors* training vectors into *num_parts* equal slices,
    write each slice to its own parquet file in parallel, then merge
    them sequentially into the final train.parquet.

    Peak RAM per process ≈ (num_vectors / num_parts) × dim × 4 bytes.
    For 50 M vectors / 5 parts × 100-d = ~2 GB per part process.
    """
    reader      = SPACEV1BReader(raw_dir)
    num_vectors = min(num_vectors, reader.num_vectors)

    # Divide vectors as evenly as possible across parts
    base, remainder = divmod(num_vectors, num_parts)
    slices = []          # list of (global_start, part_size)
    offset = 0
    for i in range(num_parts):
        size = base + (1 if i < remainder else 0)
        slices.append((offset, size))
        offset += size

    # Phase 1 – write parts in parallel
    print(f"\nPhase 1: writing {num_parts} part files in parallel …")
    processes = []
    for part_idx, (global_start, part_size) in enumerate(slices):
        if os.path.isfile(_train_part_path(parquet_dir, part_idx)):
            print(f"  [part {part_idx}] already on disk, skipping")
            continue
        p = multiprocessing.Process(
            target=_write_train_part,
            args=(raw_dir, parquet_dir, part_idx, global_start, part_size, batch_size),
            name=f"train-part-{part_idx}",
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Part-writing process '{p.name}' failed (exit {p.exitcode})"
            )

    # Phase 2 – sequential merge (one part in RAM at a time)
    print(f"\nPhase 2: merging parts into train.parquet …")
    _merge_train_parts(parquet_dir, num_parts)


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
    num_parts: int,
) -> None:
    """
    Orchestrate conversion of all three files.
    train.parquet is written via the split-then-merge path to cap RAM usage.
    test.parquet and neighbors.parquet are small enough to write directly.
    """
    tasks = []

    train_path     = os.path.join(parquet_dir, TRAIN_FILE)
    test_path      = os.path.join(parquet_dir, TEST_FILE)
    neighbors_path = os.path.join(parquet_dir, NEIGHBORS_FILE)

    # train: call directly (it manages its own sub-processes internally)
    if not os.path.isfile(train_path):
        create_train_file(raw_dir, parquet_dir, num_train, batch_size, num_parts)
    else:
        print("  Skipping train.parquet (already exists)")

    # test + neighbors: small files, run in parallel with each other
    if not os.path.isfile(test_path):
        tasks.append(multiprocessing.Process(
            target=create_test_file,
            args=(raw_dir, parquet_dir),
            name="test",
        ))
    else:
        print("  Skipping test.parquet (already exists)")

    if not os.path.isfile(neighbors_path):
        tasks.append(multiprocessing.Process(
            target=create_neighbors_file,
            args=(raw_dir, parquet_dir),
            name="neighbors",
        ))
    else:
        print("  Skipping neighbors.parquet (already exists)")

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
    parser.add_argument("--num-parts",  type=int, default=NUM_TRAIN_PARTS,
                        help="Number of part files to split training data into before "
                             "merging (default: 5). Increase if you still run out of RAM.")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir     = args.raw_dir
    parquet_dir = args.out_dir
    num_train   = args.num_train
    batch_size  = args.batch_size
    num_parts   = args.num_parts

    print(f"Raw data dir   : {raw_dir}")
    print(f"Parquet out dir: {parquet_dir}")
    print(f"Train vectors  : {num_train:,}")
    print(f"Train parts    : {num_parts}  (~{num_train // num_parts:,} vectors each)")
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

    convert_to_parquet(raw_dir, parquet_dir, num_train, batch_size, num_parts)


if __name__ == "__main__":
    main()
