import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = "/mydata/SPTAG/datasets/SPACEV1B"
PARQUET_DATA_DIR = "/mydata/spacev1b"

RAW_DATASET_PATH = "vectors.bin/vectors_merged.bin"
RAW_QUERY_FILE = "query.bin"
RAW_TRUTH_FILE = "truth.bin"

# VectorDBBench requires these 3 files
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
NEIGHBORS_FILE = "neighbors.parquet"


def data_exists():
    if not os.path.isfile(f"{RAW_DATA_DIR}/{RAW_DATASET_PATH}"):
        return False
    if not os.path.isfile(f"{RAW_DATA_DIR}/{RAW_QUERY_FILE}"):
        return False
    if not os.path.isfile(f"{RAW_DATA_DIR}/{RAW_TRUTH_FILE}"):
        return False
    return True

def download_data():
    subprocess.run(["./download_data.sh"])

def data_is_parquet():
    """
    Returns true if data already in parquet format
    """
    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{TRAIN_FILE}"):
        return False
    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{TEST_FILE}"):
        return False
    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{NEIGHBORS_FILE}"):
        return False
    return True


class SPACEV1BReader:
    """Reader for SPACEV1B dataset files"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.vectors_file = os.path.join(dataset_path, "vectors.bin/vectors_merged.bin")
        self.query_file = os.path.join(dataset_path, "query.bin")

        # Read dataset metadata
        self.num_vectors, self.dim = self._read_header(self.vectors_file)
        self.num_queries, self.query_dim = self._read_header(self.query_file)

        print(f"Dataset info:")
        print(f"  Vectors: {self.num_vectors:,} vectors of dimension {self.dim}")
        print(f"  Queries: {self.num_queries:,} queries of dimension {self.query_dim}")

        if self.dim != self.query_dim:
            raise ValueError(f"Dimension mismatch: vectors={self.dim}, queries={self.query_dim}")

    def _read_header(self, filepath):
        """Read the header of a binary file to get count and dimension"""
        with open(filepath, 'rb') as f:
            count = struct.unpack('i', f.read(4))[0]
            dim = struct.unpack('i', f.read(4))[0]
        return count, dim

    def read_vectors(self, start_idx, count):
        """Read a range of vectors from the merged file"""
        with open(self.vectors_file, 'rb') as f:
            # Skip header (8 bytes) and previous vectors
            header_size = 8
            bytes_per_vector = self.dim
            offset = header_size + start_idx * bytes_per_vector
            f.seek(offset)

            # Read the requested vectors
            bytes_to_read = count * bytes_per_vector
            data = f.read(bytes_to_read)

            # Convert to numpy array (int8 -> float32 for Milvus)
            vectors = np.frombuffer(data, dtype=np.int8).reshape(count, self.dim)
            return vectors.astype(np.float32)

    def read_queries(self):
        """Read all query vectors"""
        with open(self.query_file, 'rb') as f:
            # Skip header
            f.read(8)
            # Read all queries
            data = f.read()
            queries = np.frombuffer(data, dtype=np.int8).reshape(self.num_queries, self.query_dim)
            return queries.astype(np.float32)

def create_train_file(num_vectors=50_000_000, batch_size=10000):
    reader = SPACEV1BReader()
    num_batches = (num_vectors + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        count = min(batch_size, num_vectors - start_idx)

        # Read vectors
        vectors = reader.read_vectors(start_idx, count)

        # Prepare data
        data = [
            {
                "id": start_idx + i,
                "vector": vectors[i].tolist()
            }
            for i in range(count)
        ]

        # Insert TODO
        # client.insert(collection_name=collection_name, data=data)

        # Progress update
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            vectors_loaded = min((batch_idx + 1) * batch_size, num_vectors)
            rate = vectors_loaded / elapsed
            print(f"  Loaded {vectors_loaded:,} / {num_vectors:,} vectors "
                  f"({vectors_loaded*100.0/num_vectors:.1f}%) - "
                  f"{rate:,.0f} vectors/sec")


    queries.to_parquet(os.path.join(PARQUET_DATA_DIR, TRAIN_FILE), index=False)
    total_time = time.time() - start_time
    print(f"Initial load complete in {total_time:.1f}s ({num_vectors/total_time:,.0f} vectors/sec)")

def create_test_file():
    reader = SPACEV1BReader()
    queries = reader.read_queries()
    queries.to_parquet(os.path.join(PARQUET_DATA_DIR, TEST_FILE), index=False)

def create_neighbors_file():
    reader = SPACEV1BReader()
    truth = reader.read_truth()
    truth.to_parquet(os.path.join(PARQUET_DATA_DIR, NEIGHBORS_FILE), index=False)

def convert_to_parquet():
    processes = []
    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{TRAIN_FILE}"):
        p = multiprocessing.Process(target=create_train_file)
        processes.append(p)
        p.start()

    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{TEST_FILE}"):
        p = multiprocessing.Process(target=create_test_file)
        processes.append(p)
        p.start()

    if not os.path.isfile(f"{PARQUET_DATA_DIR}/{NEIGHBORS_FILE}"):
        p = multiprocessing.Process(target=create_neighbors_file)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

def main():
    if not data_exists():
        download_data()

    if not os.path.isdir(f"{PARQUET_DATA_DIR/spacev1b}"):
        subprocess.run(["mkdir", f"{PARQUET_DATA_DIR}/spacev1b"])

    if not data_is_parquet():
        convert_to_parquet()

if __name__ == "__main__":
    main()
