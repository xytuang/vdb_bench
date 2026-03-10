import struct
import time
import numpy as np
from pymilvus import connections, Collection, MilvusClient
import threading
from datetime import datetime
import json
import os
from collections import defaultdict
import statistics

def compact():
    connections.connect(host="node0", port=19530)
    collection = Collection("VDBBench")
    job_id = collection.compact()
    print(f"job_id: {job_id}")
    collection.wait_for_compaction_completed()




if __name__ == "__main__":
    compact()
