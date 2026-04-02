from pymilvus import connections, Collection
import numpy as np
import time

connections.connect(host='localhost', port='19530')
c = Collection('VDBBench')

vec = np.random.rand(1, 100).tolist()
consecutive_success = 0
i = 0

print('Starting cache warmup...')
while consecutive_success < 5:
    try:
        start = time.time()
        c.search(vec, 'vector',
                 {'metric_type': 'L2', 'params': {'search_list': 100}},
                 limit=10, timeout=3600)
        elapsed = time.time() - start
        consecutive_success += 1
        print(f'Search {i} succeeded in {elapsed:.1f}s ({consecutive_success}/5 consecutive)')
    except Exception as e:
        consecutive_success = 0
        print(f'Search {i} failed: {str(e)[:80]}')
    i += 1
    time.sleep(2)

print('Cache warmed!')
