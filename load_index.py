from pymilvus import connections, Collection, utility
from pymilvus.client.types import LoadState
import time

connections.connect(host='localhost', port='19530')

print('Releasing and reloading collection to trigger sync warmup')
coll = Collection('VDBBench')
coll.release()
while utility.load_state('VDBBench') != LoadState.NotLoad:
    print('.', end='', flush=True)
    time.sleep(2)

coll.load()
while utility.load_state('VDBBench') != LoadState.Loaded:
    print('.', end='', flush=True)
    time.sleep(5)

print('Collection loaded!')

