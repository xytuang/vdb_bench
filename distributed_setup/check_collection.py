from pymilvus import MilvusClient

client = MilvusClient(uri="http://10.1.1.8:31053")
collections = client.list_collections()
print(collections)

