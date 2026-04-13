from pymilvus import MilvusClient

# 1. Connect to your Milvus instance
client = MilvusClient(uri="http://10.10.1.6:31658") # Update URI as necessary


progress = client.describe_index("VDBBench", "vector_idx")
print(progress)

