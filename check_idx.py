from pymilvus import MilvusClient

# 1. Connect to your Milvus instance
client = MilvusClient(uri="http://localhost:19530") # Update URI as necessary


progress = client.describe_index("VDBBench", "vector_idx")
print(progress)

