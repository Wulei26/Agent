# import chromadb

# client = chromadb.HttpClient(host="localhost", port=8999)

# collection = client.get_or_create_collection(name="my_collection")
# collection.add(
#     ids=["id1", "id2","id3"],
#     documents=[
#         "This is a document about pineapple",
#         "This is a document about oranges",
#         "This is a document about dogs"
#     ],
# )

import chromadb

client = chromadb.HttpClient(host="localhost", port=8999)

# 看有哪些 collection
print(client.list_collections())

# 拿到 collection
collection = client.get_collection(name="my_collection")

# 看总数
print(collection.count())   # 应该是 3

# 看全部数据（ids + documents + metadatas）
print(collection.get())