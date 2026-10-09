import chromadb

# 1. 临时客户端
client = chromadb.EphemeralClient()

# 1.1 创建集合
collection = client.create_collection(name="my_collection")
# 1.2 添加数据
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
)
# 1.3 查询数据
results = collection.query(
    query_texts=["This is a query document about hawaii","this is a computer"],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)
print(results)

#2.持久化客户端
client = chromadb.PersistentClient(path="/home/terry/code/learning/Agent/image_similarity_main/test/chroma_db/chroma_persistent_db")
# 2.1 创建集合
collection = client.create_collection(name="my_collection")
# 2.2 添加数据
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
)