import chromadb

# 1.创建客户端，在内存模式下运行
chroma_client = chromadb.Client()

# 2. 创建表，chroma中指的是 collection
collection = chroma_client.create_collection(name="my_collection")

collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
)
print(collection.count()) # 统计多少条数据

print(collection.peek())

##按照相似度进行查询
results = collection.query(
    query_texts=["This is a query document about hawaii"],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)
print(results)
