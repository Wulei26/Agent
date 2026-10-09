import chromadb
#  创建客户端
client = chromadb.EphemeralClient()

# 创建集合
collection = client.create_collection(name="my_collection")
collection.add(
    ids=["id1", "id2","id3"],
    embeddings=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
)
# chroma内部是怎么衡量相似度的呢？
results = collection.query(
    query_embeddings=[[1, 2, 3]],
    n_results=3,
)
print(results) #可以看出chroma 默认使用Chroma 的 l2 使用平方欧氏距离，公式没有开平方

# 指定不同的距离度量方式
collection = client.create_collection(name="my_collection_cosine",
                                      configuration= {
                                          'hnsw': {
                                              'space': 'cosine',  # 使用余弦相似度
                                      }})
collection.add(
    ids=["id1", "id2","id3"],
    embeddings=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
)
results = collection.query(
    query_embeddings=[[1, 2, 3]],
    n_results=3,
)
print(results) 

# 指定不同的距离度量方式
collection = client.create_collection(name="my_collection_ip",
                                      configuration= {
                                          'hnsw': {
                                              'space': 'ip',  # 使用余弦相似度
                                      }})
collection.add(
    ids=["id1", "id2","id3"],
    embeddings=[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
)
results = collection.query(
    query_embeddings=[[1, 2, 3]],
    n_results=3,
)
print(results) 