import chromadb
from chromadb import Embeddings

client = chromadb.Client()
collection = client.get_or_create_collection("my_collection")
# print(collection.configuration)
"""
{
  "hnsw": {
    "space": "l2",
    "ef_construction": 100,
    "ef_search": 100,
    "max_neighbors": 16,
    "resize_factor": 1.2,
    "sync_threshold": 1000
  },
  "spann": null,
  "embedding_function": "<chromadb.api.types.DefaultEmbeddingFunction object>"
}


"""
# 1.chroma中的默认Embedding实现
"""
class DefaultEmbeddingFunction(EmbeddingFunction[Documents]):
    ""Default embedding function that delegates to ONNXMiniLM_L6_V2.""

    def __init__(self) -> None:
        if is_thin_client:
            return

    def __call__(self, input: Documents) -> Embeddings:
        # Import here to avoid circular imports
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        return ONNXMiniLM_L6_V2()(input)

"""
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

default_embedding_fn = DefaultEmbeddingFunction()
embeddings = default_embedding_fn(["wulei"])
# print(embeddings)

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings


# 2. 怎么去自定义embedding算法
# 参考 DefaultEmbeddingFunction的写法
class MyEmbeddingFunction(EmbeddingFunction[Documents]):

    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        # 我这里写一个非常简单的embedding方案，所有的Document都返回[1,2,3]
        embeddings = [np.array([1, 2, 3]) for item in input]
        return embeddings


# 现在创建一个collection然后指定我自己定义的 Embedding函数
my_collection = client.get_or_create_collection(
    name="custom_embedding",
    embedding_function=MyEmbeddingFunction(model=None),
)

my_collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This is a document about dogs",
    ],
)

print(my_collection.peek())
"""
{
  "ids": ["id1", "id2", "id3"],
  "embeddings": [
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0]
  ],
  "documents": [
    "This is a document about pineapple",
    "This is a document about oranges",
    "This is a document about dogs"
  ],
  "uris": null,
  "included": ["metadatas", "documents", "embeddings"],
  "data": null,
  "metadatas": [null, null, null]
}
"""
