"""
训练好的词向量，通常用于初始化下游NLP任务的嵌入层。

在现代深度学习的 NLP 模型中，大多数任务的输入第一层都是嵌入层。本质上，嵌入层就是一个查找表（lookup table）：输入是词在词汇表中的索引；输出是该词对应的向量表示。
"""

import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# 1. 加载预训练的 Word2Vec 模型
word_vectors = KeyedVectors.load_word2vec_format("my_vectors.kv")

# 2. 构建词表和词向量矩阵
word2index = word_vectors.key_to_index  # 词到索引的映射
embedding_dim = word_vectors.vector_size  # 词语向量维度
num_embeddings = len(word2index)  # 词表大小

print("词表大小:", num_embeddings, "词向量维度:", embedding_dim)

# 2.1如果是随机初始化
embeddings = nn.Embedding(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
)

embedding_matrix = torch.zeros(num_embeddings, embedding_dim)

for word, idx in word2index.items():
    embedding_matrix[idx] = torch.tensor(word_vectors[word])

# 2.2. 构建 PyTorch 的嵌入层
embedding_layer = nn.Embedding.from_pretrained(
    embedding_matrix,  # 词向量矩阵，形状为(num_embeddigns,embedding_dim)
    freeze=False,  # 是否冻结词向量
)

# 4. 示例：将词索引转换为向量
input_words = ["我", "喜欢", "乘坐", "地铁"]  # 分词后的句子
input_indices = [word2index[word] for word in input_words]  # token转为索引
input_tensor = torch.tensor([input_indices, input_indices])  # 构造嵌入层输入张量

# 5. 查询嵌入（即词向量查找）
output = embedding_layer(input_tensor)  # 通过嵌入层查找预训练词向量

print(output.shape)  # 例如 torch.Size([1, 4, 100]
