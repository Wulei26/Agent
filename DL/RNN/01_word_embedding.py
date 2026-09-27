import torch
import torch.nn as nn
import jieba

# 设置随机种子
torch.manual_seed(42)
text = "自然语言是由文字构成的，而语言的含义是由单词构成的。即单词是含义的最小单位。因此为了让计算机理解自然语言，首先要让它理解单词含义。"
# 自定义停用词和标点符号
stopwords = {"的", "是", "而", "由", "，", "。", "、"}
# 分词，过滤停用词和标点，去重，构建词表
words = [word for word in jieba.lcut(text) if word not in stopwords]
vocab = list(set(words)) # 词表

# 构建词到索引的映射
word2idx = dict()
for idx, word in enumerate(vocab):
    word2idx[word] = idx
# 初始化嵌入层
embed = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=5)
# 打印词向量
for idx, word in enumerate(vocab):
    word_vec = embed(torch.tensor(idx))  # 通过索引获取词向量
    print(f"{idx:>2}:{word:8}\t{word_vec.detach().numpy()}")