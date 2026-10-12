# 1.概述
**文本表示**就是将自然语言转化为计算机能够理解的数值形式，是绝大多数自然语言处理（NLP）任务的基础步骤

早期的文本表示方法（如词袋模型）通常将整段文本编码为一个向量。这类方法实现简单、计算高效，但存在明显的局限性——表达语序和上下文语义的能力较弱。因此，现代 NLP 技术逐渐引入更加精细和表达力更强的文本表示方法，以更有效地建模语言的结构和含义。
文本表示的第一步通常是**分词**和**词表构建**

## 2. 分词（Tokenization）
是将原始文本切分为若干具有独立语义的最小单元（即token）的过程，是所有 NLP 任务的起点
## 3.词表示
在分词完成之后，文本被转换为一系列的 token（词、子词或字符）。然而，这些符号本身对计算机而言是不可计算的。因此，为了让模型能够理解和处理文本，必须将这些 token 转换为计算机可以识别和操作的数值形式，这一步就是所谓的词表示（word representation）。
词表示的发展经历了从稀疏的one-hot编码，到稠密的语义化词向量，再到近年来的上下文相关的词表示。不同的词表示方法在表达能力、语义建模、上下文适应性等方面存在显著差异。

### 3.1 One-Hot编码

### 3.2 语义化词向量
#### 3.2.1 静态稠密词向量
##### Word2Vec
Word2Vec 是 Google 在 2013 年提出的一种经典词嵌入（Word Embedding）方法。它的核心目标是：**通过大量文本训练，让语义或使用场景相近的单词，在向量空间中具有相近的表示。**

一、Word2Vec 有哪两种算法？
Word2Vec 主要包含两种训练架构：
- CBOW（Continuous Bag of Words）：通过上下文预测中心词。
- Skip-gram：通过中心词预测上下文

GloVe
FastText
### 3.3 上下文动态词表示
**ELMo、BERT、GPT / Transformer**

# 应用Word2Vec词向量
前面已经通过构建的词表和Word2Vec算法，得到了每个词的向量表达，但是Word2Vec 负责学习词向量，而 nn.Embedding 本质上是一个**可训练的查找表（Lookup Table）**，负责根据词的 ID 取出对应的词向量。由于我们前面通过已有的语料构建了词表，然后通过Word2Vec训练好了词向量，那么 Embedding 层并不需要重新生成这些词向量，而是可以直接加载已有的词向量，并在后续神经网络训练中选择是否继续更新它们。

嵌入层的参数矩阵可以有两种典型的初始化方式：
- 随机初始化
模型训练开始时，嵌入向量是随机生成的，模型会通过反向传播逐步学习每个词的表示。
```python
# 2.1如果是随机初始化
embeddings = nn.Embedding(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
)
```
```python
def reset_parameters(self) -> None:
    init.normal_(self.weight)
    self._fill_padding_idx_with_zero()
```
可以看到pytorch内部是通过正态的随机数来初始化Embedding矩阵的
- **使用预训练词向量初始化**
加载训练好的词向量（如 Word2Vec）到嵌入层中作为初始参数，这样可以为模型注入丰富的语言知识，尤其在低资源任务中优势明显。并且，加载预训练词向量后，可选择是否让嵌入层继续参与训练。

在pytorch中使用预训练词向量初始化Embedding层
1. 首先加载我们训练好的词向量
2. 构建词嵌入矩阵（embedding_matrix），词嵌入矩阵的规模是：(num_embeddigns,embedding_dim)，num_embeddings 是词表大小，embedding_dim是词向量维度
3. 然后将我们构建的词向量填入embedding_matrix，但是要将词向量转化为Tensor
4. 得到词嵌入矩阵（embedding_matrix）之后，就可以使用 `nn.Embedding.from_pretrained` API 构建 PyTorch 的嵌入层
5. 然后就可以将词索引转换为向量，从而得到一个句子的词嵌入矩阵
   示例：
   ```python
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
    input_tensor = torch.tensor([input_indices])  # 构造嵌入层输入张量

    # 5. 查询嵌入（即词向量查找）
    output = embedding_layer(input_tensor)  # 通过嵌入层查找预训练词向量

    print(output.shape)  # 例如 torch.Size([1, 4, 100]
    ```