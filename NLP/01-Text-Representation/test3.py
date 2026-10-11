import pandas as pd
import jieba
# 使用真实数据集构建词表和词向量
from gensim.models import Word2Vec, KeyedVectors
df = pd.read_csv('path/to/online_shopping_10_cats.csv', encoding='utf-8', usecols=['review'])
df = df.dropna(subset=['review'])

sentences = []
for review in df['review']:
    if not isinstance(review, str):   # 跳过 NaN、数字等非字符串
        continue
    tokens = [t for t in jieba.lcut(review) if t.strip() != '']
    if tokens:                         # 跳过切完是空列表的句子
        sentences.append(tokens)
model = Word2Vec(
    sentences,  # 已分词的句子序列
    vector_size=100,  # 词向量维度
    window=5,  # 上下文窗口大小
    min_count=2,  # 最小词频（低于将被忽略）
    sg=1,  # 1 = Skip-Gram，0 = CBOW
    workers=4  # 并行训练线程数
)

model.wv.save_word2vec_format('my_vectors.kv')

my_model = KeyedVectors.load_word2vec_format('my_vectors.kv')
print(my_model.vector_size)
print(my_model['上班'])
similarity = my_model.similarity('乘坐', '每天')
print('乘坐 vs 每天 相似度：', similarity)
print(my_model)