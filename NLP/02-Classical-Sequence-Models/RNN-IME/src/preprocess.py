"""
处理原始数据，生成训练数据集，并保存成 jsonl 格式
"""
import pandas as pd
import jieba
from tqdm import tqdm
from sklearn.model_selection import train_test_split
UNK_TOKEN = '<UNK>'
SEQ_LEN=5
# 定义函数：传入语料和word2id，构建数据集
def build_dataset(sentences, word2id):
    dataset = []    # 字典对象的列表，保存每一对{input, target}
    # 1. 分词，id化：得到id列表（一句话）的列表
    sentences_id = []

    for sentence in sentences:
        sentence_id = []
        for token in jieba.lcut(sentence):
            token_id = word2id.get(token, word2id[UNK_TOKEN])
            sentence_id.append(token_id)
        sentences_id.append(sentence_id)
    # 2. 构建数据集，每个元素为 {'input':[id1, id2, id3, id4, id5], 'target': idt}
    for sentence_id_list in tqdm(sentences_id, desc="构建数据集"):
        # 遍历这句话的id列表
        for i in range(len(sentence_id_list) - SEQ_LEN):
            input = sentence_id_list[i:i+SEQ_LEN]
            target = sentence_id_list[i+SEQ_LEN]
            dataset.append({'input': input, 'target': target})

    return dataset
def preprocess_data(input_file):

    df = pd.read_json(input_file, lines=True).sample(frac=0.1)
    # print(df.head())、
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            # 对于每一句话将前面的user截取
            sentences.append(sentence.split("：")[1].strip())
    # 划分数据集和训练集
    train_sentences, test_sentences = train_test_split(sentences,test_size=0.2)
    # 根据训练集构建词表

    vocab_set= set()
    for sentence in sentences:
        vocab_set.update(jieba.lcut(sentence))
    # print(f"词表大小 {len(vocab_size)}")
        # 增加特殊token：未登录词
    id2word = [UNK_TOKEN] + list(vocab_set)
    word2id = {word:id for id, word in enumerate(id2word)}
    print(id2word[0:10])
    # 5. 保存词表
    with open("../models/test.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(id2word))

    # 6. 构建数据集
    train_dataset = build_dataset(train_sentences, word2id)
    test_dataset = build_dataset(test_sentences, word2id)

    # 7. 保存数据集到文件
    pd.DataFrame(train_dataset).to_json("../models/train.jsonl", orient='records', lines=True)
    pd.DataFrame(test_dataset).to_json("../models/test.jsonl", orient='records', lines=True)

if __name__=="__main__":
    preprocess_data("../data/raw/synthesized_.jsonl")