"""
处理原始数据构建语料
    由于LSTM输入的尺寸是(batch_size, seq_len, input_size) ,但是每条评论的长度不一致怎么办？
    通常设定一个固定长度，大于这个长度的文本进行截取，小于这个长度的使用某个特定token进行补齐
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer

# 读取原始文件 -> 进行截取 -> 获取corpus ->通过分词器保存


def preprocess():
    raw_data = pd.read_csv(config.RAW_DATA_FILE, sep="\t", header=None, usecols=[0, 1], names=["en", "zh"], encoding="utf-8")
    print(raw_data.head(n=10))
    # 数据清洗
    raw_data = raw_data.dropna()
    # raw_data = raw_data.drop("cat", axis=1).dropna()
    # raw_data = raw_data[raw_data["review"].str.strip().ne("")]

    # 1. 直接开始划分数据集,注意这里要分层抽样，好评差评都要各自随机取0.2
    train_df, test_df = train_test_split(raw_data, test_size=config.TEST_RATE)

    # 2.直接构建词表
    zh_tokenizer = ChineseTokenizer.build_vocab_from_corpus(train_df["zh"].tolist(), vocab_save_path=config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.build_vocab_from_corpus(train_df["en"].tolist(), vocab_save_path=config.EN_VOCAB_FILE)

    # ## 3. 构建训练集饿和数据集
    zh_encode_fn = lambda text: zh_tokenizer.encode(text, mark=False)
    en_encode_fn = lambda text: en_tokenizer.encode(text, mark=True)
    train_df["zh"] = train_df["zh"].apply(zh_encode_fn)
    test_df["zh"] = test_df["zh"].apply(zh_encode_fn)
    train_df["en"] = train_df["en"].apply(en_encode_fn)
    test_df["en"] = test_df["en"].apply(en_encode_fn)

    ## 4.保存训练集和测试集
    train_df.to_json(config.TRAIN_DATA_FILE, orient="records", lines=True)
    test_df.to_json(config.TEST_DATA_FILE, orient="records", lines=True)


if __name__ == "__main__":
    preprocess()
