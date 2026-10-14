# 定义分词器

import jieba
from tqdm import tqdm
from typing import List, Any
import config

jieba.setLogLevel(jieba.logging.WARNING)


class JiebaTokenizer:
    """
    基于 jieba的分词器，用于分词、编码和词表管理
    """

    unk_token = config.UNK_TOKEN
    pad_token = config.PAD_TOKEN

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.id2token = vocab
        self.token2id = {word: id for id, word in enumerate(self.vocab)}
        # 获取 未知词 索引
        self.unk_token_index = self.token2id[__class__.unk_token]
        # 获取 pad 词索引
        self.pad_token_index = self.token2id[__class__.pad_token]

    @classmethod
    def build_vocab_from_corpus(cls, corpus: List[str], vocab_save_path: str):
        """
        基于语料构建词表并保存，语料是句子列表
        """
        vocab_set = set()
        for sentence in tqdm(corpus,desc="building vocab..."):
            vocab_set.update(jieba.lcut(sentence))
        # 增加特殊token：未登录词 、PAD
        id2word = [cls.pad_token, cls.unk_token]+ list(vocab_set)
        # 5. 保存词表
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(id2word))
        return  cls(id2word)
    @classmethod
    def from_vocab(cls, vocab_file_path: str):
        """
        基于已经构建好的词表，创建JiebaTokenizer对象
        """
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = f.readlines()
            vocab_list = [s.strip() for s in vocab_list]
        return cls(vocab_list)

    @staticmethod
    def tokenize(text: str) -> List[Any | str]:
        """
        对句子进行分词。

        :param sentence: 输入句子。
        :return: 分词后的 token 列表。
        """
        return jieba.lcut(text)

    def encode(self, sentence: str):
        """
        将句子编码为索引列表。

        :param sentence: 输入句子。
        :return: 索引列表。
        """
        tokens = JiebaTokenizer.tokenize(sentence)
        # 根据 SEQ_LEN  进行填充或者截取 到指定长度
        if len(tokens) > config.SEQ_LEN:
            tokens = tokens[0:config.SEQ_LEN]
        elif len(tokens) < config.SEQ_LEN:
            tokens = tokens + [self.pad_token] * (config.SEQ_LEN - len(tokens))
        return [self.token2id.get(i, self.unk_token_index) for i in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """
        将索引列表解码为 token 列表。

        :param ids: 索引列表。
        :return: 对应的 token 列表。
        """
        return [self.id2token[i] for i in ids]


if __name__ == "__main__":
    tokenizer = JiebaTokenizer.from_vocab(
        config.VOCAB_DATA_FILE
    )
    print(f"词表大小:{tokenizer.vocab_size}")
    print(f"PAD 符号:{tokenizer.pad_token} ,对应id {tokenizer.pad_token_index}")
    print(f"unk符号:{tokenizer.unk_token} ,对应id {tokenizer.unk_token_index}")
    print(tokenizer.tokenize("我爱北京天安门"))
    print(tokenizer.encode("我爱北京天安门"))
