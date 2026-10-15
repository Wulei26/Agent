# 定义分词器

import jieba
from tqdm import tqdm
from typing import List, Any
import config
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

jieba.setLogLevel(jieba.logging.WARNING)


class Tokenizer:
    """
    基于 jieba的分词器，用于分词、编码和词表管理
    """

    unk_token = config.UNK_TOKEN
    pad_token = config.PAD_TOKEN
    start_token = config.START_TOKEN
    end_token = config.END_TOKEN

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.id2token = vocab
        self.token2id = {word: id for id, word in enumerate(self.vocab)}
        # 获取 未知词 索引
        self.unk_token_index = self.token2id[__class__.unk_token]
        # 获取 pad 词索引
        self.pad_token_index = self.token2id[__class__.pad_token]
        # 获取 start 词索引
        self.start_token_index = self.token2id[__class__.start_token]
        # 获取 end 词索引
        self.end_token_index = self.token2id[__class__.end_token]

    @classmethod
    def from_vocab(cls, vocab_file_path: str):
        """
        基于已经构建好的词表，创建 Tokenizer 对象
        """
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = f.readlines()
            vocab_list = [s.strip() for s in vocab_list]
        return cls(vocab_list)

    @classmethod
    def build_vocab_from_corpus(cls, corpus: List[str], vocab_save_path: str):
        """
        基于语料构建词表并保存，语料是句子列表
        """
        vocab_set = set()
        for sentence in tqdm(corpus, desc="building vocab..."):
            vocab_set.update(cls.tokenize(sentence))
        # 增加特殊token：未登录词 、PAD, START ,END
        id2word = [cls.pad_token, cls.unk_token, cls.start_token, cls.end_token] + list(vocab_set)
        # 5. 保存词表
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(id2word))
        return cls(id2word)

    @classmethod
    def tokenize(cls, text: str) -> List[Any | str]:
        """
        对句子进行分词。

        :param sentence: 输入句子。
        :return: 分词后的 token 列表。
        """
        pass

    # 编码方法（sentence ->id ） ，是否添加开始和结束的标记 -mark 参数
    def encode(self, sentence: str, mark: bool = False) -> List[int]:
        """
        将句子编码为索引列表。

        :param sentence: 输入句子。
        :param mark: 是否添加开始和结束的标记

        :return: 索引列表。
        """
        tokens = self.tokenize(sentence)

        ## 如果是目标序列，那么就添加 start 和 end 标记
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]

        return [self.token2id.get(i, self.unk_token_index) for i in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """
        将索引列表解码为 token 列表。

        :param ids: 索引列表。
        :return: 对应的 token 列表。
        """
        return [self.id2token[i] for i in ids]


class ChineseTokenizer(Tokenizer):

    @classmethod
    def tokenize(cls, text):
        return list(text)  # 按照字符级分词


class EnglishTokenizer(Tokenizer):

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text):
        """
        分词方法，将text进行分词
        """
        return cls.tokenizer.tokenize(text)

    def decode(self, ids: List[int]):
        # 获取分词结果
        tokens = [self.id2token[id] for id in ids]
        return self.detokenizer.detokenize(tokens)


if __name__ == "__main__":
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    print(f"词表大小:{zh_tokenizer.vocab_size}")
    print(f"PAD 符号:{zh_tokenizer.pad_token} ,对应id {zh_tokenizer.pad_token_index}")
    print(f"unk符号:{zh_tokenizer.unk_token} ,对应id {zh_tokenizer.unk_token_index}")
    print(zh_tokenizer.tokenize("我爱北京天安门"))
    print(zh_tokenizer.encode("我爱北京天安门"))
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)
    print(f"词表大小:{en_tokenizer.vocab_size}")
    print(f"PAD 符号:{en_tokenizer.pad_token} ,对应id {en_tokenizer.pad_token_index}")
    print(f"unk符号:{en_tokenizer.unk_token} ,对应id {en_tokenizer.unk_token_index}")
    s1 = "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88."
    print(en_tokenizer.tokenize(s1))
    print(en_tokenizer.encode(s1, mark=True))
