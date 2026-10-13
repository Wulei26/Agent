import jieba
from tqdm import tqdm
from typing import List,Any
jieba.setLogLevel(jieba.logging.WARNING)

class JiebaTokenizer:
    """
    基于 jieba的分词器，用于分词、编码和词表管理
    """
    unk_token = "<UNK>"

    def __init__(self,vocab:List[str]):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.id2token = vocab
        self.token2id = {word:id for id ,word in enumerate(self.vocab)}
        # 获取未知词索引
        self.unk_token_index = self.token2id[self.unk_token]

    @classmethod
    def build_vocab_from_corpus(cls,corpus:List[str],vocab_save_path:str):
        """
        基于语料构建词表并保存，语料是句子列表
        """
        vocab_set = set()
        for sentence in corpus:
            vocab_set.update(jieba.lcut(sentence))
        # 增加特殊token：未登录词
        id2word = [cls.unk_token] + list(vocab_set)
        # 5. 保存词表
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(id2word))

    
    @classmethod
    def from_vocab(cls,vocab_file_path:str):
        """
        基于已经构建好的词表，创建JiebaTokenizer对象
        """
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab_list = f.readlines()
            vocab_list = [s.strip() for s in vocab_list]
        return cls(vocab_list)
    
    @staticmethod
    def tokenize(text:str) ->List[Any | str]:
        """
        对句子进行分词。

        :param sentence: 输入句子。
        :return: 分词后的 token 列表。
        """
        return jieba.lcut(text)

    def encode(self, sentence:str):
        """
        将句子编码为索引列表。

        :param sentence: 输入句子。
        :return: 索引列表。
        """
        tokens = JiebaTokenizer.tokenize(sentence)
        return [self.token2id.get(i, self.unk_token_index) for i in tokens]


if __name__ == "__main__":
    tokenizer = JiebaTokenizer.from_vocab("/home/terry/code/learning/Agent/NLP/02-Classical-Sequence-Models/RNN-IME/models/vocab.txt")
    print(f"词表大小:{tokenizer.vocab_size}")
    print(f"unk符号:{tokenizer.unk_token} ,对应id {tokenizer.unk_token_index}")
    print(tokenizer.tokenize("自然语言处理"))
    print(tokenizer.encode("自然语言处理"))