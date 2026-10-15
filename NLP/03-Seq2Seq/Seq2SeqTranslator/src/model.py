import torch
import torch.nn as nn
import config


class TranslatorEncoder(nn.Module):

    def __init__(self, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # 1. Embedding 权重矩阵：(V_src, EMBEDDING_DIM) V_src :源语言词表大小
        # 输入：(N, SEQ_LEN)，每个元素是源语言 token 的索引
        # 输出：(N, SEQ_LEN, EMBEDDING_DIM)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        # 2. 单向、单层 GRU
        # 输入：(N, SEQ_LEN, EMBEDDING_DIM) SEQ_LEN:源语言序列长度
        # 初始隐藏状态 h0：(1, N, HIDDEN_SIZE)，不传时默认全 0
        # 输出 output：(N, SEQ_LEN, HIDDEN_SIZE)
        # 输出 hn：(1, N, HIDDEN_SIZE)
        self.encoder = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            bidirectional=False,
            batch_first=True,
        )


    def forward(self, x):
        # # x：(N, SEQ_LEN)
        embed = self.embedding(x)
        # embed：(N, SEQ_LEN, EMBEDDING_DIM)

        output, _ = self.encoder(embed)
        # output：(N, SEQ_LEN, HIDDEN_SIZE)
        # output[n, t]：第 n 个样本在第 t 个时间步的隐藏状态

        # print(f"经过 Encoder 后的形状为:{output.shape}")
        indices = torch.arange(output.shape[0])  # 生成 [0, 1, 2, ..., batch_size-1]，形状 (batch_size,)
        x_true_lenth = (x != self.embedding.padding_idx).sum(dim=1)
        # 每个样本的真实长度（非 pad 的 token 数），形状 (batch_size,)

        last_hidden_state = output[indices, x_true_lenth - 1]
        # last_hidden_state：(N, HIDDEN_SIZE)
        # 取每个样本最后一个有效 token 对应的隐藏状态

        context_vector = last_hidden_state.unsqueeze(0)
        # context_vector：(1, N, HIDDEN_SIZE)
        # 增加 GRU 所需的“层数 × 方向数”维度
        return context_vector


class TranslatorDecoder(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # 1. Embedding 权重矩阵：(V_tgt, EMBEDDING_DIM)  V_tgt：目标语言词表大小
        # 输入：(N, SEQ_LEN)
        # 输出：(N, SEQ_LEN, EMBEDDING_DIM)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        # 2.单向、单层 GRU
        # 输入 embed：(N, SEQ_LEN, EMBEDDING_DIM)
        # 初始隐藏状态 context_vector：(1, N, HIDDEN_SIZE)
        # 输出 output：(N, SEQ_LEN, HIDDEN_SIZE)
        # 输出 hn：(1, N, HIDDEN_SIZE)
        self.decoder = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            bidirectional=False,
            batch_first=True,
        )
        # 3.对每个时间步的隐藏状态进行线性映射
        # 输入：(N, SEQ_LEN, HIDDEN_SIZE)
        # 输出：(N, SEQ_LEN, V_tgt) V_tgt：目标语言词表大小
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size,
        )


    def forward(self, x, context_vector):
        """
        参数：
            x：(N, SEQ_LEN)，目标语言 token 索引
            context_vector：(1, N, HIDDEN_SIZE)，GRU 初始隐藏状态

        返回：
            logits：(N, SEQ_LEN, V_tgt)，每个位置的词表预测分数 (V_tgt :目标语言词表大小)
            hn：(1, N, H)，最后一个时间步的隐藏状态
        """
        embed = self.embedding(x)
        # embed：(N, SEQ_LEN, EMBEDDING_DIM) SEQ_LEN:输入解码器的序列长度

        output, hn = self.decoder(embed, context_vector)
        # output：(N, SEQ_LEN, HIDDEN_SIZE)
        # hn：(1, N, HIDDEN_SIZE)

        # 特征聚合, 得到输出维度(N,L,hidden_size)
        logits = self.linear(output)
        # logits：(N, T, V_tgt) V_tgt：目标语言词表大小
        # 每个时间步对应 V_tgt 个未归一化分数
        return logits, hn


class TranslatorModel(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        target_vocab_size,
        input_padding_idx,
        target_padding_idx,
    ):
        super().__init__()
        self.encoder = TranslatorEncoder(input_vocab_size, input_padding_idx)
        self.decoder = TranslatorDecoder(target_vocab_size, target_padding_idx)

    # 前向传播逻辑省略


if __name__ == "__main__":
    model = TranslatorModel(1000, 1024, 0, 0)

    input = torch.randint(1000, (64, 20))

    context_vector = model.encoder(input)
    print(context_vector.shape)

    input_dec = torch.randint(1024, (64, 16))
    output, hn = model.decoder(input_dec, context_vector)
    print(output.shape)
    print(hn.shape)
