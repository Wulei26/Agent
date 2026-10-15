import torch
import torch.nn as nn
import config


class TranslatorEncoder(nn.Module):

    def __init__(self, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # 1. embedding层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        # 单向单层 GRU
        self.encoder = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            bidirectional=False,
            batch_first=True,
        )

    # 前向传播，输入形状 (N, L)
    def forward(self, x):
        # x = (bathch_size,seq_len)
        embed = self.embedding(x)

        # x -> (batch_size,seq_len,embdding_dim)
        output, _ = self.encoder(embed)
        # print(f"经过 Encoder 后的形状为:{output.shape}")
        indices = torch.arange(output.shape[0])  # 生成 [0, 1, 2, ..., batch_size-1]，形状 (batch_size,)
        x_true_lenth = (x != self.embedding.padding_idx).sum(dim=1)
        # 每个样本的真实长度（非 pad 的 token 数），形状 (batch_size,)

        last_hidden_state = output[indices, x_true_lenth - 1]
        # 输出形状 -> (N,L,embedding_dim)
        return last_hidden_state


class TranslatorDecoder(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # 1. embedding层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx,
        )
        # 单向单层 GRU
        self.decoder = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            bidirectional=False,
            batch_first=True,
        )
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size,
        )

    def forward(self, x, context_vector):
        """
        x形状 ： (N,L)
        context_vector (N,L,embedding_dim)
        """
        embed = self.embedding(x)
        output, hn = self.decoder(embed, context_vector)
        # 特征聚合, 得到输出维度(N,L,hidden_size)
        output = self.linear(output)
        return output, hn


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
    # x = torch.randint(1000, (64, 20))
    # print(x.shape)
    # encoder = TranslatorEncoder(1000)
    # output = encoder(x)
    # print(output)
    # print(output.shape)
    model = TranslatorModel(1000, 1024, 0, 0)

    input = torch.randint(1000, (64, 20))

    context_vector = model.encoder(input)
    print(context_vector.shape)

    input_dec = torch.randint(1024, (64, 16))
    output, hn = model.decoder(input_dec, context_vector.unsqueeze(0))
    print(output.shape)
    print(hn.shape)
