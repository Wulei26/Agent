import torch
import torch.nn as nn


class IMEModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # 1.词嵌入层
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim=128,  # 每个词向量的维度
        )
        # 2.RNN
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=256,
            bidirectional=False,
            batch_first=True,
        )
        # 3. 全连接层
        self.linear = nn.Linear(
            in_features=256,
            out_features=vocab_size,
        )

    def forward(self, x):
        # 传入x的形状为(batch_size, L=5)
        # print(f"进入模型前的形状为:{x.shape}")
        # 经过embedding层之后 -> （batch_size, L,embedding_dim = 128)
        embeddings = self.embeddings(x)
        # print(f"经过embedding后的形状为:{embeddings.shape}")
        # 经过 rnn之后，输出维度为 (batch_size, L,1*256)
        rnn_output, hn = self.rnn(embeddings)
        # print(f"经过 rnn 后的形状为:{rnn_output.shape}")
        # 直接取 output的最后一个时间步，这里面的信息是最丰富的, 作为最终特征
        # last_hidden = rnn_output[:, -1, :]  # 形状为 （batch_size, hidden_size）
        # 或者通过隐藏层矩阵直接获取最后一个时间步
        last_hidden = hn[-1]  # 形状为 （batch_size, hidden_size）
        # print(f"经过 rnn_output 压缩之后 后的形状为:{last_hidden.shape}")
        # 经过全连接层之后（batch_size ,vocab_size)
        result = self.linear(last_hidden)
        # print(f"经过 全链接层 压缩之后 后的形状为:{result.shape}")

        return result


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 32
    windows = 5
    input = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, windows),
    )
    model = IMEModel(vocab_size=1000)
    result = model(input)
