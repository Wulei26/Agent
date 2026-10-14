import torch
import torch.nn as nn
import config

class GRUSentimentModel(nn.Module):

    def __init__(self, vocab_size: int,padding_idx:int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        # 1.词嵌入层
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,  # 词表大小
            embedding_dim=config.EMBEDDING_DIM,  # 每个词向量的维度
            padding_idx=padding_idx
        )
        # 2.GRU 单层单项
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            bidirectional=False,
            batch_first=True,
        )
        # 3. 全连接层
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=1,
        )

    def forward(self, x):
        # 传入x的形状为(batch_size, L=SEQ_LEN)
        # print(f"进入模型前的形状为:{x.shape}")
        # 经过embedding层之后 -> （batch_size, L,embedding_dim = 128)
        embeddings = self.embeddings(x)
        # print(f"经过embedding后的形状为:{embeddings.shape}")
        # 经过 GRU 之后，输出维度为 
        output, _ = self.gru(embeddings)
        # print(f"经过 GRU 后的形状为:{output.shape}")
        """
        但是现在有一个问题，我们对原来的句子进行了填充到128，所以这里的最后一个时间步前面可能包含了很多填充<PAD>
        所以，要截止到填充前的那个时间步
        output的形状为 （batch_size ,sen_len , directions * hidden_size)

        """
        indices = torch.arange(output.shape[0])                    # 生成 [0, 1, 2, ..., batch_size-1]，形状 (batch_size,)
        x_true_lenth = (x != self.embeddings.padding_idx).sum(dim=1)  # 每个样本的真实长度（非 pad 的 token 数），形状 (batch_size,)

        last_hidden_state = output[indices, x_true_lenth - 1]
        # output[indices, lengths] 是二维高级索引：
        #   - 第 0 维：按 indices 逐样本取（即取所有样本）
        #   - 第 1 维：按每个样本各自的真实最后一步下标取
        #   - 因为两个索引都是 1D 张量且等长，它们会逐对配对：
        #       last_hidden_state[i] = output[i, x_true_lenth[i] - 1]
        #   - 第 2 维（hidden 特征维）被完整保留
        #
        # 维度变化：
        #   output                : (batch_size, seq_len, num_directions * hidden_size)
        #   indices               : (batch_size,)
        #   x_true_lenth - 1      : (batch_size,)  每个样本真实最后一步的下标
        #   last_hidden_state     : (batch_size, num_directions * hidden_size)
        # 经过全连接层之后（batch_size ,1)
        result = self.linear(last_hidden_state).squeeze(-1)
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
    model = GRUSentimentModel(vocab_size=1000)
    result = model(input)
