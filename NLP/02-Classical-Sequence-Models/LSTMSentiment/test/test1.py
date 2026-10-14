import torch
import torch.nn as nn

model = nn.LSTM(
    input_size=3,  # 每个token对应的词向量维度是3
    hidden_size=4,  #
    num_layers=6,
    batch_first=True,
    bidirectional=True,
)
input = torch.randn(2, 4, 3)
print(f"输入 input (batch_size, seq_len, input_size), {input.shape}")
output, (h_n, c_n) = model(input)
print(f" 输出output {output.shape}")
print(f"输出 h_n {h_n.shape}")
print(f"输出 c_n {h_n.shape}")
