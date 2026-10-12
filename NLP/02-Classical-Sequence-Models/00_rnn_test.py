import torch
import torch.nn as nn

input = torch.randn(2, 4, 3)

rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=1,
    batch_first=True,
    bidirectional=False,
)

output, hn = rnn(input, None)
print(output.shape)
print(hn.shape)
