import torch
import torch.nn as nn

input = torch.randn(2, 4, 3)

# 1.单层单向RNN
rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=1,
    batch_first=True,
    bidirectional=False,
)

output, hn = rnn(input, None)
print(output.shape) # torch.Size([2, 4, 4])
print(hn.shape) # torch.Size([1, 2, 4])

# 2.单层双向RNN
rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=1,
    batch_first=True,
    bidirectional=True,
)

output, hn = rnn(input, None)
print(output.shape) # torch.Size([2, 4, 8])
print(hn.shape) # torch.Size([2, 2, 4])

# 3.多层双向RNN
rnn = nn.RNN(
    input_size=3,
    hidden_size=4,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
)

output, hn = rnn(input, None)
print(output.shape) # torch.Size([2, 4, 8])
print(hn.shape) # torch.Size([4, 2, 8])
