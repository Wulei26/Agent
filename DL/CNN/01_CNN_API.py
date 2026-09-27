import torch
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("/storage/data/尚硅谷ai/09_尚硅谷AI大模型之深度学习/2.资料/data")
img = plt.imread(data_dir / "duck.jpg")
print("图片数据形状：", img.shape)

# 将图片数据转换为张量并改变形状
input = torch.tensor(img).permute(2, 0, 1).float()
print("输入特征图形状：", input.shape)

"""
1. torch.tensor(img)
将输入（通常是numpy数组或PIL图像）转换为PyTorch张量。
2. .permute(2, 0, 1)
维度重排，这是关键操作：

原始图像通常是 HWC 格式：(Height, Width, Channels)

例如 (224, 224, 3) 表示 224×224 的RGB图像

PyTorch 卷积网络要求 CHW 格式：(Channels, Height, Width)

转换为 (3, 224, 224)

permute(2, 0, 1) 的含义：

新维度0 ← 原维度2（Channels）

新维度1 ← 原维度0（Height）

新维度2 ← 原维度1（Width）
"""
# 初始化卷积核
conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, stride=3, padding=0, bias=False)
print(conv.weight.shape) ## torch.Size([3, 3, 9, 9])
# 对输入特征图进行卷积操作
output = conv(input)
print("输出特征图形状：", output.shape)

# 将输出特征图转换为图片
output = torch.clamp(output.int(), 0, 255) # 限制输出在0到255之间
output = output.permute(1, 2, 0).detach().numpy()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img)
ax[1].imshow(output)
plt.axis("off")
plt.show()