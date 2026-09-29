# 定义模块的公开接口，仅暴露ConvEncoder和ConvDecoder类
__all__ = ["ConvDenoiser"]

import torch
import torch.nn as nn


class ConvDenoiser(nn.Module):

    def __init__(self):
        super().__init__()
        # 自编码
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        ##池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 两个池化层是一样的
        # 转置卷积层，卷积核为2，步幅为2，将空间维度增加2倍
        self.t_conv1 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # 最后一个普通的卷积层，用于减少通道数
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    # forward函数用于向前传播，该函数接受一个输入张量x，并返回一个输出张量，该函数的实现真正决定了神经网络的架构
    def forward(self, x):
        # 1.卷积池化
        x = torch.relu(self.conv1(x))  ## torch.relu是一个函数，nn.ReLU是一个nn.Module
        x = self.pool(x)
        # 2.卷积池化
        x = torch.relu(self.conv2(x))  ## 同样用函数式relu激活
        x = self.pool(x)  ## 池化下采样
        # 3.卷积池化
        x = torch.relu(self.conv3(x))  ## 第三次卷积提取更深层特征
        x = self.pool(x)  ## 第三次池化，完成编码

        # 解码过程
        x = torch.relu(self.t_conv1(x))  ## 转置卷积上采样，恢复尺寸
        x = torch.relu(self.t_conv2(x))  ## 继续上采样
        x = torch.relu(self.t_conv3(x))  ## 最后一次上采样

        # 最后一个普通卷积
        x = torch.sigmoid(self.conv_out(x))  ## sigmoid将输出映射到[0,1]
        return x


if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor = torch.randn(100, 3, 64, 64)
    # 创建一个卷积编码器模型
    denoiser = ConvDenoiser()

    # 编码输入张量
    encoded_tensor = denoiser(input_tensor).shape
    print(encoded_tensor)
