__all__ = ["ImageClassification"]

import torch.nn as nn
import torch.nn.functional as F

class ImageClassification(nn.Module):

    def __init__(self, num_classes = 5):
        super().__init__()
         # 定义第一个卷积层，输入通道数为3，输出通道数为8，卷积核大小为3x3，步幅为1，填充为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        # 使用填充为1的卷积操作，输出特征图的尺寸与输入相同（Same convolutions）

        # 定义最大池化层，池化核大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层，池化核大小为2x2，步幅为2
        # 定义第二个卷积层，输入通道数为8，输出通道数为16，卷积核大小为3x3，步幅为1，填充为1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 上一层的输出通道数为8，因此这一层的输入通道数为8
        # 定义全连接层，输入大小为16*16*16，输出大小为num_classes（分类数），全连接层的输入的尺寸计算在forward函数中解释
        self.fc1 = nn.Linear(16*16*16, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一个卷积层后接ReLU激活函数
        x = self.pool(x)  # 第一个池化层
        x = F.relu(self.conv2(x))  # 第二个卷积层后接ReLU激活函数
        x = self.pool(x)  # 第二个池化层
        x = x.reshape(x.size(0), -1)  # 将特征图展平为一维向量，准备输入全连接层
        x = self.fc1(x)  # 全连接层输出分类结果
        return x  # 返回最终输出

if __name__ == "__main__":
    import torch
    x = torch.randn(1, 3, 64, 64)  # 创建一个随机输入张量，模拟一张64x64的RGB图像
    model = ImageClassification(num_classes=5)  # 创建模型实例，指定分类数为5
    output = model(x)  # 前向传播，获取模型输出
    print(output.shape)  # 打印输出张量的形状，应该为[1, 5]，表示1个样本的5个分类结果