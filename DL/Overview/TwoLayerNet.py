"""
实现一个TwoLayerNet类，用于实现手写数字识别，使用SGD进行学习
一个2层的神经网络（中间只有1个隐藏层、后面是输出层） 使用softmax进行输出，
"""

import numpy as np
from common import numberical_gradient, sigmoid, softmax, cross_entropy_error


class TwoLayerNet:
    def __init__(
        self,
        input_size: int,  # 输入特征数目
        hidden_size: int,  # 隐藏层数目
        output_size: int,  # 输出特征
        weight_init_std: float = 0.01,  # 初始化系数
    ):
        # 初始化输入曾的系数和偏置
        self.params = {}
        self.params["W1"] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params["b2"] = np.zeros(output_size)

    def forward(self, x: np.ndarray):
        """输入矩阵(batch_size, x_features), 得到预测输出y"""
        w1, w2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        y = softmax(z2)
        return y

    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(y, t)

    def acuracy(self, x, t):
        """计算输出y中预测对的标签占总标签的比例"""
        y = self.forward(x)
        # 这里的y是经过读热编码的，进行转换
        # 将预测概率分布转化为预测分类号
        y = np.argmax(y, axis=1)
        t = t.reshape(-1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        """计算当前x对应的梯度
        注意这个损失函数一定是关于W1,W2,b1,b2的复合函数
        所以分别计算W1,W2,b1,b2的梯度
        """
        loss_w = lambda _: self.loss(x, t)
        grads = {}
        grads["W1"] = numberical_gradient(loss_w, self.params["W1"])
        grads["b1"] = numberical_gradient(loss_w, self.params["b1"])
        grads["W2"] = numberical_gradient(loss_w, self.params["W2"])
        grads["b2"] = numberical_gradient(loss_w, self.params["b2"])
        return grads
