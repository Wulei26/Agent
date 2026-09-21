# 使用反向传播对数字识别进行优化
import numpy as np
from collections import OrderedDict
from common import Affine, ReLU, SoftmaxWithLoss, get_data


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
        # 生成层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """x :输入数据； t: 正确标签"""
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 向前传播
        self.loss(x, t)
        # 反向传播
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db
        return grads
