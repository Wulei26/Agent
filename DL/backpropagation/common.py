import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ReLU:

    def __init__(self) -> None:
        # 用属性记录哪些输入小于0（布尔掩码）
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 记录输入 <= 0 的位置
        self.mask = x <= 0
        out = x.copy()
        # 大于0保留本身，小于等于0置为0
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout 是从上一层传递过来的梯度
        dout[self.mask] = 0
        dx = dout.copy()
        return dx


import numpy as np


class Sigmoid:

    def __init__(self) -> None:
        # 前向输出，初始为 None
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out
        return dx


import numpy as np


class Affine:

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        """仿射层（Affine / 全连接层）：Y = XW + b

        Parameters
        ----------
        W : np.ndarray
            权重矩阵，形状 (D, M)。
        b : np.ndarray
            偏置向量，形状 (M,)。
        """
        self.W: np.ndarray = W
        self.b: np.ndarray = b

        # 保存输入的 x（已展平为二维）
        self.X: np.ndarray | None = None

        # 反向传播时计算的梯度
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # 记录输入原始形状，考虑输入可能是三维甚至多维的情况
        self.original_x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : np.ndarray
            前一层的输出，形状任意（第一维为 batch 维）。

        Returns
        -------
        np.ndarray
            线性变换结果，形状 (N, M)。
        """
        self.original_x_shape = x.shape
        # 将输入展平为二维 (N, D) 再计算
        x = x.reshape(x.shape[0], -1)
        self.X = x
        Y = np.dot(self.X, self.W) + self.b
        return Y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """反向传播。

        Parameters
        ----------
        dy : np.ndarray
            上游传来的梯度，形状 (N, M)。

        Returns
        -------
        np.ndarray
            对输入 x 的梯度，形状与 forward 的输入 x 相同。
        """
        # 对输入 x 的梯度：∂L/∂X = ∂L/∂Y · Wᵀ
        dx = np.dot(dy, self.W.T)

        # 对权重 W 的梯度：∂L/∂W = Xᵀ · ∂L/∂Y
        self.dW = self.X.T @ dy

        # 对偏置 b 的梯度：∂L/∂b = Σ_i ∂L/∂Y_i
        #
        # 前向时 b 被广播到了 N 个样本上，每个样本都对 b 有贡献；
        # 反向时按链式法则，b 的梯度就是这 N 份贡献的累加，
        # 所以要对 dy 沿样本维（axis=0）求和，
        # 得到与 b 同形状的 (M,) 向量。
        self.db = np.sum(dy, axis=0)

        # 解包恢复原来的维度
        return dx.reshape(*self.original_x_shape)


def get_data():
    # 加载数据集
    data = pd.read_csv("/storage/data/尚硅谷ai/09_尚硅谷AI大模型之深度学习/2.资料/data/train.csv")
    # 划分训练集和测试集
    X = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test, y_train.values, y_test.values


class SoftmaxWithLoss:
    @staticmethod
    def softmax(x: np.ndarray):  # 这里的x是二维数组
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return y

    @staticmethod
    def cross_entropy_error(y: np.ndarray, t: np.ndarray):

        if y.ndim == 1:  # 如果是输入一个样本，那么还是将其转化为二维矩阵
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        if t.size == y.size:  # 也就是说经过了独热编码
            # 转换
            t = t.argmax(axis=1)  # 返回的是最大值所在的索引（下标）
        batch_size = t.shape[0]
        loss_sum = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))
        return loss_sum / batch_size

    def __init__(self):
        self.loss = None
        self.y = None  # 这是softmax的输出
        self.t = None  ##这是正确的标签，这里采用独热编码

    def forward(
        self,
        x: np.ndarray,
        t: np.ndarray,
    ):
        self.t = t
        self.y = __class__.softmax(x)
        self.loss = __class__.cross_entropy_error(self.y, t)
        return self.loss

    def backward(
        self,
        dout=1.0,  # 这个dout 是上一层传递过来的梯度，但是我们这里是把SoftmaxWithLoss作为一个整体，所以上一层传递过来的梯度就是DL/DL = 1
    ):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            # t 是One-hot
            dx = (self.y - self.t) / batch_size
        else:
            # t如果是类别下标,这里其实就是巧妙的将类别下标转化为了One-hot,直接用类别作为下标去取y中的值，对应的就是正确标签，如果是独热编码，那么t这个位置的值就是1,  所以这里才会减去1。非常巧妙
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class SGD:

    def __init__(self, lr: float):
        self.lr = lr

    def update(
        self,
        params: dict,  # 保存参数的字典
        grad: dict,  # 保存梯度的字典
    ):
        # 参数更新策略 w = w - lr * dw
        for key in params.keys():
            params[key] = params[key] - self.lr * grad[key]


class Momentum:
    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0.9,
    ):
        self.lr = lr
        self.beta = beta
        self.v: dict = None  # 记录历史梯度

    def update(
        self,
        parms: dict,  # 当前参数
        grads: dict,  # 当前梯度
    ):
        # v = β*v + grad
        # W← W + v
        # 初始状态（历史梯度为0）
        if self.v is None:
            self.v = {}
            for key, val in parms.items():
                self.v[key] = np.zeros_like(val)
        for key in parms.keys():
            # 更新历史梯度
            self.v[key] = self.beta * self.v[key] - self.lr * grads[key]
            # 更新参数
            parms[key] += self.v[key]

class AdaGrad:
    def __init__(self,beta:0.01):
        self.beta = beta
        self.h:dict = None #记录的是历史梯度的平方和
    def update(self,
        params:dict, #当前的参数
        grad:dict, #当前的参数梯度
    ):
        if self.h is None:
            # 初始化
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            # 加上当前梯度的平方和
            self.h[key] = grad[key] * grad[key] + self.h[key]
            # 更新参数
            params[key] -= self.beta*grad[key] / (np.sqrt(self.h[key]) + 1e-7)
            ##随着学习的进行 h会越来越大，那么更新的幅度就会越来越小，也就是变相的把学习率减小了

class RMSProp:
    
    def __init__(self,
        alpha:float, ##衰减系数
        lr:float,
    ):
        self.alpha = alpha
        self.lr = lr
        self.h:dict = None #记录的是历史梯度的平方和
    def update(self,
        params:dict, #当前的参数
        grad:dict, #当前的参数梯度
    ):
        if self.h is None:
            # 初始化
            self.h = {}
            for key, value in grad.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            # 更新历史梯度
            self.h[key] *= self.alpha 
            self.h[key] += (1-self.alpha) * grad[key] **2
            # 更新参数
            params[key] -= self.lr*grad[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:

    def __init__(self,
        alpha1:float = 0.9,
        alpha2:float = 0.999,
        lr:float = 0.01,
     ):
        self.alpha1 = alpha1 #一次动量系数
        self.alpha2 = alpha2 # 二次动量系数
        self.lr = lr #学习率
        self.v:dict = None # 当前的v
        self.h:dict = None # 当前h
        self.i:int = 0 #当前的迭代次数
    
    def update(self,
        params:dict, #当前的参数
        grads:dict, #当前的参数梯度
    ):
        if self.v is None or self.h is None:
            # 初始化
            self.v = {}
            self.h = {}
            for key, value in grads.items():
                self.v[key] = np.zeros_like(value)
                self.h[key] = np.zeros_like(value)
        self.i += 1 #注意这是总的迭代次数，不是每个epoch的迭代次数
        for key, value in params.items():
            # 更新v
            self.v[key] *= self.alpha1
            self.v[key] += (1 - self.alpha1) * grads[key]
            # 更新h
            self.h[key] *= self.alpha2
            self.h[key] += (1 - self.alpha2) * grads[key] ** 2
            # 更新参数
            v_hat = self.v[key] / (1 - self.alpha1 ** self.i)
            h_hat = self.h[key] / (1 - self.alpha2 ** self.i)
            params[key] -= self.lr * v_hat / (np.sqrt(h_hat) + 1e-7)