import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NDArrayFloat = npt.NDArray[np.floating]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray):  # 这里的x是二维数组
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y


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


def _numerical_gradient(
    f: callable,
    x: NDArrayFloat,
) -> NDArrayFloat:
    """
    对一维向量 x 计算数值梯度。

    Parameters
    ----------
    f : callable
        目标函数。
        输入：一维 NumPy 数组
        输出：标量 float

    x : NDArrayFloat
        一维参数向量，例如:
        shape = (2,)

    Returns
    -------
    NDArrayFloat
        梯度向量，与 x 形状相同。
        shape = (2,)
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
    return grad


def numberical_gradient(
    f: callable,
    X: NDArrayFloat,
) -> NDArrayFloat:
    """
    对一维向量或者二维矩阵计算数值梯度。

    X.ndim == 1:
        X 是一个向量
        shape = (n,)

    X.ndim == 2:
        X 是一个矩阵
        shape = (m, n)

    返回的梯度 grad 与 X 形状完全相同。
    """
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad


# 梯度下降法
def gradient_descent(
    f: callable,
    init_x: NDArrayFloat,
    learning_rate: float = 0.01,
    iter_num: int = 1000,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    x = init_x.copy()

    x_history: list[NDArrayFloat] = []
    x_history.append(x.copy())
    # 循环迭代
    for i in range(iter_num):
        # 计算当前 init_x点处的梯度向量
        grad = numberical_gradient(f, x)
        x -= learning_rate * grad
        x_history.append(x.copy())
    return x, np.array(x_history)


def get_data():
    # 加载数据集
    data = pd.read_csv(r"D:\BaiduNetdiskDownload\DL\2.资料\data\train.csv")
    # 划分训练集和测试集
    X = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test, y_train.values, y_test.values
