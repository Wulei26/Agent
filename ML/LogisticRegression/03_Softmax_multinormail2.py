import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#  1. 加载数据集
digit = pd.read_csv(r"D:\BaiduNetdiskDownload\2.资料\data\train.csv")
X = digit.drop("label", axis=1).values
y = digit["label"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 特征转换 X 经过归一化处理， y通过读热编码,给x_train,x_test 添加偏置项
# 2. 特征转换

MinMax_scaler = MinMaxScaler()

x_train = MinMax_scaler.fit_transform(x_train)
x_test = MinMax_scaler.transform(x_test)

# One-Hot 编码
OneHot_scaler = OneHotEncoder(sparse_output=False)

y_train_onehot = OneHot_scaler.fit_transform(y_train.reshape(-1, 1))

y_test_onehot = OneHot_scaler.transform(y_test.reshape(-1, 1))

print(x_train.shape)
print(y_train.shape)
print(y_train_onehot.shape)

print(x_test.shape)
print(y_test.shape)
print(y_test_onehot.shape)


# 3.定义Softmax函数，损失函数，梯度函数
def Softmax(z: np.ndarray):
    """
    Softmax 函数。

    z.shape:
        (batch_size, 10)

    返回：
        (batch_size, 10)
    """
    # 这里的z是一个概率分布，表示batch_size个样本，各自取0-9的得分
    # 为了防止溢出，那么对每个得分进行处理，这里选择减去各自行的最大值
    # 那么每个样本在 0-9 上面的得分最大值就变成了0,这样取指数就不会溢出
    # 为什么这样可以？因为分子分母都是指数，分子分母同时除以e^c,不会改变这个分数的大小
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    # 然后再求和，也就是每个样本得分除以在 0-9 得分的总和，这样就把样本在 0-9 上面的得分，转化为取 0-9 的概率
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    # keep_dims 在运算压缩数组维度后，计算后的结果保持维度不变，关键用途是保证形状对齐，让广播正确工作。


def loss(beta: np.ndarray, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
    """
    Cross Entropy Loss。

    beta:
        (785, 10)

    X_batch:
        (batch_size, 785)

    y_batch:
        (batch_size, 10)
    """
    y_pred = Softmax(x_batch @ beta)  # (batch_size * 10)
    # 这里面表示的是概率，有可能非常小，所以为了避免log0,把最小猪转化为 1e-15
    eps = 1e-15
    ## np.clip(a, a_min, a_max, out=None) 小于a_min转化为a_min,大于a_max转化为a_max
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # 因为这里的y_batch 采用的是独热编码，所以可以显示地替代示性函数
    #  One-Hot 标签本身就是一个“指示函数矩阵”
    loss_value = -(1 / x_batch.shape[0]) * np.sum(y_batch * np.log(y_pred))
    return loss_value


def gradient(beta: np.ndarray, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """
    Z = X @ beta

    Y_hat = Softmax(Z)

    gradient =
        1/n * X.T @ (Y_hat - Y)
    """
    # 当前 batch 的样本数量
    n = len(x_batch)

    # 前向传播
    z = x_batch @ beta

    # Softmax
    y_pred = Softmax(z)

    # 梯度
    grad = (1 / n) * x_batch.T @ (y_pred - y_batch)

    return grad


def predict(beta, x_batch):
    """
    根据模型参数预测数字。

    X:
        (batch_size, 785)

    返回：
        (batch_size,)
    """
    y_pred = Softmax(x_batch @ beta)
    return np.argmax(y_pred, axis=1)


def accuracy(beta, X, y):
    """
    计算分类准确率。
    """

    y_pred = predict(beta, X)

    return np.mean(y_pred == y)


# 初始化模型参数
n_features = x_train.shape[1]
n_classes = 10
beta = np.random.randn(n_features, n_classes) * 0.01
learning_rate = 0.1
epochs = 100
batch_size = 64
n_samples = len(x_train)

loss_history = []

for epoch in range(epochs):

    # 每个 epoch 随机打乱训练集
    indices = np.random.permutation(n_samples)

    x_train_shuffled = x_train[indices]
    y_train_onehot_shuffled = y_train_onehot[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0

    for start in range(0, n_samples, batch_size):
        end = start + batch_size

        x_batch = x_train_shuffled[start:end]
        y_batch = y_train_onehot_shuffled[start:end]

        grad = gradient(beta, x_batch, y_batch)

        beta -= learning_rate * grad

        batch_loss = loss(beta, x_batch, y_batch)

        epoch_loss += batch_loss * len(x_batch)

    epoch_loss /= n_samples

    loss_history.append(epoch_loss)

    train_acc = accuracy(beta, x_train, y_train)

# 测试集上的准确率
test_acc = accuracy(beta, x_test, y_test)

print()
print(f"Test Accuracy: {test_acc:.4f}")

# plt.plot(range(1, epochs + 1), loss_history)

# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.title("Training Loss")

# plt.show()
