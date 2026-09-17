import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. 读取数据
# ============================================================

digit = pd.read_csv(
    "/storage/data/尚硅谷ai/08_尚硅谷AI大模型之机器学习/2.资料/data/train.csv"
)

"""
# 看下第1行的数字是什么
plt.imshow(
    digit.iloc[1, 1:].values.reshape(28, 28),
    cmap="gray"
)
plt.show()
"""


# ============================================================
# 2. 划分训练集和测试集
# ============================================================

X = digit.drop("label", axis=1).values
y = digit["label"].values

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 3. 归一化
# ============================================================

preprocessor = MinMaxScaler()

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)


# ============================================================
# 4. One-Hot 编码
# ============================================================

def one_hot(y, num_classes=10):
    """
    将类别标签转换成 One-Hot。

    例如：

        y = 3

    转换成：

        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    输入：
        y.shape = (batch_size,)

    返回：
        (batch_size, 10)
    """

    result = np.zeros(
        (len(y), num_classes)
    )

    result[
        np.arange(len(y)),
        y
    ] = 1

    return result


y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)


# ============================================================
# 5. 增加偏置项
# ============================================================

"""
原来的：

x_train.shape
    (batch_size, 784)

增加一列 1：

x_train.shape
    (batch_size, 785)

最后一列就是 bias 对应的特征。
"""

x_train = np.hstack(
    [
        x_train,
        np.ones(
            (len(x_train), 1)
        )
    ]
)

x_test = np.hstack(
    [
        x_test,
        np.ones(
            (len(x_test), 1)
        )
    ]
)


# ============================================================
# 6. Softmax
# ============================================================

def softmax(z):
    """
    Softmax 函数。

    z.shape:
        (batch_size, 10)

    返回：
        (batch_size, 10)
    """

    # 防止 exp(z) 数值溢出
    z = z - np.max(
        z,
        axis=1,
        keepdims=True
    )

    exp_z = np.exp(z)

    return (
        exp_z
        / np.sum(
            exp_z,
            axis=1,
            keepdims=True
        )
    )


# ============================================================
# 7. 定义损失函数
# ============================================================

def loss(beta, X_batch, y_batch):
    """
    Cross Entropy Loss。

    beta:
        (785, 10)

    X_batch:
        (batch_size, 785)

    y_batch:
        (batch_size, 10)
    """

    # 预测
    z = X_batch @ beta

    # Softmax
    y_pred = softmax(z)

    # 防止 log(0)
    eps = 1e-15

    y_pred = np.clip(
        y_pred,
        eps,
        1 - eps
    )

    # Cross Entropy
    loss_value = -(
        1 / len(X_batch)
    ) * np.sum(
        y_batch * np.log(y_pred)
    )

    return loss_value


# ============================================================
# 8. 计算梯度
# ============================================================

def gradient(beta, x_batch, y_batch):
    """
    计算 Softmax Regression 的梯度。

    数学公式：

        Z = X @ beta

        Y_hat = Softmax(Z)

        gradient =
            1/n * X.T @ (Y_hat - Y)

    beta:
        (785, 10)

    x_batch:
        (batch_size, 785)

    y_batch:
        (batch_size, 10)

    返回：

        gradient:
        (785, 10)
    """

    # 当前 batch 的样本数量
    n = len(x_batch)

    # 前向传播
    z = x_batch @ beta

    # Softmax
    y_pred = softmax(z)

    # 梯度
    grad = (
        (1 / n)
        * x_batch.T
        @ (y_pred - y_batch)
    )

    return grad


# ============================================================
# 9. 预测
# ============================================================

def predict(beta, X):
    """
    根据模型参数预测数字。

    X:
        (batch_size, 785)

    返回：
        (batch_size,)
    """

    # 计算 10 个类别的得分
    z = X @ beta

    # 转换成概率
    y_pred = softmax(z)

    # 找到概率最大的类别
    return np.argmax(
        y_pred,
        axis=1
    )


# ============================================================
# 10. 计算准确率
# ============================================================

def accuracy(beta, X, y):
    """
    计算分类准确率。
    """

    y_pred = predict(
        beta,
        X
    )

    return np.mean(
        y_pred == y
    )


# ============================================================
# 11. 初始化模型参数
# ============================================================

n_features = x_train.shape[1]
n_classes = 10

"""
x_train:

(batch_size, 785)

所以：

beta:

(785, 10)
"""

beta = np.random.randn(
    n_features,
    n_classes
) * 0.01


# ============================================================
# 12. Mini-Batch Gradient Descent
# ============================================================

learning_rate = 0.1
epochs = 100
batch_size = 64

n_samples = len(x_train)

loss_history = []


for epoch in range(epochs):

    # --------------------------------------------------------
    # 每一个 epoch 随机打乱数据
    # --------------------------------------------------------

    indices = np.random.permutation(
        n_samples
    )

    x_train = x_train[indices]

    y_train_onehot = y_train_onehot[indices]

    y_train = y_train[indices]


    # --------------------------------------------------------
    # 保存当前 epoch 的总 Loss
    # --------------------------------------------------------

    epoch_loss = 0


    # --------------------------------------------------------
    # Mini-Batch
    # --------------------------------------------------------

    for start in range(
        0,
        n_samples,
        batch_size
    ):

        end = start + batch_size

        x_batch = x_train[start:end]

        y_batch = y_train_onehot[start:end]


        # ----------------------------------------------------
        # 计算梯度
        # ----------------------------------------------------

        grad = gradient(
            beta,
            x_batch,
            y_batch
        )


        # ----------------------------------------------------
        # 更新参数
        # ----------------------------------------------------

        beta -= (
            learning_rate
            * grad
        )


        # ----------------------------------------------------
        # 计算当前 batch Loss
        # ----------------------------------------------------

        batch_loss = loss(
            beta,
            x_batch,
            y_batch
        )

        epoch_loss += (
            batch_loss
            * len(x_batch)
        )


    # --------------------------------------------------------
    # 当前 epoch 平均 Loss
    # --------------------------------------------------------

    epoch_loss /= n_samples

    loss_history.append(
        epoch_loss
    )


    # --------------------------------------------------------
    # 计算训练集准确率
    # --------------------------------------------------------

    train_acc = accuracy(
        beta,
        x_train,
        y_train
    )

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f}"
        )


# ============================================================
# 13. 测试集准确率
# ============================================================

test_acc = accuracy(
    beta,
    x_test,
    y_test
)

print()
print(
    f"Test Accuracy: {test_acc:.4f}"
)


# ============================================================
# 14. 随机查看一张测试图片
# ============================================================

index = 123

prediction = predict(
    beta,
    x_test[index:index + 1]
)[0]

true_label = y_test[index]


# 去掉最后的 bias
image = x_test[
    index,
    :-1
]


plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"True: {true_label}, "
    f"Predict: {prediction}"
)

plt.axis("off")

plt.show()


# ============================================================
# 15. 绘制 Loss 曲线
# ============================================================

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training Loss"
)

plt.show()