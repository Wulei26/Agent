import numpy as np
import matplotlib.pyplot as plt
from TwoLayerNet import TwoLayerNet
from common import get_data

x_train, x_test, y_train, y_test = get_data()

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
epochs = 10
train_size = x_train.shape[0]  # 训练集样本总数
batch_size = 100
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = np.ceil(train_size / batch_size)
# 总迭代次数 = epoch * iter_per_epoch
iterations = int(epochs * iter_per_epoch)
# 外层循环：epoch
for epoch in range(epochs):
    """
    1）随机选择批数据（mini-batch）
    从训练数据中随机选出一部分数据，学习的目标就是要减少这个mini-batch数据的损失函数值。
    2）计算梯度
    对当前的各权重参数，计算出梯度的值，负梯度就表示了损失函数减小最多的方向。
    3）更新参数
    按照3.4.1节中梯度下降法的公式，对权重参数沿负梯度方向进行微小更新。
    4）重复迭代
    重复上面的步骤1）2）3），直到完成预定的总迭代次数。
    """
    # 内层循环：一个 epoch 中的 mini-batch
    for i in range(iter_per_epoch):
        # 1.随机选取一个minibatch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = y_train[batch_mask]
        # 2.计算梯度
        grads = network.gradient(x_batch, t_batch)
        # 3.更新参数，分维度更新
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grads[key]
        # 4.计算当前batch的loss
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    # 一个 epoch 完成后计算 accuracy
    train_acc = network.acuracy(x_train, y_train)
    test_acc = network.acuracy(x_test, y_test)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(f"epoch {epoch + 1}/{epochs}, " f"train acc: {train_acc}, " f"test acc: {test_acc}")

# 绘制图形
markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
