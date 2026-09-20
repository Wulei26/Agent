import numpy as np
from common import gradient_descent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 定义目标函数，一个二元函数
    def f(x):
        # 定义一个f(x) = x1^2 + x2^2
        return x[0] ** 2 + x[1] ** 2

    # 定义初始点
    init_x = np.array([-3, 4], dtype=float)
    # 定义超参数
    learning_rate = 0.05
    iterations = 200
    # 梯度下降求解最小值点
    x, x_history = gradient_descent(f, init_x, learning_rate, iterations)
    plt.scatter(x_history[:, 0], x_history[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    # print(x_history)
    # print(x)

# 定义一个类，构建一个神经网络，从而实现手写数字识别
# 构建一个两层的神经网络
