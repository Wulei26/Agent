"""
（1）批量梯度下降（Batch Gradient Descent，BGD）
每次迭代使用全部训练数据计算梯度。
优点：稳定收敛。
缺点：计算开销大。
"""
# 演示 批量梯度下降法求解线性回归问题
import numpy as np
from sklearn.model_selection import train_test_split
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
# 因变量，数学考试成绩
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]

X = np.array(X)
y = np.array(y).reshape(-1, 1)
n = X.shape[0]  # 样本数量
X = np.hstack((np.ones((n, 1)), X))  # 添加偏置项

# 定义损失函数
def J(beta:np.array): #beta  的维度是 (2, 1)
    return np.mean((X @ beta - y) ** 2)
# 定义梯度下降函数
def gradient(beta:np.array):
    return (2/n) * X.T @ (X @ beta - y)
# 定义学习率和迭代次数
alpha = 0.01
num_iterations = 10000
# 初始化参数
beta = np.zeros((2, 1))

for i in range(num_iterations):
    # 计算梯度
    grad = gradient(beta)
    # 更新参数
    beta -= alpha * grad
    # 打印一下当前beta
    if i % 100 == 0:
        print(f"Iteration {i}: beta = {beta.ravel()}, Loss = {J(beta)}")
    