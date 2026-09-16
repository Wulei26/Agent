"""
（3）小批量梯度下降（Mini-batch Gradient Descent，MBGD）
每次迭代使用一小批样本（如32、64个）计算梯度。
平衡了BGD的稳定性和SGD的速度，是最常用的方法。
优点：速度快，适合大规模数据，梯度更新方向相对稳定。
"""
# 演示 小批量梯度下降法求解线性回归问题
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
def gradient(beta:np.array, batch_indices:np.array):
    return (2/len(batch_indices)) * X[batch_indices].T @ (X[batch_indices] @ beta - y[batch_indices])   

# 定义学习率和迭代次数
alpha = 0.01
num_iterations = 10000
batch_size = 3  # 小批量的大小
# 初始化参数
beta = np.zeros((2, 1))     

for i in range(num_iterations):
    # 随机选择一个小批量样本
    batch_indices = np.random.choice(n, batch_size, replace=False)
    # 计算梯度
    grad = gradient(beta, batch_indices)
    # 更新参数
    beta -= alpha * grad
    # 打印一下当前beta
    if i % 100 == 0:
        print(f"Iteration {i}: beta = {beta.ravel()}, Loss = {J(beta)}")    
