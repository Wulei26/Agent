"""
（2）随机梯度下降（Stochastic Gradient Descent，SGD）
每次迭代随机选取一个样本计算梯度。
优点：速度快，适合大规模数据。
缺点：梯度更新方向不稳定，优化过程震荡较大，可能难以收敛。
"""
# 演示 随机梯度下降法求解线性回归问题
import numpy as np
from sklearn.model_selection import train_test_split
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
# 因变量，数学考试
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]
X = np.array(X)
y = np.array(y).reshape(-1, 1)
n = X.shape[0]  # 样本数量
X = np.hstack((np.ones((n, 1)), X))  # 添加偏置项

# 定义损失函数
def J(beta:np.array): #beta  的维度是 (2, 1)
    return np.mean((X @ beta - y) ** 2)
# 定义梯度下降函数
def gradient(beta:np.array, i:int):
    return 2 * X[i:i+1].T @ (X[i:i+1] @ beta - y[i:i+1])

# 定义学习率和迭代次数
alpha = 0.01
num_iterations = 10000
# 初始化参数
beta = np.zeros((2, 1))

for i in range(num_iterations):
    # 随机选择一个样本
    idx = np.random.randint(0, n)
    # 计算梯度
    grad = gradient(beta, idx)
    # 更新参数
    beta -= alpha * grad
    # 打印一下当前beta
    if i % 100 == 0:
        print(f"Iteration {i}: beta = {beta.ravel()}, Loss = {J(beta)}")    

# 使用API
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(
    loss="squared_error", # 损失函数
    fit_intercept=True, #是否计算截距
    learning_rate="constant", # 学习率策略,常数学习率
    eta0=0.01, # 学习率
    max_iter=10000, # 最大迭代次数
    tol=1e-3, # 收敛阈值
)
sgd_reg.fit(X, y.ravel())
print(f"API Result: beta = {sgd_reg.coef_}, intercept = {sgd_reg.intercept_}")