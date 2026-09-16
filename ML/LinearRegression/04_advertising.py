import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # 均方误差

data = pd.read_csv("/storage/data/尚硅谷ai/08_尚硅谷AI大模型之机器学习/2.资料/data/advertising.csv")
data = data.drop(data.columns[0], axis=1)  # 删除第一列
data = data.dropna()  # 删除缺失值
print(data.head())
print(data.shape)

X = data.drop("Sales", axis=1)
y = data["Sales"]
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义损失函数
def J(beta: np.array):  # beta  的维度是 (4, 1)
    return np.mean((X @ beta - y) ** 2)

# 定义梯度下降函数
def gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray):
    n = X.shape[0]
    return (2 / n) * X.T @ (X @ beta - y)

# 定义学习率和迭代次数
alpha = 0.01
num_iterations = 10000
# 初始化参数
beta = np.zeros((4, 1)) # 全为0的4行1列矩阵

# 对数据进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转为 numpy 矩阵（n_samples, n_features），并给 X 加一列偏置项 x0 = 1
X_mat = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]  # (n, 4)
y_vec = y_train.to_numpy().reshape(-1, 1)   
# 使用小批量梯度下降
batch_size = 32
num_samples = X_mat.shape[0]

for i in range(num_iterations):
    # 随机选取一个批次的索引
    batch_indices = np.random.choice(num_samples, size=batch_size, replace=False)

    # 取出对应批次的数据
    X_batch = X_mat[batch_indices]   # (batch_size, 4)
    y_batch = y_vec[batch_indices]   # (batch_size, 1)

    # 计算梯度并更新参数
    grad = gradient(beta, X_batch, y_batch)
    beta = beta - alpha * grad

# 输出训练后的参数
print("训练后的 beta:\n", beta)

# 在测试集上评估
X_test_mat = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
y_test_vec = y_test.to_numpy().reshape(-1, 1)
y_pred = X_test_mat @ beta
mse = np.mean((y_pred - y_test_vec) ** 2)
print("测试集 MSE:", mse)

# 使用API
from sklearn.linear_model import LinearRegression, SGDRegressor
normal_equation = LinearRegression()
normal_equation.fit(X_train_scaled, y_train)
print("正规方程法解得模型系数:", normal_equation.coef_)
print("正规方程法解得模型偏置:", normal_equation.intercept_)

# 使用随机梯度下降拟合
sgd_model = SGDRegressor()
sgd_model.fit(X_train_scaled,y_train)
print("随机梯度下降法解得模型系数:", sgd_model.coef_)
print("随机梯度下降法解得模型偏置:", sgd_model.intercept_)

print("###################################################")
# 使用均方误差评估三种方法
print("小批量梯度下降法均方误差:", mean_squared_error(y_test, X_test_mat @ beta))
print("正规方程法均方误差:", mean_squared_error(y_test, normal_equation.predict(X_test_scaled)))
print("随机梯度下降法均方误差:", mean_squared_error(y_test, sgd_model.predict(X_test_scaled)))