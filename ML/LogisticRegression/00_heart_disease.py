import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

# 加载数据集
heart_disease = pd.read_csv(r"D:\BaiduNetdiskDownload\2.资料\data\heart_disease.csv")
heart_disease.dropna()

# 划分为训练集与测试集
X = heart_disease.drop("是否患有心脏病", axis=1)  # 特征
y = heart_disease["是否患有心脏病"]  # 标签
x_train_mat, x_test_mat, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 特征工程
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]
# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        # 对数值型特征进行标准化
        ("num", StandardScaler(), numerical_features),
        # 对类别型特征进行独热编码，使用drop="first"避免多重共线性
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        # 二元特征不进行处理
        ("binary", "passthrough", binary_features),
    ]
)
# 执行特征转换
x_train = preprocessor.fit_transform(x_train_mat)  # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(x_test_mat)  # 使用训练集计算的信息对测试集进行转换
print(x_train.shape)  # (717, 19)
print(y_train.shape)  # (717,)
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])  # 加上偏置项
print(x_train.shape)  # (717, 19)
# 将y也转成numpy数组
y_train = y_train.to_numpy().reshape(-1, 1)  # (n, 1)
y_test = y_test.to_numpy().reshape(-1, 1)
n = x_train.shape[0]


def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))


# 定义损失函数
def J(beta: np.array, x_batch: np.array, y_batch: np.array):  # bata是一个 （20,1）的array
    p = sigmod(x_batch @ beta)
    eps = 1e-15
    p = np.clip(
        p, eps, 1 - eps
    )  # 这个是防止p过小导致取logp 变成无穷小了,np.clip(x, min, max) = 把 x 卡在 [min, max] 区间里，常用于防止除零、log(0)、数值溢出等问题。
    # 逐元素相乘后求和，得到标量
    logL = np.sum(y_batch * np.log(p) + (1 - y_batch) * np.log(1 - p))
    return -logL / n


# 定义梯度函数
def gradient(beta, x_batch, y_batch):
    return (1 / n) * x_batch.T @ (sigmod(x_batch @ beta) - y_batch)


# 初始化beta矩阵
beta = np.ones((x_train.shape[1], 1))
alpha = 0.01
max_iter = 10000
batch_size = 128
for i in range(max_iter):
    # 从 x_train中随机取一个batch_size
    idx = np.random.choice(n, batch_size, replace=False)  # 不放回抽样
    x_batch = x_train[idx]  # (32, 20)
    y_batch = y_train[idx]
    # 计算损失
    loss = J(beta, x_batch, y_batch)
    # 计算梯度
    grad = gradient(beta, x_batch, y_batch)
    if (i + 1) % 1000 == 0:
        print(f"f第{i+1}迭代, 参数beta 为: {beta}, MSE为: {mean_squared_error(y_train[idx],x_batch @ beta)}")
    # 更新beta
    beta = beta - alpha * grad

# 测试集上评估
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
print("小批量梯度下降法均方误差:", mean_squared_error(y_test, x_test @ beta))


# 使用API
# 执行特征转换
x_train = preprocessor.fit_transform(x_train_mat)  # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(x_test_mat)  # 使用训练集计算的信息对测试集进行转换

# 模型训练
model = LogisticRegression()
model.fit(x_train, y_train)

# 模型评估，计算准确率
print("使用API均方误差:", mean_squared_error(y_test, model.predict(x_test)))
