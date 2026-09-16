import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

digit = pd.read_csv(r"D:\BaiduNetdiskDownload\2.资料\data\train.csv")
""" 看下第1行的数字是什么
plt.imshow(digit.iloc[1, 1:].values.reshape(28, 28), cmap="gray")
plt.show()
"""
# 划分训练集和测试级
X = digit.drop("label", axis=1)
y = digit["label"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##进行归一化
preprocessor = MinMaxScaler()
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

# 模型训练
model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)
print(model.coef_.shape)
# 评估
print(model.score(x_test, y_test))
# 预测
plt.imshow(digit.iloc[123, 1:].values.reshape(28, 28), cmap="gray")
plt.show()
print(model.predict(digit.iloc[123, 1:].values.reshape(1, -1)))
