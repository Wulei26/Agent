import pandas as pd
import numpy as np
import joblib  # 用于加载已经训练好的参数
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Tanh(x: np.ndarray):
    exp_x = np.exp(-2 * x)
    return (1 - exp_x) / (1 + exp_x)


def ReLU(x: np.ndarray):
    return np.maximum(0, x)

def leakyReLU(x:np.ndarray, alpha:float = 0.01):
    return np.maximum(alpha * x ,x)
    

# def softmax(x: np.ndarray):  # 这里的x是二维数组
#     x = x.T
#     x = x - np.max(x, axis=0)
#     exp_x = np.exp(x)
#     y = exp_x / np.sum(exp_x, axis=0)
#     return y.T


def softmax(x: np.ndarray):  # 这里的x是二维数组
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y


"""
文件train.csv中包含手绘数字（从0到9）的灰度图像，每张图像为28×28像素，共784像素。每个像素有一个0到255的值表示该像素的亮度。
文件第1列为标签，之后784列分别为784个像素的亮度值。
我们的任务，就是要搭建一个神经网络，实现它的前向传播；也就是要根据输入的数据（28×28 = 784数据点表示的图像），推断出它到底是哪个数字，这个过程也被称为“推理”。
这里，我们构建的也是一个三层神经网络，输入层应该有784个神经元，输出层有10个神经元（表示0~9的分类结果）；中间设置2个隐藏层，第一个隐藏层有50个神经元，第二个隐藏层有100个神经元。这里的参数是需要 学习 得到的；我们假设已经学习完毕，直接从保存好的文件nn_sample中进行读取即可。
"""


def get_data():
    # 加载数据集
    data = pd.read_csv(r"D:\BaiduNetdiskDownload\DL\2.资料\data\train.csv")
    # 划分训练集和测试集
    X = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # 归一化
    preprocessor = MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_test, y_test


def init_network():
    # 加载模型
    network = joblib.load(r"D:\BaiduNetdiskDownload\DL\2.资料\data\nn_sample")
    return network


def predict(network, x):
    # 把模型的各层的参数取出来
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    """
    # 我们可以算一下w1，w2，w3 各自的维度
    w1 的维度 (784,50) b1 的维度是 (50,) 每个神经元都有自己的偏置
    w2 的维度 (50,100) b2 的维度是 (100,) 每个神经元都有自己的偏置
    w2 的维度 (100,10) b2 的维度是 (10,) 每个神经元都有自己的偏置
    就是当前神经元，对前一层的每个神经元都有自己对应的系数和偏置，和每个batch的样本数完全没有关系
    """

    # 784 → 50
    a1 = np.dot(x, w1) + b1  # (B, 784) @ (784, 50) → (B, 50)
    z1 = sigmoid(a1)  # (B, 50)

    # 50 → 100
    a2 = np.dot(z1, w2) + b2  # (B, 50) @ (50, 100) → (B, 100)
    z2 = sigmoid(a2)  # (B, 100)

    # 100 → 10
    a3 = np.dot(z2, w3) + b3  # (B, 100) @ (100, 10) → (B, 10)
    y = softmax(a3)  # (B, 10)
    return y


x, t = get_data()

network = init_network()

batch_size = 50
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i : i + batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
