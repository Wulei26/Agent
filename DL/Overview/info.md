
### 二、常见激活函数
激活函数是连接感知机和神经网络的桥梁，在神经网络中起着至关重要的作用。
如果没有激活函数，整个神经网络就等效于单层线性变换，不论如何加深层数，总是存在与之等效的“无隐藏层的神经网络”。激活函数必须是非线性函数，也正是激活函数的存在为神经网络引入了非线性，使得神经网络能够学习和表示复杂的非线性关系

#### 2.1 Sigmoid函数
![alt text](../assets/sigmoid.png)
```python
def sigmoid(x:np.ndarray):
    return 1 / (1 + np.exp(x))
```
#### 2.2 Tanh函数
![alt text](../assets/Tanu.png)
```python
def Tanh(x: np.ndarray):
    exp_x = np.exp(-2 * x)
    return (1 - exp_x) / (1 + exp_x)
```
#### 2.3 ReLU函数
![alt text](../assets/ReLU.png)
```python
def ReLU(x: np.ndarray):
    return np.maximum(0, x)
```
#### 2.4 Softmax 函数
![alt text](../assets/Softmax.png)
理解：softmax的输入函数是一个二维矩阵，行是样本，列是标签，元素是各个样本在不同标签的评分，现在softmax的目标就是将这个得分转化为样本取不同标签的概率
```python
def softmax(x: np.ndarray):  # 这里的x是二维数组
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return y

```
> keepdims=True 的作用，就是求完最大值/求和以后，保留那个被压缩的维度，方便后面的广播。
>
理解：这里为什么要 `x = x - np.max(x, axis=1)` ？
是防止有些值过大，导致指数计算溢出，而且这样是不影响结果的，因为你看softmax的公式，分子分母都是指数，而且没有常数项，而我们这里计算的是个分数，所以分子分母同时除以一个数，这个分数大小是不会变化的