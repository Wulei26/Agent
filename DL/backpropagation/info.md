# 一 、反向传播
以数字识别为例
![alt text](../assets/../assets/image.png)
![alt text](../assets/../assets/image-1.png)
## 二、激活层的反向传播
### 2.1 Relu的反向传播
![alt text](../assets/image-2.png)
```python
import numpy as np


class ReLU:

    def __init__(self) -> None:
        # 用属性记录哪些输入小于0（布尔掩码）
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 记录输入 <= 0 的位置
        self.mask = (x <= 0)
        out = x.copy()
        # 大于0保留本身，小于等于0置为0
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout 是从上一层传递过来的梯度
        dout[self.mask] = 0
        dx = dout.copy()
        return dx
```
![alt text](../assets/image-3.png)
### 2.2 Sigmoid 的反向传播
![alt text](../assets/image-4.png)
```python
class Sigmoid:

    def __init__(self) -> None:
        # 前向输出，初始为 None
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out) * self.out
        return dx
```
![alt text](../assets/image-5.png)
x不是参数x，而是输入到Sigmoid函数的向量/矩阵
### 2.3 Affine的反向传播和实现
在全连接层（Fully Connected Layer，Dense Layer）中，每个输入节点与输出节点相连，通过权重矩阵和偏置进行线性变换，这种操作在几何领域称为仿射变换（Affine transformation，几何中，仿射变换包括一次线性变换和一次平移，分别对应神经网络的加权求和运算与加偏置运算）。
考虑N个数据一起进行正向传播的情况，写成矩阵计算形式：
\[
Y = XW+B
\]
这里的X是形状为N×m的矩阵，m就是Affine层输入神经元的个数；而W是形状为m×n的权重矩阵，n就是Affine层输出神经元的个数。
根据矩阵求导的运算法则，可以得到损失函数L关于X、W的偏导数：
**对 $X$ 的梯度：**

$$
\frac{\partial L}{\partial X}
= \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X}
= \frac{\partial L}{\partial Y} \cdot W^T
$$

**对 $W$ 的梯度：**

$$
\frac{\partial L}{\partial W}
= X^T \cdot \frac{\partial L}{\partial Y}
$$
```python
class Affine:

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        """仿射层（Affine / 全连接层）：Y = XW + b

        Parameters
        ----------
        W : np.ndarray
            权重矩阵，形状 (D, M)。
        b : np.ndarray
            偏置向量，形状 (M,)。
        """
        self.W: np.ndarray = W
        self.b: np.ndarray = b

        # 保存输入的 x（已展平为二维）
        self.X: np.ndarray | None = None

        # 反向传播时计算的梯度
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

        # 记录输入原始形状，考虑输入可能是三维甚至多维的情况
        self.original_x_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : np.ndarray
            前一层的输出，形状任意（第一维为 batch 维）。

        Returns
        -------
        np.ndarray
            线性变换结果，形状 (N, M)。
        """
        self.original_x_shape = x.shape
        # 将输入展平为二维 (N, D) 再计算
        self.X = x.reshape(x.shape[0], -1)
        Y = np.dot(self.X, self.W) + self.b
        return Y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """反向传播。

        Parameters
        ----------
        dy : np.ndarray
            上游传来的梯度，形状 (N, M)。

        Returns
        -------
        np.ndarray
            对输入 x 的梯度，形状与 forward 的输入 x 相同。
        """
        # 对输入 x 的梯度：∂L/∂X = ∂L/∂Y · Wᵀ
        dx = np.dot(dy, self.W.T)

        # 对权重 W 的梯度：∂L/∂W = Xᵀ · ∂L/∂Y
        self.dW = self.X.T @ dy

        # 对偏置 b 的梯度：∂L/∂b = Σ_i ∂L/∂Y_i
        #
        # 前向时 b 被广播到了 N 个样本上，每个样本都对 b 有贡献；
        # 反向时按链式法则，b 的梯度就是这 N 份贡献的累加，
        # 所以要对 dy 沿样本维（axis=0）求和，
        # 得到与 b 同形状的 (M,) 向量。
        self.db = np.sum(dy, axis=0)

        # 解包恢复原来的维度
        return dx.reshape(*self.original_x_shape)
```
### 2.4 Softmax 的反向传播极其实现
![alt text](../assets/image-6.png)
![alt text](../assets/image-7.png)
![alt text](../assets/image-8.png)
这是经过softmax函数和交叉熵损失函数的示意图
```
Z2
 ↓
Softmax
 ↓
y
 ↓
Cross Entropy
 ↓
Loss (L)
```
我们这里实际上要求的是
$$
\frac{\partial L}{\partial Z2}
$$
​
神奇的地方就在于：
$$
\frac{\partial L}{\partial Z_2} = \frac{y - t}{B}
$$
其中 t 必须是 One-Hot：
>  y = [0.1, 0.7, 0.2]
t = [0,   1,   0]
这里的B就是batch_size

**推导过程：**
这是 Softmax + Cross Entropy 最经典的结论：

$$
L = -\sum_j t_j \log y_j
$$

Softmax：

$$
y_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

把 Softmax 代入交叉熵：

$$
L = -\sum_j t_j \log \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

展开：

$$
L = -\sum_j t_j z_j + \log \sum_k e^{z_k}
$$

对 $z_j$ 求导：

$$
\frac{\partial L}{\partial z_j} = -t_j + \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

而后面这一项正好就是：

$$
y_j
$$

所以：

$$
\boxed{\frac{\partial L}{\partial z_j} = y_j - t_j}
$$

如果你的 Loss 是 batch 平均：

$$
\boxed{D_2 = \frac{y - t}{B}}
$$

这就是你之前看到的 **为什么 $D_2 = Y - T$ 的来源**。

如果t不是独热编码怎么办？那也很简单，那么此时就把t当成下标不就行了，比如t=[1,0,2],那么，t作为下标在y中的取值不就是正确标签对应的概率嘛
``` 
                 t 是类别下标
                       |
                       ↓
                   [1, 0, 2]
                       |
                       ↓
                 相当于 One-Hot
                       |
                       ↓
Y — Softmax —> Y — Cross Entropy —> L
↑
|___ 反向传播
     |
     ↓
dZ = (Y - T) / B
```
代码实现：
```python
class SoftmaxWithLoss:
    @staticmethod
    def softmax(x: np.ndarray):  # 这里的x是二维数组
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return y

    @staticmethod
    def cross_entropy_error(y: np.ndarray, t: np.ndarray):

        if y.ndim == 1:  # 如果是输入一个样本，那么还是将其转化为二维矩阵
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        if t.size == y.size:  # 也就是说经过了独热编码
            # 转换
            t = t.argmax(axis=1)  # 返回的是最大值所在的索引（下标）
        batch_size = t.shape[0]
        loss_sum = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))
        return loss_sum / batch_size

    def __init__(self):
        self.loss = None
        self.y = None  # 这是softmax的输出
        self.t = None  ##这是正确的标签，这里采用独热编码

    def forward(
        self,
        x: np.ndarray,
        t: np.ndarray,
    ):
        self.t = t
        self.y = __class__.softmax(x)
        self.loss = __class__.cross_entropy_error(self.y, t)
        return self.loss

    def backward(
        self,
        dout=1.0,  # 这个dout 是上一层传递过来的梯度，但是我们这里是把SoftmaxWithLoss作为一个整体，所以上一层传递过来的梯度就是DL/DL = 1
    ):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            # t 是One-hot
            dx = (self.y - self.t) / batch_size
        else:
            # t如果是类别下标,这里其实就是巧妙的将类别下标转化为了One-hot,直接用类别作为下标去取y中的值，对应的就是正确标签;
            # 如果是独热编码，那么t这个位置的值就是1,  所以这里才会减去1。非常巧妙
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
```
# 三、神经网络的训练优化
### 3.1 梯度消失和梯度爆炸
 在某些神经网络中，随着网络深度的增加，梯度在隐藏层反向传播时倾向于变小。这就意味着，前面隐藏层中的神经元要比后面的学习起来更慢。这种现象被称为“梯度消失”;反之梯度在传递的过程中越变越大，导致参数振荡，训练不稳定。这也是为什么深层神经网络后来会使用 ReLU、He 初始化、BatchNorm、残差连接等方法来缓解这些问题。
#### 3.1.1 SGD（随机梯度下降）

**公式**

$$
w_{t+1} = w_t - \eta \, g_t
$$

其中：

- $w_t$：第 $t$ 步的参数
- $\eta$：学习率（lr）
- $g_t$：第 $t$ 步的梯度 $\nabla L(w_t)$

**实现：**

```python
class SGD:

    def __init__(self, lr: float):
        self.lr = lr

    def update(
        self,
        params: dict,  # 保存参数的字典
        grad: dict,  # 保存梯度的字典
    ):
        # 参数更新策略 w = w - lr * dw
        for key in params.keys():
            params[key] = params[key] - self.lr * grad[key]
```
#### 3.1.2  动量法 Momentum
保存历史梯度，Momentum 就是在 SGD 基础上给参数更新加一个"速度"变量，用 `v = β*v + grad `累积历史梯度，再  `w = w - lr*v `更新，从而加速同向更新、抑制反向震荡，收敛更快更稳。
$$
v_{t+1} = \beta \, v_t + g_t
$$

$$
w_{t+1} = w_t - \eta \, v_{t+1}
$$

其中：

- $v_t$：第 $t$ 步的速度（累积梯度）
- $\beta$：动量系数（通常取 $0.9$）
- $\eta$：学习率
- $g_t$：第 $t$ 步的梯度 $\nabla L(w_t)$
**实现：**
```python
class Momentum:
    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0.9,
    ):
        self.lr = lr
        self.beta = beta
        self.v: dict = None  # 记录历史动量

    def update(
        self,
        parms: dict,  # 当前参数
        grads: dict,  # 当前梯度
    ):
        # v = β*v + grad
        # W← W + v
        # 初始状态（历史梯度为0）
        if self.v is None:
            self.v = {}
            for key, val in parms.items():
                self.v[key] = np.zeros_like(val)
        # 更新梯度
        for key in parms.keys():
            self.v[key] = self.beta * self.v[key] - self.lr * grads[key]
            parms[key] += self.v[key]

```
#### 3.1.3 学习率衰减
深度学习模型训练中调整最频繁的当属学习率，好的学习率可以使模型逐渐收敛并获得更好的精度。较大的学习率可以加快收敛速度，但可能在最优解附近震荡或不收敛；较小的学习率可以提高收敛的精度，但训练速度慢。**学习率衰减是一种平衡策略，初期使用较大学习率快速接近最优解，后期逐渐减小学习率，使参数更稳定地收敛到最优解。**
- 等间隔衰减：比如学习率每隔20个epoch就衰减为之前的0.7
- 指定间隔衰减：在指定的epoch，让学习率按照一定系数进行衰减，比如在epoch达到[10，50，200] 达到这些epoch的时候，学习率就衰减为原来的0.7
- 指数衰减：让学习率按照指数进行衰减，比如每一个epoch学习率衰减为原来的原来的0.99，这样学习率就会原来越小

#### 3.1.4 AdaGrad（Adaptive Gradient，自适应梯度）
**为每个参数适当地调整学习率，并且伴随着学习的进行，学习率会逐渐减小**
$$
h \leftarrow h + \nabla^2
$$

$$
W \leftarrow W - \eta \frac{1}{\sqrt{h}} \nabla
$$
> h：历史梯度的平方和
> 这里 ∇2 就表示了梯度的平方和，即 ∂L∂W ⊙ ∂L∂W ，这里的 ⊙ 表示对应矩阵元素的乘法。
>使用AdaGrad时，学习越深入，更新的幅度就越小。如果无止境地学习，更新量就会变为0，完全不再更新。 

代码实现：
```python
class AdaGrad:
    def __init__(self,beta:0.01):
        self.beta = beta #初始学习率
        self.h:dict = None #记录的是历史梯度的平方和
    def update(self,
        params:dict, #当前的参数
        grad:dict, #当前的参数梯度
    ):
        if self.h is None:
            # 初始化
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            # 加上当前梯度的平方和
            self.h[key] = grad[key] * grad[key] + self.h[key]
            # 更新参数
            params[key] -= self.beta*grad[key] / (np.sqrt(self.h[key]) + 1e-7)
            ##随着学习的进行 h会越来越大，那么更新的幅度就会越来越小，也就是变相的把学习率减小了
```
#### 3.1.5 RMSProp（Root Mean Square Propagation，均方根传播）
在AdaGrad基础上的改进，它并非将过去所有梯度一视同仁的相加，而是逐渐遗忘过去的梯度，采用指数移动加权平均，呈指数地减小过去梯度的尺度。有点像结合结合了Mountent和Adagrad的一种优化方法
$$
h \leftarrow \alpha h + (1 - \alpha) \nabla^2
$$

$$
W \leftarrow W - \eta \frac{1}{\sqrt{h}} \nabla
$$

- \(h\)：历史梯度平方和的指数移动加权平均
- \(\alpha\)：权重

实现：
```python
class RMSProp:
    
    def __init__(self,
        alpha:float, ##衰减系数
        lr:float,
    ):
        self.alpha = alpha
        self.lr = lr
        self.h:dict = None #记录的是历史梯度的平方和
    def update(self,
        params:dict, #当前的参数
        grad:dict, #当前的参数梯度
    ):
        if self.h is None:
            # 初始化
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            # 更新历史梯度
            self.h[key] *= self.alpha 
            self.h[key] += (1-self.alpha) * grad[key] **2
            # 更新参数
            params[key] -= self.lr*grad[key] / (np.sqrt(self.h[key]) + 1e-7)
```
#### 3.1.6 Adam（Adaptive Moment Estimation，自适应矩估计）
**Adam 算法本质上是Momentum（动量法）和 RMSProp（均方根传播**）这两种优化算法思想的集大成者，同时加入了独特的偏差校正（Bias Correction）机制。
$$
v \leftarrow \alpha_1 v + (1 - \alpha_1) \nabla
$$

$$
h \leftarrow \alpha_2 h + (1 - \alpha_2) \nabla^2
$$

$$
\hat{v} = \frac{v}{1 - \alpha_1^t}
$$

$$
\hat{h} = \frac{h}{1 - \alpha_2^t}
$$

$$
W \leftarrow W - \eta \frac{\hat{v}}{\sqrt{\hat{h}}}
$$

- η：学习率
- α1、α2：一次动量系数和二次动量系数(都属于[0,1],但更接近于1)
- t：迭代次数，从1开始
实现：
```python
class Adam:

    def __init__(self,
        alpha1:float = 0.9,
        alpha2:float = 0.999,
        lr:float = 0.01,
     ):
        self.alpha1 = alpha1 #一次动量系数
        self.alpha2 = alpha2 # 二次动量系数
        self.lr = lr #学习率
        self.v:dict = None # 当前的v
        self.h:dict = None # 当前h
        self.i:int = 0 #当前的迭代次数
    
    def update(self,
        params:dict, #当前的参数
        grads:dict, #当前的参数梯度
    ):
        if self.v is None or self.h is None:
            # 初始化
            self.v = {}
            self.h = {}
            for key, value in grads.items():
                self.v[key] = np.zeros_like(value)
                self.h[key] = np.zeros_like(value)
        self.i += 1 #注意这是总的迭代次数，不是每个epoch的迭代次数
        for key, value in params.items():
            # 更新v
            self.v[key] *= self.alpha1
            self.v[key] += (1 - self.alpha1) * grads[key]
            # 更新h
            self.h[key] *= self.alpha2
            self.h[key] += (1 - self.alpha2) * grads[key] ** 2
            # 更新参数
            v_hat = self.v[key] / (1 - self.alpha1 ** self.i)
            h_hat = self.h[key] / (1 - self.alpha2 ** self.i)
            params[key] -= self.lr * v_hat / (np.sqrt(h_hat) + 1e-7)
```
## 3.2 参数初始化
参数初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要。此外，这些初始化方案的选择可以与激活函数的选择有趣的结合在一起。我们选择哪个激活函数以及如何初始化参数，可以决定优化算法收敛的速度有多快；糟糕选择可能会导致我们在训练时遇到梯度爆炸或梯度消失。
#### 3.2.1 常数初始化
注意：将权重初始值设为0将无法正确进行学习。严格地说，不能将权重初始值设成一样的值。因为这意味着反向传播时权重全部都会进行相同的更新，被更新为相同的值（对称的值）。这使得神经网络拥有许多不同的权重的意义丧失了。为了防止“权重均一化”（瓦解权重的对称结构），必须随机生成初始值。
#### 3.2.2 秩初始化
权重参数初始化为单位矩阵，即
$$
W=I
$$
这里I为单位矩阵，即主对角线上元素为1，其它元素为0。
#### 3.2.3 正态分布初始化
权重参数按指定均值μ与标准差σ正态分布初始化。因为不能直接将权重初始化为相同的常数，所以需要对参数进行随机初始化。最常见的随机分布就是 正态分布（也叫 高斯分布），记作 X ~ N(μ, σ2)。
#### 3.2.4 均匀分布初始化
权重参数在指定区间内均匀分布初始化。均匀分布一般记作 X ~ U(a, b)。
#### 3.2.5 Xavier 初始化（Glorot 初始化）
Xavier 初始化根据输入和输出的神经元数量调整权重的初始范围，确保每一层的输出方差与输入方差相近。

Xavier 正态分布初始化：均值为 0，标准差为 \(\sqrt{\frac{2}{n_{in} + n_{out}}}\) 的正态分布。

Xavier 均匀分布初始化：区间 \(\left[ -\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}} \right]\) 内均匀分布。

其中 \(n_{in}\) 表示输入数，\(n_{out}\) 表示输出数。

**Xavier初始化参数适用于Sigmoid和Tanh等激活函数，能有效缓解梯度消失或爆炸问题。**
#### 3.2.6 He初始化（Kaiming初始化）
He 初始化根据输入的神经元数量调整权重的初始范围。

He 正态分布初始化：均值为 0，标准差为 \(\sqrt{\frac{2}{n_{in}}}\) 的正态分布。

He 均匀分布初始化：区间 \(\left[ -\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}} \right]\) 内均匀分布。

其中 \(n_{in}\) 表示输入数。

He 初始化参数主要适用于 ReLU 及其变体（如 Leaky ReLU）激活函数。
## 3.3 正则化
机器学习的问题中，过拟合 是一个很常见的问题。
过拟合指的是能较好拟合训练数据，但不能很好地拟合不包含在训练数据中的其他数据。机器学习的目标是提高泛化能力，希望即便是不包含在训练数据里的未观测数据，模型也可以进行正确的预测。因此可以通过 正则化 方法来抑制过拟合。
常用的正则化方法有Batch Normalization、权值衰减、Dropout、早停法等。
- **Batch Normalization批量标准化**
- **权值衰减**
- **Dropout 随机失活**：训练时以概率p随机关闭神经元，迫使网络不依赖特定神经元，增强鲁棒性，同时未被关闭的神经元的输出值以1/（1−p）的比例进行缩放，以保持期望值不变；而测试时通常不使用Dropout，即所有神经元保持激活状态并且不进行缩放。**通常放在激活函数之后，线性层（全连接层/卷积层）之前**