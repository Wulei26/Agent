# 反向传播算法：以手写数字识别为例

## 1. 反向传播到底是什么？

> **反向传播（Backpropagation）= 利用链式法则，从损失函数出发，沿着神经网络从后往前，高效计算所有参数的梯度。**

需要特别注意：

**反向传播不是一种优化算法，而是一种高效计算梯度的方法。**

真正负责更新参数的是梯度下降等优化方法。

---

# 2. 先看手写数字识别网络

以 MNIST 为例。

一张手写数字图片：

```text
28 × 28 = 784 个像素
```

把图片拉平成一个向量：

```text
x = [x1, x2, ..., x784]
```

建立一个两层神经网络：

```text
784 个输入
      ↓
   全连接层
      ↓
50 个隐藏神经元
      ↓
   全连接层
      ↓
10 个输出
```

数学形式：

\[
a_1 = xW_1+b_1
\]

\[
z_1 = ReLU(a_1)
\]

\[
a_2 = z_1W_2+b_2
\]

\[
y = Softmax(a_2)
\]

最后使用交叉熵：

\[
L = CrossEntropy(y,t)
\]

整个前向过程：

```text
x
│
▼
xW1+b1
│
▼
ReLU
│
▼
z1
│
▼
z1W2+b2
│
▼
Softmax
│
▼
预测概率 y
│
▼
Cross Entropy
│
▼
Loss
```

---

# 3. 神经网络为什么需要梯度？

神经网络有很多参数：

```text
W1
b1
W2
b2
```

训练的目标是让 Loss 越来越小。

也就是：

\[
L(W_1,b_1,W_2,b_2)
\]

我们需要知道：

> 每一个参数应该往哪个方向调整？

例如：

\[
\frac{\partial L}{\partial W_{1,23}}
\]

如果：

\[
\frac{\partial L}{\partial W_{1,23}}>0
\]

说明：

```text
W1[23] 增大
    ↓
Loss 增大
```

因此应该让：

```text
W1[23] 减小
```

梯度下降：

\[
W_{1,23}
\leftarrow
W_{1,23}
-\eta
\frac{\partial L}{\partial W_{1,23}}
\]

所以：

> **训练神经网络，本质上需要知道 Loss 对每一个参数的梯度。**

---

# 4. 之前的方法：数值微分

以前可以使用数值微分计算梯度。

例如：

```python
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

    return grad
```

它计算的是：

\[
\frac{dL}{dw}
\approx
\frac{L(w+h)-L(w-h)}{2h}
\]

也就是说：

```text
参数 w
 │
 ├── w + h → 算一次 Loss
 │
 └── w - h → 算一次 Loss

两次 Loss
    ↓
计算梯度
```

这个方法非常直观，而且适合学习梯度的概念。

但是它非常慢。

---

# 5. 为什么数值微分很慢？

假设神经网络有：

```text
40,000 个参数
```

数值微分需要对每一个参数进行：

```text
参数1 → forward 两次
参数2 → forward 两次
参数3 → forward 两次
...
参数40000 → forward 两次
```

大约需要：

\[
40000\times2=80000
\]

次前向计算。

也就是说：

> **每修改一个参数，都要重新把整个神经网络跑一遍。**

这非常浪费。

---

# 6. 反向传播的核心思想

反向传播换了一种思路：

> **前向传播已经计算过一次网络了，为什么不利用前向传播产生的中间结果？**

前向传播：

```text
x
 ↓
a1
 ↓
z1
 ↓
a2
 ↓
y
 ↓
L
```

前向传播已经计算出了：

```text
a1
z1
a2
y
L
```

那么反向传播：

```text
L
↑
y
↑
a2
↑
z1
↑
a1
↑
W1
```

也就是说：

> **Loss 从后往前传播梯度。**

这就是 Backpropagation。

---

# 7. 反向传播本质上是什么？

反向传播的数学核心只有一个：

# 链式法则

例如：

\[
z=xy
\]

\[
L=z^2
\]

假设：

\[
x=2,\quad y=3
\]

前向传播：

\[
z=2\times3=6
\]

\[
L=6^2=36
\]

现在想求：

\[
\frac{\partial L}{\partial x}
\]

计算图：

```text
x ──┐
    × ── z ── square ── L
y ──┘
```

从 Loss 开始往回：

### 第一步

\[
L=z^2
\]

所以：

\[
\frac{\partial L}{\partial z}=2z
\]

当前：

\[
z=6
\]

因此：

\[
\frac{\partial L}{\partial z}=12
\]

### 第二步

\[
z=xy
\]

所以：

\[
\frac{\partial z}{\partial x}=y=3
\]

### 第三步：链式法则

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial x}
\]

因此：

\[
=12\times3=36
\]

完成。

---

# 8. 数值微分和反向传播的区别

## 数值微分

思路：

> 每个参数稍微改变一下，然后观察 Loss 如何变化。

```text
参数1 → forward
参数2 → forward
参数3 → forward
...
参数N → forward
```

如果参数很多，就会非常慢。

---

## 反向传播

思路：

> 先完成一次 forward，然后从 Loss 开始利用链式法则向后计算梯度。

```text
             ┌→ grad_W1
             │
forward → backward
             │
             ├→ grad_b1
             │
             ├→ grad_W2
             │
             └→ grad_b2
```

只需要：

```text
一次 forward
+
一次 backward
```

就可以得到所有参数的梯度。

---

# 9. 回到手写数字识别

我们的网络：

\[
a_1=xW_1+b_1
\]

\[
z_1=ReLU(a_1)
\]

\[
a_2=z_1W_2+b_2
\]

\[
y=Softmax(a_2)
\]

\[
L=CrossEntropy(y,t)
\]

前向：

```text
x → W1 → a1 → ReLU → z1 → W2 → a2 → Softmax → y → Loss
```

现在开始反向传播：

```text
Loss
  ↑
  y
  ↑
 a2
  ↑
 z1
  ↑
 a1
  ↑
 W1
```

---

# 10. 第一站：Loss → Softmax

Softmax + Cross Entropy 有一个非常重要的结果：

\[
\boxed{
\frac{\partial L}{\partial a_2}=y-t
}
\]

假设预测：

```text
y =
[0.01, 0.03, 0.02, 0.05, 0.01,
 0.08, 0.04, 0.60, 0.10, 0.06]
```

真实标签是数字 `7`：

```text
t =
[0,0,0,0,0,0,0,1,0,0]
```

那么：

\[
y-t
\]

得到：

```text
[ 0.01,
  0.03,
  0.02,
  0.05,
  0.01,
  0.08,
  0.04,
 -0.40,
  0.10,
  0.06]
```

它表达了一个非常直观的信息：

```text
预测概率过高的错误类别
        ↓
产生正梯度
        ↓
应该压低

真实类别
        ↓
产生负梯度
        ↓
应该提高
```

所以：

\[
da_2 = y-t
\]

---

# 11. 第二站：a2 → W2

我们有：

\[
a_2=z_1W_2+b_2
\]

已经知道：

\[
da_2=\frac{\partial L}{\partial a_2}
\]

现在要求：

\[
\frac{\partial L}{\partial W_2}
\]

根据链式法则：

\[
\boxed{
grad\_W_2=z_1^Tda_2
}
\]

代码：

```python
grad_W2 = z1.T @ da2
```

假设：

```text
z1.shape = (100, 50)
da2.shape = (100, 10)
```

那么：

```text
z1.T
↓
(50, 100)

da2
↓
(100, 10)
```

相乘：

```text
(50,100) @ (100,10)
          ↓
       (50,10)
```

刚好得到：

```text
grad_W2.shape = (50,10)
```

与：

```text
W2.shape = (50,10)
```

完全一致。

---

# 12. 第三站：a2 → z1

我们还需要继续把梯度向前传播。

因为：

\[
a_2=z_1W_2+b_2
\]

所以：

\[
\boxed{
dz_1=da_2W_2^T
}
\]

代码：

```python
dz1 = da2 @ W2.T
```

于是：

```text
Loss
 ↓
da2
 ↓
dz1
```

梯度继续向前传播。

---

# 13. 第四站：穿过 ReLU

我们有：

\[
z_1=ReLU(a_1)
\]

ReLU：

\[
ReLU(x)=
\begin{cases}
x & x>0\\
0 & x\leq0
\end{cases}
\]

它的导数：

\[
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq0
\end{cases}
\]

因此：

\[
\boxed{
da_1=dz_1\times ReLU'(a_1)
}
\]

代码：

```python
da1 = dz1 * (a1 > 0)
```

例如：

```text
a1：

[-2, 3, -1, 5, 0.2]
```

ReLU 导数：

```text
[0, 1, 0, 1, 1]
```

如果：

```text
dz1：

[0.4, 0.7, 0.3, -0.2, 0.5]
```

那么：

```text
da1：

[0, 0.7, 0, -0.2, 0.5]
```

这意味着：

> 前向传播中没有激活的神经元，反向传播时不会接收梯度。

---

# 14. 第五站：a1 → W1

我们有：

\[
a_1=xW_1+b_1
\]

根据链式法则：

\[
\boxed{
grad\_W_1=x^Tda_1
}
\]

代码：

```python
grad_W1 = x.T @ da1
```

如果：

```text
x.shape = (100, 784)
da1.shape = (100, 50)
```

那么：

```text
x.T
↓
(784,100)

da1
↓
(100,50)
```

相乘：

```text
(784,100) @ (100,50)
          ↓
       (784,50)
```

正好得到：

```text
grad_W1.shape = (784,50)
```

与：

```text
W1.shape = (784,50)
```

一致。

---

# 15. Bias 的梯度

对于：

\[
a_1=xW_1+b_1
\]

因为：

\[
\frac{\partial a_1}{\partial b_1}=1
\]

所以：

\[
\boxed{
grad\_b_1=\sum_{batch}da_1
}
\]

代码：

```python
grad_b1 = np.sum(da1, axis=0)
```

同理：

```python
grad_b2 = np.sum(da2, axis=0)
```

---

# 16. 整个反向传播过程

完整过程可以记成：

```text
                        Forward
                           ↓

x → W1 → a1 → ReLU → z1 → W2 → a2 → Softmax → y → Loss
                                                                  │
                                                                  │
                        Backward                                 ▼
                                                                  │
                                                                  ▼
                         da2 = y - t
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        grad_W2 = z1.T @ da2       grad_b2 = sum(da2)
                 │
                 ▼
        dz1 = da2 @ W2.T
                 │
                 ▼
        da1 = dz1 * (a1 > 0)
                 │
          ┌──────┴───────────┐
          ▼                  ▼
grad_W1 = x.T @ da1    grad_b1 = sum(da1)
```

最终得到：

```python
grads = {
    "W1": grad_W1,
    "b1": grad_b1,
    "W2": grad_W2,
    "b2": grad_b2
}
```

---

# 17. 得到梯度之后怎么办？

反向传播只负责：

> **计算梯度。**

真正更新参数的是梯度下降。

例如：

```python
W1 -= learning_rate * grad_W1
b1 -= learning_rate * grad_b1

W2 -= learning_rate * grad_W2
b2 -= learning_rate * grad_b2
```

所以整个训练过程是：

```text
① Forward
      ↓
② 计算 Loss
      ↓
③ Backward
      ↓
④ 得到梯度
      ↓
⑤ 梯度下降更新参数
      ↓
⑥ 下一批数据
      ↓
回到 ①
```

---

# 18. 为什么反向传播比数值微分快？

假设网络有：

```text
P 个参数
```

## 数值微分

每个参数需要大约两次前向计算：

\[
O(P\times Forward)
\]

如果：

\[
P=1,000,000
\]

就需要大约：

```text
2,000,000 次参数扰动对应的计算
```

---

## 反向传播

只需要：

```text
一次 Forward
+
一次 Backward
```

反向传播的计算量通常与一次前向传播是同一个数量级。

因此：

> **反向传播不是少算了一些参数，而是利用链式法则共享了大量中间计算结果。**

这是它高效的根本原因。

---

# 19. 为什么反向传播可以“一次性”算出所有梯度？

假设：

```text
x
 ↓
A
 ↓
B
 ↓
C
 ↓
Loss
```

数值微分会：

```text
改变参数1 → 整个网络重新算
改变参数2 → 整个网络重新算
改变参数3 → 整个网络重新算
...
```

而反向传播：

```text
                Loss
                 │
                 ▼
                dL/dC
                 │
                 ▼
                dL/dB
                 │
                 ▼
                dL/dA
                 │
                 ▼
                dL/dx
```

每经过一个节点，就利用一次链式法则：

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
\]

于是一次反向过程，就可以把梯度传遍整个网络。

---

# 20. 一个非常直观的类比

假设有：

```text
10000 个旋钮
```

你想知道：

> 每个旋钮对最终温度有多大影响？

## 数值微分

一个一个试：

```text
旋钮1 ↑一点 → 测温度
旋钮1 ↓一点 → 测温度

旋钮2 ↑一点 → 测温度
旋钮2 ↓一点 → 测温度

...

旋钮10000 ↑一点
旋钮10000 ↓一点
```

非常慢。

## 反向传播

先让整个系统运行一次：

```text
旋钮
 ↓
机器
 ↓
最终温度
```

然后从最终温度开始：

```text
温度
 ↓
最后一个部件
 ↓
上一个部件
 ↓
上一个部件
 ↓
...
 ↓
所有旋钮
```

利用链式法则，一次性算出：

```text
旋钮1的影响
旋钮2的影响
旋钮3的影响
...
旋钮10000的影响
```

这就是反向传播。

---

# 21. “反向传播”传播的到底是什么？

不是：

```text
Loss
 ↓
告诉上一层“你错了”
```

严格来说，传播的是：

\[
\boxed{
\frac{\partial L}{\partial 当前变量}
}
\]

例如：

```text
Loss
 ↓
∂L/∂a2
 ↓
∂L/∂z1
 ↓
∂L/∂a1
```

最终得到：

```text
∂L/∂W2
∂L/∂b2
∂L/∂W1
∂L/∂b1
```

所以：

> **反向传播的本质，就是链式法则的工程化实现。**

---

# 22. 为什么叫“传播”？

前向传播：

```text
输入
 ↓
 ↓
 ↓
Loss
```

数据从前往后传播：

```text
Forward Propagation
```

反向传播：

```text
Loss
 ↑
 ↑
 ↑
参数
```

梯度从后往前传播：

```text
Backward Propagation
```

所以可以记住：

> **前向传播的是数据和计算结果；反向传播的是梯度。**

---

# 23. 神经网络训练的完整闭环

手写数字识别训练可以理解成：

```text
                 ┌─────────────────────┐
                 │                     │
                 ▼                     │
图片 X → 神经网络 → 预测 Y → Loss      │
             ▲               │         │
             │               │         │
             │               ▼         │
             │          反向传播        │
             │               │         │
             │               ▼         │
             └──────── 梯度下降更新 ────┘
```

每一个 batch：

### ① Forward

```python
y = network.predict(x_batch)
```

得到预测结果。

### ② Loss

```python
loss = network.loss(x_batch, t_batch)
```

计算预测有多差。

### ③ Backward

```python
grads = network.gradient(x_batch, t_batch)
```

计算：

```text
W1 应该往哪里调整？
b1 应该往哪里调整？
W2 应该往哪里调整？
b2 应该往哪里调整？
```

### ④ Update

```python
W1 -= learning_rate * grads["W1"]
b1 -= learning_rate * grads["b1"]

W2 -= learning_rate * grads["W2"]
b2 -= learning_rate * grads["b2"]
```

然后处理下一批数据。

---

# 24. 最终建立你的知识地图

你前面学习的内容其实是一条完整的知识链：

```text
线性模型
   ↓
梯度
   ↓
梯度下降
   ↓
逻辑回归
   ↓
Softmax
   ↓
交叉熵
   ↓
神经网络
   ↓
数值微分
   ↓
链式法则
   ↓
反向传播
   ↓
高效训练
```

其中最重要的思想变化是：

```text
以前：

一个参数
   ↓
稍微改变
   ↓
重新计算 Loss
   ↓
估计梯度

现在：

一次 Forward
   ↓
保存中间结果
   ↓
从 Loss 开始
   ↓
利用链式法则反向传播
   ↓
一次性得到所有参数梯度
```

---

# 25. 一句话记住反向传播

## 数值微分

> **“我把每个参数稍微改一下，看看 Loss 怎么变化。”**

## 反向传播

> **“我已经知道最终 Loss 了，我利用链式法则，从 Loss 开始往回推，一次性计算所有参数对 Loss 的影响。”**

因此：

\[
\boxed{
\text{反向传播不是让神经网络能够学习，而是让神经网络能够高效地计算梯度。}
}
\]

而：

\[
\boxed{
\text{梯度下降负责利用这些梯度更新参数。}
}
\]

---

# 26. 下一步最值得彻底弄懂的三个公式

如果要真正掌握 `TwoLayerNet.gradient()`，下一步重点理解：

### 公式 1

\[
\boxed{
da_2=y-t
}
\]

它来自：

```text
Softmax + Cross Entropy
```

---

### 公式 2

\[
\boxed{
grad\_W_2=z_1^Tda_2
}
\]

它来自：

```text
a2 = z1 @ W2 + b2
```

以及链式法则。

---

### 公式 3

\[
\boxed{
dz_1=da_2W_2^T
}
\]

它表示：

```text
Loss 的梯度
    ↓
穿过 W2
    ↓
继续传回隐藏层
```

一旦这三个公式彻底理解，`TwoLayerNet.gradient()` 中的大部分代码就不再神秘。

---

# 总结

反向传播可以浓缩成四句话：

1. **前向传播：** 输入数据，计算预测结果和 Loss。
2. **损失函数：** 衡量预测结果与真实答案之间的差距。
3. **反向传播：** 从 Loss 开始，利用链式法则从后往前计算每个参数的梯度。
4. **梯度下降：** 根据梯度更新参数，让下一次预测更准确。

最终：

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Gradient Descent
   ↓
Update Parameters
   ↓
重复
```

这就是神经网络训练最核心的循环。
