## 逻辑回归的损失函数

![示意图](../assets/logisticregression.png)

## 逻辑回归损失函数的梯度

![示意图](../assets/logisticregression_loss.png)

## 推广到多分类

![示意图](../assets/softmax_logisticregression.png)
这里其实是计算了每个类的概率，分母是所有类的评分和，分子是C类的评分，那么最终输出的就是y为C类的概率。其实损失函数就是前面的二元交叉熵损失函数的变体，示性函数仅当Y=c时取1,其余均取0。其实内部累加符号就是y=c时的概率。
