import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


# ============================================================
# 1. 读取数据
# ============================================================

data_dir = Path(
    "/storage/data/尚硅谷ai/09_尚硅谷AI大模型之深度学习/2.资料/data"
)

fashion_mnist_train = pd.read_csv(
    data_dir / "fashion-mnist_train.csv"
)

fashion_mnist_test = pd.read_csv(
    data_dir / "fashion-mnist_test.csv"
)


# ============================================================
# 2. 转换为 Tensor
# ============================================================

# 原始像素范围 0~255，归一化到 0~1
X_train = torch.tensor(
    fashion_mnist_train.iloc[:, 1:].values,
    dtype=torch.float32
).reshape(-1, 1, 28, 28) / 255.0

y_train = torch.tensor(
    fashion_mnist_train.iloc[:, 0].values,
    dtype=torch.int64
)

X_test = torch.tensor(
    fashion_mnist_test.iloc[:, 1:].values,
    dtype=torch.float32
).reshape(-1, 1, 28, 28) / 255.0

y_test = torch.tensor(
    fashion_mnist_test.iloc[:, 0].values,
    dtype=torch.int64
)


# ============================================================
# 3. 构建数据集
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)


# ============================================================
# 4. 构建 LeNet 风格 CNN
# ============================================================

model = nn.Sequential(

    # (N, 1, 28, 28)
    nn.Conv2d(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        stride=1,
        padding=2
    ),

    # (N, 6, 28, 28)
    nn.ReLU(),

    # (N, 6, 14, 14)
    nn.AvgPool2d(
        kernel_size=2,
        stride=2
    ),

    # (N, 16, 10, 10)
    nn.Conv2d(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        stride=1,
        padding=0
    ),

    nn.ReLU(),

    # (N, 16, 5, 5)
    nn.AvgPool2d(
        kernel_size=2,
        stride=2
    ),

    # (N, 400)
    nn.Flatten(),

    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),

    nn.Linear(120, 84),
    nn.ReLU(),

    # 最终输出 logits：(N, 10)
    nn.Linear(84, 10)
)


# ============================================================
# 5. 查看每一层输出形状
# ============================================================

X = torch.rand(
    size=(100, 1, 28, 28),
    dtype=torch.float32
)

print("模型各层输出：")

for layer in model:
    X = layer(X)

    print(
        f"{layer.__class__.__name__:<12} "
        f"output shape: {X.shape}"
    )


# ============================================================
# 6. 训练函数
# ============================================================

def train(
    model,
    train_dataset,
    test_dataset,
    lr,
    epochs,
    batch_size,
    device
):

    # --------------------------------------------------------
    # 初始化模型参数
    # --------------------------------------------------------

    # PyTorch 本身已经有默认初始化方式。
    # 如果希望手动使用 Xavier，可以打开下面代码。

    # def init_weights(layer):
    #     if isinstance(layer, (nn.Linear, nn.Conv2d)):
    #         nn.init.xavier_uniform_(layer.weight)
    #
    #         if layer.bias is not None:
    #             nn.init.zeros_(layer.bias)
    #
    # model.apply(init_weights)


    # --------------------------------------------------------
    # 将模型放到 CPU / GPU
    # --------------------------------------------------------

    model.to(device)


    # --------------------------------------------------------
    # 损失函数
    # --------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # 优化器
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )


    # --------------------------------------------------------
    # DataLoader
    #
    # 不需要每个 epoch 都重新创建
    # --------------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ========================================================
    # 开始训练
    # ========================================================

    for epoch in range(epochs):

        # ====================================================
        # 一、训练阶段
        # ====================================================

        model.train()

        # 累计损失
        loss_accumulate = 0.0

        # 累计预测正确数量
        train_correct_accumulate = 0

        # 累计已经训练过的样本数量
        train_total = 0


        for batch_count, (X, y) in enumerate(train_loader):

            # -----------------------------------------------
            # 1. 数据移动到设备
            # -----------------------------------------------

            X = X.to(device)
            y = y.to(device)


            # -----------------------------------------------
            # 2. 前向传播
            # -----------------------------------------------

            output = model(X)


            # -----------------------------------------------
            # 3. 计算损失
            # -----------------------------------------------

            loss_value = loss_fn(output, y)


            # -----------------------------------------------
            # 4. 清空梯度
            # -----------------------------------------------

            optimizer.zero_grad()


            # -----------------------------------------------
            # 5. 反向传播
            # -----------------------------------------------

            loss_value.backward()


            # -----------------------------------------------
            # 6. 更新参数
            # -----------------------------------------------

            optimizer.step()


            # -----------------------------------------------
            # 7. 累加 loss
            # -----------------------------------------------

            loss_accumulate += loss_value.item()


            # -----------------------------------------------
            # 8. 获得预测类别
            #
            # output:
            # (batch_size, 10)
            #
            # 在 dim=1 上寻找最大值对应的位置
            # -----------------------------------------------

            pred = output.argmax(dim=1)


            # -----------------------------------------------
            # 9. 统计预测正确数量
            # -----------------------------------------------

            train_correct_accumulate += (
                pred.eq(y).sum().item()
            )


            # -----------------------------------------------
            # 10. 统计已经处理的样本数量
            # -----------------------------------------------

            train_total += y.size(0)


            # -----------------------------------------------
            # 11. 当前平均 loss
            #
            # 注意：
            # 这里不能除以 len(train_loader)
            #
            # 因为当前只训练了 batch_count + 1 个 batch
            # -----------------------------------------------

            current_loss = (
                loss_accumulate /
                (batch_count + 1)
            )


            # -----------------------------------------------
            # 12. 当前训练准确率
            #
            # 注意：
            # 这里不能直接除 len(train_dataset)
            #
            # 因为当前只处理了 train_total 个样本
            # -----------------------------------------------

            current_train_acc = (
                train_correct_accumulate /
                train_total
            )


            # -----------------------------------------------
            # 13. 显示训练进度
            # -----------------------------------------------

            progress = int(
                (batch_count + 1)
                / len(train_loader)
                * 50
            )

            print(
                f"\r"
                f"epoch:{epoch + 1:02d} "
                f"[{'=' * progress:<50}] "
                f"loss:{current_loss:.6f} "
                f"train_acc:{current_train_acc:.4f}",
                end=""
            )


        # ====================================================
        # 当前 epoch 最终训练结果
        # ====================================================

        train_loss = (
            loss_accumulate /
            len(train_loader)
        )

        train_acc = (
            train_correct_accumulate /
            len(train_dataset)
        )


        # ====================================================
        # 二、验证阶段
        # ====================================================

        model.eval()

        test_correct_accumulate = 0


        # 关闭自动求导
        with torch.no_grad():

            for X, y in test_loader:

                X = X.to(device)
                y = y.to(device)


                # 前向传播
                output = model(X)


                # 获得预测类别
                pred = output.argmax(dim=1)


                # 累加预测正确数量
                test_correct_accumulate += (
                    pred.eq(y).sum().item()
                )


        # ====================================================
        # 测试集准确率
        # ====================================================

        test_acc = (
            test_correct_accumulate /
            len(test_dataset)
        )


        # ====================================================
        # 输出当前 epoch 最终结果
        # ====================================================

        print(
            f"\r"
            f"epoch:{epoch + 1:02d} "
            f"[{'=' * 50}] "
            f"loss:{train_loss:.6f}, "
            f"train_acc:{train_acc:.4f}, "
            f"test_acc:{test_acc:.4f}"
        )


# ============================================================
# 7. 选择计算设备
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\n使用设备: {device}")


# ============================================================
# 8. 开始训练
# ============================================================

train(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    lr=0.001,
    epochs=50,
    batch_size=256,
    device=device
)