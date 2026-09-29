import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


# 训练一个epoch
def train_step(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    epoch_train_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        # 前向传播
        output = model(input)
        loss_value = loss_fn(output, target)
        # 梯度清零
        model.zero_grad()
        # 反向传播计算梯度
        loss_value.backward()
        # 梯度更新
        optimizer.step()
        # 累加当前损失
        epoch_train_loss += loss_value.item()
    this_epoch_train_loss = epoch_train_loss / len(train_loader)
    return this_epoch_train_loss


# 返回一个epoch验证集的平均损失
def val_step(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            # 前向传播
            output = model(input)
            loss_value = loss_fn(output, target)
            # 累加当前损失
            epoch_test_loss += loss_value.item()
        this_epoch_test_loss = epoch_test_loss / len(test_loader)
    return this_epoch_test_loss
