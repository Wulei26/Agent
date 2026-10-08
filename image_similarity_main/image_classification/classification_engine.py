"""
训练、验证和测试的实现
"""
__all__ = ["train_step", "val_step"]

import torch
def train_step(model,train_loader,loss_fn,optimizer,device):
    """
    执行一个训练步骤

    参数:
    - model : PyTorch模型
    - train_loader : 训练数据加载器
    - loss_fn : 损失函数
    - optimizer : 优化器
    - device : 计算设备（如'cuda'或'cpu'）

    返回:
    - 平均训练损失
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0.0  # 初始化总损失
    num_batches = 0 # 初始化批次数量
    for train_img, classification in train_loader:
        # 将数据移动到device
        train_img, classification = train_img.to(device), classification.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(train_img)  # 前向传播
        loss = loss_fn(output, classification)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()  # 累加损失
        num_batches += 1  # 累加批次数量
    avg_loss = total_loss / num_batches  # 计算平均损失
    return avg_loss  # 返回平均训练损失

def val_step(model,val_loader,loss_fn,device):
    """
    执行验证步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 验证集的平均损失（标量值）
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0  # 初始化总损失
    num_batches = 0 # 初始化批次数量
    with torch.no_grad():  # 禁用梯度计算
        for val_img, classification in val_loader:
            # 将数据移动到device
            val_img, classification = val_img.to(device), classification.to(device)
            output = model(val_img)  # 前向传播
            loss = loss_fn(output, classification)  # 计算损失
            total_loss += loss.item()  # 累加损失
            num_batches += 1  # 累加批次数量
    avg_loss = total_loss / num_batches  # 计算平均损失
    return avg_loss  # 返回平均验证损失

def test_step(model,test_loader,device):
    """
    执行测试步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 测试集的准确率（标量值）
    """
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确预测计数
    total = 0  # 初始化总样本计数
    with torch.no_grad():  # 禁用梯度计算
        for test_img, classification in test_loader:
            # 将数据移动到device
            test_img, classification = test_img.to(device), classification.to(device)
            output = model(test_img)  # 前向传播
            predicted = torch.argmax(output, dim=1)  # 获取预测类别（取最大值的索引）
            total += classification.size(0)  # 累加总样本数
            correct += (predicted == classification).sum().item()  # 累加正确预测数
    accuracy = correct / total  # 计算准确率
    return accuracy  # 返回测试集准确率
