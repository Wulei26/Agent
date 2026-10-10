__all__ = ["train_step", "val_step"]

import torch


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    执行一个完整的训练迭代

    参数:
    - encoder: 卷积编码器（如ConvEncoder）
    - decoder: 卷积解码器（如ConvDecoder）
    - train_loader: 训练数据加载器，提供批次化的（输入图像, 目标图像）
    - loss_fn: 损失函数（如MSE）
    - optimizer: 优化器（如Adam）
    - device: 计算设备（"cuda" 或 "cpu"）

    返回值:
    - 当前epoch的平均训练损失（标量值）
    """
    encoder.train()
    decoder.train()
    total_loss = 0.0  # 平均损失
    num_batches = 0  # 批次计数器
    for train_img, target_img in train_loader:
        train_img, target_img = train_img.to(device), target_img.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # encoder 对图片进行编码，返回图片的潜在表示
        encoder_output = encoder(train_img)
        # decoder 对编码后的向量进行还原
        decoder_output = decoder(encoder_output)
        # 计算decoder之后的向量和target之间的损失
        loss = loss_fn(decoder_output, target_img)
        # 反向传播计算梯度
        loss.backward()
        # optimizer 更新参数
        optimizer.step()
        # 计算当前batch损失
        num_batches += 1
        total_loss += loss.item()
    return (
        total_loss / num_batches
    )  # 因为这里构建loader之后，会drop_last, 所以total_loss就是每个batch平均损失的和，再除以 batch_num就是平均损失


def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    执行验证步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 验证集的平均损失（标量值）
    """
    total_loss = 0.0
    num_batchs = 0
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for val_img, target_img in val_loader:
            val_img, target_img = val_img.to(device), target_img.to(device)
            # 前向传播
            encoder_out = encoder(val_img)
            decoder_out = decoder(encoder_out)
            # 计算损失
            loss = loss_fn(decoder_out, target_img)
            # 累加损失
            total_loss += loss.item()
            num_batchs += 1
    return total_loss / num_batchs
