import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn as nn
from similarity_config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    ENCODER_MODEL_NAME,
    DECODER_MODEL_NAME,
)

# 导入优化器模块
import torch.optim as optim
from tqdm import tqdm
from similarity_model import ConvDecoder, ConvEncoder
from similarity_engine import train_step, val_step

import torchvision.transforms as T
from similarity_data import create_dateset
from common.utils import seed_everything

transform = T.Compose(
    [
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
    ]
)
if torch.cuda.is_available():
    device = "cuda"  # 优先使用GPU
else:
    device = "cpu"  # 回退到CPU
# 1. 创建数据集
full_dataset, train_dataset, val_dataset = create_dateset()
# 2. 创建 data_loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    drop_last=True,
)

# 3. 定义损失函数，使用均方误差
loss_fn = nn.MSELoss()

# 4. 创建模型
encoder = ConvEncoder().to(device)
decoder = ConvDecoder().to(device)

# 5.定义优化器
autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)

min_loss = 999

# 6.开始训练
for epoch in tqdm(range(EPOCHS), desc="训练进度", unit="epoch"):
    train_loss = train_step(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
    print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")
    val_loss = val_step(
        encoder,
        decoder,
        val_loader,
        loss_fn,
        device=device,
    )
    if val_loss < min_loss:
        print("验证集的损失减小了，保存新的最好的模型。")
        min_loss = val_loss
        # 保存编码器和解码器状态字典
        torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
        torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
    else:
        print("验证集的损失没有减小，不保存模型。")
    # 打印验证损失
    print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")
