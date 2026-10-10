
import torch
from image_classification.classification_model import ImageClassification # 模型
from image_classification.classification_engine import train_step, val_step, test_step # 训练、验证和测试函数
from image_classification.classification_data import create_dateset # 数据集创建函数
from image_classification.classification_config import (IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE, EPOCHS,
                                    TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, FULL_BATCH_SIZE,
                                    PACKAGE_NAME, CLASSIFIER_MODEL_NAME,SEED) # 配置参数
from common.utils import seed_everything

import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，否则使用CPU

seed_everything(SEED)  # 设置随机种子以确保实验可复现

# 1.创建数据集
train_dataset, val_dataset, test_dataset = create_dateset()  # 创建训练集、验证集和测试集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,drop_last=True)  # 创建训练数据加载器
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last=True)  # 创建验证数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last=True)  # 创建测试数据加载器

# 2. 定义模型
model = ImageClassification().to(device)  # 初始化模型并移动到设备

# 3. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # 使用AdamW优化器

# 4. 训练模型
min_val_loss = float('inf')  # 初始化最小验证损失为无穷大
for epoch in tqdm(range(EPOCHS), desc="训练进度", unit="epoch"):
    train_loss = train_step(model, train_loader, loss_fn, optimizer, device)  # 执行训练步骤
    val_loss = val_step(model, val_loader, loss_fn, device)  # 执行验证步骤

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")  # 输出训练和验证损失

    if val_loss < min_val_loss:  # 如果当前验证损失小于历史最小值
        min_val_loss = val_loss
        torch.save(model.state_dict(), CLASSIFIER_MODEL_NAME)  # 保存模型参数
        print(f"模型已保存到 {CLASSIFIER_MODEL_NAME}")  # 输出保存信息
    else:
        print("当前验证损失未降低，模型未保存")  # 输出未保存信息
print("训练完成，开始测试模型...")  # 输出训练完成信息