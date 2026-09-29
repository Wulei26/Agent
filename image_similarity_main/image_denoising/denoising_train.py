import torch
import torch.nn as nn
from tqdm import tqdm
from denoising_data import create_dataloader
from denoising_model import ConvDenoiser
from denoising_engine import train_step, val_step
from denoising_config import LEARNING_RATE, EPOCHS, DENOISER_MODEL_NAME

# 1.定义设备
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# 2.导入数据
train_loader, test_loader = create_dataloader()

# 3.定义模型
denoiser = ConvDenoiser()
denoiser.to(device)

# 4.定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)

min_val_loss = float("inf")

for epoch in tqdm(range(EPOCHS)):
    # 执行一个epoch训练
    train_loss = train_step(
        model=denoiser,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
    # 打印当前epoch的训练损失
    print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")
    # 执行一个验证流程
    val_loss = val_step(
        model=denoiser,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    # 如果验证损失shangshang1，那么就保存模型
    if val_loss < min_val_loss:
        min_loss = val_loss
        # 保存编码器和解码器状态字典
        torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)
    print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")
# 训练结束提示
print("\n==========> 训练结束 <==========\n")
