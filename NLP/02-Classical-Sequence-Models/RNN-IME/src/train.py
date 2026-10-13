import torch
import torch.nn as nn
from tqdm import tqdm
from model import IMEModel
from torch.optim import Adam
from dataset import get_dataloadr, get_vocab
import config


def train_one_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    model.to(device)

    total_loss = 0.0
    for input, target in tqdm(dataloader, desc="训练"):
        input, target = input.to(device), target.to(device)
        # 1.前向传播
        output = model(input)
        # 2. 计算损失
        batch_loss = loss_fn(output, target)
        # 3. 梯度清零
        optimizer.zero_grad()
        # 4. 反向传播计算梯度
        batch_loss.backward()
        # 5.更新参数
        optimizer.step()

        total_loss += batch_loss.item()
    return total_loss / len(dataloader)


# def val_one_epoch()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.加载数据集
    train_loader = get_dataloadr(train=True)
    # 2. 构建模型
    vocab_list = get_vocab(config.VOCAB_DATA_FILE)
    model = IMEModel(vocab_size=len(vocab_list))
    # 3. 定义损失函数 CrossEntroLoss
    loss_fn = nn.CrossEntropyLoss()
    # 4. 定义优化器
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 5.训练模型
    min_loss = float("inf")
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        print(f"train loss: {train_loss} on Epoch { epoch + 1}")
        if train_loss < min_loss:
            state_dict = model.state_dict()
            torch.save(state_dict, config.BEST_MODEL)
            min_loss = train_loss
        else:
            print("损失未减小..不保存模型")


if __name__ == "__main__":
    train()
