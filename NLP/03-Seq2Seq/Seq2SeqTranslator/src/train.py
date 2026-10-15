import time

import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import get_dataloader
from model import TranslatorModel
import config
from tokenizer import ChineseTokenizer, EnglishTokenizer

from torch.utils.tensorboard import SummaryWriter

# 定义函数：训练一个epoch，返回平均损失
def train_one_epoch(model:TranslatorModel, device, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    # 按批次迭代训练集数据
    for inputs, targets in tqdm(dataloader, desc="train..."):
        inputs, targets = inputs.to(device), targets.to(device)
        # 1.前向传播：训练过程的前向传播比较复杂
        # 1.1 编码阶段-得到上下文状态
        context_vector = model.encoder(inputs)
        # 1.2. 解码器
        # 1.2.1 由于inputs和targets 的形状都是 (N,L)，里面包含了 开头的 eos 和末尾的 sos
        decoder_inputs = targets[:,: -1] #去掉末尾的 <eos>
        decoder_targets = targets[:, 1:] #去掉开头的 <sos>
        # 1.2.2 解码器前向传播
        # decoder_outputs (N, SEQ_LEN, V_tgt) V_tgt：目标语言词表大小
        decoder_outputs, hn = model.decoder(decoder_inputs,context_vector)


        # 2.计算损失
        # decoder_outputs 形状为：(N, SEQ_LEN, V_tgt)，decoder_targets 为 (N,V_tgt)，所以要进行维度转换
        loss = loss_fn(decoder_outputs.transpose(1, 2),decoder_targets)
        # 3. 反向传播，计算梯度
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练流程
def train():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 构建数据集加载器
    train_loader = get_dataloader()

    # 3. 创建分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)

    # 4. 定义模型
    model = TranslatorModel(input_vocab_size=zh_tokenizer.vocab_size, target_vocab_size=en_tokenizer.vocab_size,
                             input_padding_idx=zh_tokenizer.pad_token_index, target_padding_idx=en_tokenizer.pad_token_index)
    model.to(device)

    # 5. 定义损失函数
    """
    ignore_index 告诉损失函数：当目标值为 pad_token_index 时，不计算这个位置的损失。因为：padding 是"凑长度"用的，不是真实标签
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 6. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 定义一个写入器
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 7. 开始训练
    min_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}: ")
        train_loss = train_one_epoch(model=model, device=device, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Train Loss: {train_loss}")

        writer.add_scalar('Loss/train', train_loss, epoch)

        # 判断是否需要保存模型
        if train_loss < min_loss:
            state_dict = model.state_dict()
            torch.save(state_dict, config.BEST_MODEL)
            min_loss = train_loss
            print("模型保存成功！")

    writer.close()

if __name__ == '__main__':
    train()