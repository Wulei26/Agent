import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloadr
from predict import _load_model, _predict_batch
import config
from tokenizer import JiebaTokenizer
"""
使用测试集进行评估模型准确度
评估逻辑:
1.使用测试集的语料，每5个词预测 topk 个词
2.预测出的topk个词如果有一个正确那么就算预测准确
"""
def eva_batch():
    pass

def evaluate() -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建 tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_DATA_FILE)
    #  获取model
    model = _load_model(tokenizer,device)
    # 1.获取测试集合
    val_dataloader = get_dataloadr(train=False)
    total_count = 0 #因为我们构建 val_dataloader的时候drop_last了，所以这里要统计一共传入了多少，而不是测试集的总大小大
    num_correct = 0
    for inputs, targets in tqdm(val_dataloader):
        total_count += config.BATCH_SIZE
        inputs,targets = inputs.to(device), targets.to(device)
        # 前向传播进行计算
        positive_probas = _predict_batch(model,inputs)
        # 比较result和真实标签
        for target,proba in zip(targets,positive_probas):
            # print(target,proba)
            # 如果标签是正类，并且预测为正类的概率大于0.8，那么就认为预测正确了
            if target == 1.0 and proba > 0.8 :
                num_correct += 1
            # 如果标签是负类，并且预测为正类的概率小于0.2，那么就认为预测正确了
            if target == 0 and proba < 0.2:
                num_correct += 1
    # 计算出了所有正确个数
    return num_correct / total_count





if __name__=="__main__":
    print(evaluate())