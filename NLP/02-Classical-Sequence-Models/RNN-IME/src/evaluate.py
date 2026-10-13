import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloadr, get_vocab
from predict import _load_model, _process_topk
import config
"""
使用测试集进行评估模型准确度
评估逻辑:
1.使用测试集的语料，每5个词预测 topk 个词
2.预测出的topk个词如果有一个正确那么就算预测准确
"""
def eva_batch():
    pass

def evaluate(topk:int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取词表
    vocab  = get_vocab(config.VOCAB_DATA_FILE)
    # word2id = {word:id for id , word in enumerate(vocab)}
    #  获取model
    model = _load_model(len(vocab),device)
    # 1.获取测试集合
    val_dataloader = get_dataloadr(train=False)
    total_count = 0 #因为我们构建 val_dataloader的时候drop_last了，所以这里要统计一共传入了多少，而不是测试集的总大小大
    top1_correct = 0 # 预测的第一个就是正确的个数
    topk_correct = 0 # 预测的k个候选中有正确答案的个数
    for inputs, targets in tqdm(val_dataloader):
        total_count += config.BATCH_SIZE
        inputs,targets = inputs.to(device), targets.to(device)
        # 前向传播计算后面一个token
        top_ids = _process_topk(model=model,input=inputs,K=topk)
        # 看这个top_ids是否有target 注意这里的top_ids是（batch_size,topk)
        for target_id, topk_id in zip(targets,top_ids):
            if target_id == topk_id[0]:
                top1_correct += 1
            if target_id in topk_id:
                topk_correct += 1
        
    # 计算出了所有正确个数
    return (top1_correct / total_count, topk_correct / total_count)





if __name__=="__main__":
    print(evaluate(10))