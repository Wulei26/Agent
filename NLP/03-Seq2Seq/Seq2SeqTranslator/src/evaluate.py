import torch
from tqdm import tqdm

from predict import predict_batch
import config
from model import TranslatorModel
from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer, Tokenizer

from nltk.translate.bleu_score import corpus_bleu


# 评估核心逻辑，返回BLEU值
def evaluate(model, dataloader, tokenizer: Tokenizer, device):
    model.eval()
    # 定义预测输出和目标译文的列表
    predictions = []
    references = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="评估"):
            inputs, targets = inputs.to(device), targets.tolist()
            # 前向传播，直接调用预测方法，得到一组预测结果（id列表）
            batch_result = predict_batch(model, inputs, tokenizer, device)
            # 将预测结果和目标添加到列表
            predictions.extend(batch_result)
            references.extend([[target[1 : target.index(tokenizer.end_token_index)]] for target in targets])
    # 调用评分函数，得到BLEU值
    return corpus_bleu(references, predictions)


# 运行评估流程
def run_evaluate():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.ZH_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(config.EN_VOCAB_FILE)
    print("词表加载成功！")

    # 3. 加载模型
    model = TranslatorModel(
        input_vocab_size=zh_tokenizer.vocab_size,
        input_padding_idx=zh_tokenizer.pad_token_index,
        target_vocab_size=en_tokenizer.vocab_size,
        target_padding_idx=en_tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL, map_location=device))
    print("模型加载成功！")

    # 4. 定义测试集加载器
    test_dataloader = get_dataloader(train=False)

    # 5. 调用评估逻辑
    bleu = evaluate(model, test_dataloader, tokenizer=en_tokenizer, device=device)
    print("评估结果：")
    print("BLEU值: {:.6f}".format(bleu))


if __name__ == "__main__":
    run_evaluate()
