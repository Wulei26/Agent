import torch
from model import GRUSentimentModel
import config
from tokenizer import JiebaTokenizer


def _load_model(tokenizer:JiebaTokenizer, device):
    model = GRUSentimentModel(tokenizer.vocab_size,tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL, map_location=device))  # 加载模型参数
    return model


def _predict_batch(model, input: str):
    model.eval()
    with torch.no_grad():
        outputs = model(input)

    # outputs 的形状为 (N,)
    # 直接转化为正类的概率
    proba = torch.sigmoid(outputs)

    return proba.tolist()



def predict(text: str) -> float:
    """
    推理过程，输入一段话得到是正类的概率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.获取词表
    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_DATA_FILE)
    # 2. 分词
    token_id_lst = tokenizer.encode(text)
    # 3. 转化为 tensor
    inputs = torch.tensor(token_id_lst)  # 注意这里要增加一个batch维度
    inputs = inputs.unsqueeze(0).to(device)
    # 4.加载模型前项传播
    model = _load_model(tokenizer, device)

    # 进行预测
    results = _predict_batch(model=model,input=inputs)

    return results[0]


if __name__ == "__main__":
    result = predict("东西很好，下次还买")
    if result > 0.5:
        print(f"好评：置信度为 {result}")
    if result <= 0.5:
        print(f"差评：置信度为 {1 - result}")
