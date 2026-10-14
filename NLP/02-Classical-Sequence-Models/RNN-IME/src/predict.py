import torch
import jieba
from model import IMEModel
import config
from dataset import get_vocab
from tokenizer import JiebaTokenizer


def _load_model(vocab_size, device):
    model = IMEModel(vocab_size).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL, map_location=device))  # 加载模型参数
    return model


def _process_topk(model, input: str, K=5):
    model.eval()
    with torch.no_grad():
        outputs = model(input)
    # 取前 topk个token
    top_ids = torch.topk(outputs, k=K, dim=-1).indices.tolist()

    return top_ids
    # sentence_id = []
    # for token in jieba.lcut(text):
    #     token_id = word2id.get(token, word2id[UNK_TOKEN])
    #     sentence_id.append(token_id)
    # sentences_id.append(sentence_id)


def predict(text: str, K=5) -> list[str]:
    """
    推理过程，输入一段话得到接下来最有可能出现的5个词
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.获取词表
    tokenizer = JiebaTokenizer.from_vocab(config.VOCAB_DATA_FILE)
    # vocab = get_vocab(config.VOCAB_DATA_FILE)
    # word2id = {word: id for id, word in enumerate(vocab)}
    # 2. 分词
    # token_lst = jieba.lcut(text)
    # token_id_lst = [word2id.get(i, word2id[config.UNK_TOKEN]) for i in token_lst]
    token_id_lst = tokenizer.encode(text)
    # 3. 转化为 tensor
    inputs = torch.tensor(token_id_lst)  # 注意这里要增加一个batch维度
    inputs = inputs.unsqueeze(0).to(device)
    # 4.加载模型前项传播
    model = _load_model(tokenizer.vocab_size, device)

    topk_ids = _process_topk(model, inputs, K=K)
    # 将id转为token
    # topk_ids = topk_ids.squeeze(0)
    topk_tokens = tokenizer.decode(topk_ids[0])
    return topk_tokens


if __name__ == "__main__":
    print(predict("我们公司最近在尝试用"))
