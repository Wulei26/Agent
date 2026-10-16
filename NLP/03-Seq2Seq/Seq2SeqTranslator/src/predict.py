import torch
from model import TranslatorModel
import config
from tokenizer import Tokenizer, ChineseTokenizer, EnglishTokenizer


# 预测核心逻辑：自回归生成 -> 模型前向传播，得到一批数据的预测概率
def predict_batch(model: TranslatorModel, inputs, tokenizer: Tokenizer, device):
    model.eval()
    model.to(device)
    # 1. 自回归生成
    with torch.no_grad():
        # 前向传播
        # 1. 编码，输出上下文向量 (1, N, hidden_size)
        context_vector = model.encoder(inputs)
        # 2. 解码
        # 2.1 构建第一个时间步的输入<SOS>，(N, L=1)
        batch_size = inputs.shape[0]
        decoder_input = torch.full((batch_size, 1), tokenizer.start_token_index).to(device)
        # 2.2 逐时间步自回归生成，最大长度SEQ_LEN
        generated = []
        # 保存批次中的每个序列是否已生成完毕（已有<EOS>）
        is_finished = torch.full([batch_size], False).to(device)
        for i in range(config.SEQ_LEN):
            # 2.2.1 解码器向前传播，现在拿到了每个时间步的初始隐状态(context_vector)，第一个时间步的输入(decoder_input)
            decoder_output, context_vector = model.decoder(decoder_input, context_vector)
            # 2.2.2 贪心解码：预测下一时间步的词id，(N, 1)
            next_token_ids = torch.argmax(decoder_output, dim=-1)
            # 2.2.3 保存到列表
            generated.append(next_token_ids)
            # 2.2.4 更新输入，将输出预测词id作为下一时间步输入
            decoder_input = next_token_ids
            # 2.3.5 判断序列生成是否结束
            is_finished |= next_token_ids.squeeze(1) == tokenizer.end_token_index
            if is_finished.all():
                break
        # 自回归生成结束，得到(N, 1)形状张量的列表
    # 2. 转换成 (N, L) 的tensor
    generated_tensor = torch.cat(generated, dim=1)
    # 3. 转换成列表，为截取不同长度的序列做准备
    generated_list = generated_tensor.tolist()
    # 4. 按照<EOS>的位置，截取真实的预测序列
    for index, sentence_id in enumerate(generated_list):
        # 寻找<EOS>的索引号，进行截取
        if tokenizer.end_token_index in sentence_id:
            eos_index = sentence_id.index(tokenizer.end_token_index)
            generated_list[index] = sentence_id[:eos_index]
    return generated_list  # 返回id列表的列表


# 预测函数：传入一串文本，得到接下来最可能的5个词
def predict(text, model, input_tokenizer, target_tokenizer, device):

    # 1. 处理文本，得到模型输入
    # 1.1 分词、id化
    ids = input_tokenizer.encode(text)
    # 1.2 转换为Tensor，形状 (N=1, L)
    inputs = torch.tensor([ids]).to(device)

    # 2. 调用核心预测逻辑
    result = predict_batch(model, inputs, target_tokenizer, device)

    # 3. 解码得到目标译文句子
    sentence = target_tokenizer.decode(result[0])

    return sentence


# 调用预测逻辑的应用程序
def run_predict():
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
    state_dict = torch.load(config.BEST_MODEL, map_location=device)
    model.load_state_dict(state_dict)
    print("模型加载成功！")

    print("欢迎使用中英翻译模型！输入 quit 或者 q 退出...")

    # 核心是一个死循环
    while True:
        # 等待用户输入
        user_input_text = input("中文：")

        # 如果是 quit 或者 q，就退出
        if user_input_text in ["q", "quit"]:
            print("欢迎下次使用！")
            break

        # 如果输入为空字符，继续输入
        if user_input_text.strip() == "":
            print("请输入有效内容...")
            continue

        # 调用预测逻辑，得到预测结果
        result = predict(user_input_text, model=model, input_tokenizer=zh_tokenizer, target_tokenizer=en_tokenizer, device=device)

        print("英文：", result)


if __name__ == "__main__":
    run_predict()
