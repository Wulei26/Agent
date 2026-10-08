import torch
from classification_data import create_dateset # 数据集创建函数
from classification_config import  TEST_BATCH_SIZE # 配置参数
from classification_engine import test_step # 测试函数
_, _, test_dataset = create_dateset()  # 创建训练集、验证集和测试集
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,drop_last=True)  # 创建测试数据加载器

# 加载模型
from classification_model import ImageClassification # 模型
from classification_config import CLASSIFIER_MODEL_NAME # 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #检查是否有可用的GPU，否则使用CPU
model = ImageClassification().to(device)
model.load_state_dict(torch.load(CLASSIFIER_MODEL_NAME, map_location=device))  # 加载模型参数
model.eval()  # 设置模型为评估模式

test_loss = test_step(model, test_loader, device)  # 执行测试步骤
print(f"测试集准确率: {test_loss:.4f}")  # 输出
