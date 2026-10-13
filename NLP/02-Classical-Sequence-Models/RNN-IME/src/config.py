from pathlib import Path

# 目录参数
ROOT_DIR = Path(__file__).parent.parent.absolute()  # 项目根目录
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
# 文件名
RAW_DATA_FILE = RAW_DIR / "synthesized_.jsonl"
TRAIN_DATA_FILE = PROCESSED_DIR / "train.jsonl"
TEST_DATA_FILE = PROCESSED_DIR / "test.jsonl"

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WINDOWS = 5

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256

# 模型文件
BEST_MODEL = MODEL_DIR / "IME.pt"
VOCAB_DATA_FILE = MODEL_DIR / "vocab.txt"

# 特殊字符
UNK_TOKEN = "<UNK>"
