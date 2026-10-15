from pathlib import Path

# 目录参数
ROOT_DIR = Path(__file__).parent.parent.absolute()  # 项目根目录
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
# 文件名
RAW_DATA_FILE = RAW_DIR / "cmn.txt"
TRAIN_DATA_FILE = PROCESSED_DIR / "train.jsonl"
TEST_DATA_FILE = PROCESSED_DIR / "test.jsonl"

# 模型文件
BEST_MODEL = MODEL_DIR / "Seq2Seq.pt"
ZH_VOCAB_FILE = MODEL_DIR / "zh_vocab.txt"
EN_VOCAB_FILE = MODEL_DIR / "en_vocab.txt"

# 特殊字符
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"

# 训练超参数
TEST_RATE = 0.2
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
SEQ_LEN = 128


EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
