import numpy as np
import os
import torch
import random

def seed_everything(seed:str):   
    """
    为了保证训练过程可复现，使用确定的随机数种子。对 torch，numpy 和 random 都使用相同的种子。

    参数:
    - seed: 随机数种子（整数）
    """

    random.seed(seed)  # 设置Python内置随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN操作确定性
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN性能优化