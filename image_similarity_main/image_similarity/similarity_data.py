__all__ = ["ImageDataset"]

# 导入必要的库
from PIL import Image  # 图像处理库
import os  # 操作系统接口库
from torch.utils.data import Dataset  # PyTorch数据集基类
import pandas as pd
from similarity_config import *  # 导入分类配置和图像尺寸
import re
import torchvision.transforms as T  # 图像预处理变换库
from torch.utils.data import random_split  # 数据集随机划分函数


def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    # 定义转换函数：将数字部分转换为整数，非数字部分转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)


class ImageDataset(Dataset):
    """
    从图像文件夹创建PyTorch数据集，返回图像的张量表示

    参数:
    - main_dir : 图片存储路径（字符串）
    - transform (可选) : 图像预处理变换（如torchvision.transforms）
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir  # 保存主目录路径
        self.transform = transform  # 保存图像预处理变换
        # 获取主目录下的所有文件名，并按字母数字混合顺序排序
        self.total_imgs = sorted_alphanumeric(os.listdir(main_dir))

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """
        根据索引获取图像样本

        参数:
        - idx : 样本索引（整数）

        返回:
        - 图像张量（经过预处理）
        """
        """加载并返回指定索引的图像张量（输入和目标相同，适用于自编码器）"""
        # 拼接完整图像路径
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        # 打开图像并转换为RGB格式
        image = Image.open(img_loc).convert("RGB")

        # 对图像进行预处理（如果定义了 transform）
        if self.transform is not None:
            tensor_image = self.transform(image)  # 应用预处理操作，将图像转换为张量
        else:
            raise RuntimeError("transform参数不能为None，需指定预处理方法")
        # 返回两次相同张量（输入和目标相同，用于自编码器训练，也就是说原始图片既作为输入也作为标签，进行自训练）
        return tensor_image, tensor_image


def create_dateset():
    transform = T.Compose(
        [
            T.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小为指定高度和宽度
            T.ToTensor(),  # 将图像转换为张量
        ]
    )
    dataset = ImageDataset(main_dir=IMG_PATH, transform=transform)  # 创建数据集实例
    train_dataset, val_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO])  # 按比例划分训练集和验证集
    return dataset, train_dataset, val_dataset  # 返回训练集、验证集


if __name__ == "__main__":
    train_dataset, val_dataset = create_dateset()  # 创建数据集
    print(f"训练集样本数: {len(train_dataset)}")  # 打印训练集样本数
    print(f"验证集样本数: {len(val_dataset)}")  # 打印验证集样本数
