# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset"]

import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from denoising_config import NOISE_FACTOR, IMG_PATH
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from denoising_config import TRAIN_RATIO, VAL_RATIO, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE


def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)


class ImageDataset(Dataset):  # 定义一个名为ImageDataset的类，继承自PyTorch的Dataset类
    def __init__(
        self,
        main_dir,
        transform=None,
    ):  # 定义类的构造函数，接受两个参数：main_dir（主目录路径）和transform（图像预处理操作，默认为None）
        self.main_dir = main_dir  # 将主目录路径保存为类的属性
        self.transform = transform  # 将图像预处理操作保存为类的属性
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))  # 获取主目录下的所有文件名，并保存为列表

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.all_imgs[index])
        image = Image.open(img_loc).convert("RGB")  # 使用PIL库打开图像并将其转换为RGB格式
        if self.transform is not None:  # 如果定义了图像预处理操作
            tensor_image = self.transform(image)  # 对图像进行预处理，并将其转换为张量
        else:
            # 若无变换，抛出异常提示必须提供预处理
            raise ValueError("transform参数不能为None，需指定预处理方法")
        ## 向输入图像添加随机噪声
        # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
        noisy_imgs = tensor_image + torch.randn_like(tensor_image) * NOISE_FACTOR
        # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
        noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)
        return noisy_imgs, tensor_image  # 返回预处理后的图像张量（将噪声图片作为输入，将原图作为目标）


def create_dataloader():
    transforms = T.Compose(
        [
            T.Resize((64, 64)),
            T.ToTensor(),
        ]
    )
    dataset = ImageDataset(IMG_PATH, transforms)
    train_dataset, test_dataset = random_split(dataset, lengths=[TRAIN_RATIO, VAL_RATIO])
    train_loader = DataLoader(
        train_dataset,
        TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    dataset = ImageDataset(main_dir=IMG_PATH)
    print(dataset.all_imgs[0:100])
