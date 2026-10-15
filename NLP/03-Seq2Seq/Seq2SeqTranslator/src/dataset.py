import os
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  # 用于将不同长度的长度的序列 padding到同一个长度
import config


class Translatordataset(Dataset):

    def __init__(self, file_path):
        self.dataset = pd.read_json(file_path, lines=True).to_dict(orient="records")

    def __len__(self):
        """DataLoader 的 __len__ 返回的是 batch 个数"""
        return len(self.dataset)

    def __getitem__(self, index):
        input = torch.tensor(self.dataset[index]["zh"])
        target = torch.tensor(self.dataset[index]["en"])
        return input, target


def collate_fn(data_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor]]:
    """
    data_list ： [(input, target)] 组成的元组列表

    return : 返回处理好的 ([input], [target])
    """
    input_lst = [item[0] for item in data_list]
    target_lst = [item[1] for item in data_list]

    # 对 input_lst进行填充
    input_tensor = pad_sequence(input_lst, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_lst, batch_first=True, padding_value=0)

    return input_tensor, target_tensor


def get_dataloadr(train: bool = True):
    file_path = config.TRAIN_DATA_FILE if train else config.TEST_DATA_FILE
    dataset = Translatordataset(file_path)
    # 创建dataloader,自定义 collate_fn ，将 input进行截取或者填充，到指定长度
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloadr(True)
    print(len(dataloader))
    for input, target in dataloader:
        print(input.size(), target.size())
        print(input)
        break
    dataloader = get_dataloadr(False)
    print(len(dataloader))
