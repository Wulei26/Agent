import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class IMEDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_json(file_path, lines=True).to_dict(orient="records")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input = torch.tensor(self.dataset[index]["input"])
        target = torch.tensor(self.dataset[index]["target"])
        return input, target


# 获取数据集对象
def get_dataloadr(train: bool = True):
    file_path = config.TRAIN_DATA_FILE if train else config.TEST_DATA_FILE
    dataset = IMEDataset(file_path)
    # 创建dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader


def get_vocab(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        vocab_list = f.readlines()
    vocab_list = [s.strip() for s in vocab_list]
    return vocab_list


if __name__ == "__main__":
    dataloader = get_dataloadr(False)
    print(len(dataloader))
