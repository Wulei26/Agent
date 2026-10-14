import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class LSTMdataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_json(file_path, lines=True).to_dict(orient="records")

    def __len__(self):
        """DataLoader 的 __len__ 返回的是 batch 个数"""
        return len(self.dataset)

    def __getitem__(self, index):
        input = torch.tensor(self.dataset[index]["review"])
        target = torch.tensor(self.dataset[index]["label"]).float() # BCEWithLogitsLoss 要求 target必须是float
        return input, target


def get_dataloadr(train: bool = True):
    file_path = config.TRAIN_DATA_FILE if train else config.TEST_DATA_FILE
    dataset = LSTMdataset(file_path)
    # 创建dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloadr(True)
    print(len(dataloader))
    for input ,target in dataloader:
        print(input.size(), target.size())
        break
    dataloader = get_dataloadr(False)
    print(len(dataloader))