import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # 处理缺失值
from sklearn.pipeline import Pipeline  # 管道处理
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader


def process_dataset():
    """构建数据集"""
    data_dir = Path(r"D:\BaiduNetdiskDownload\DL\2.资料\data")
    data = pd.read_csv(data_dir / "house_prices.csv")
    # 删除ID列
    data.drop(["Id"], axis=1, inplace=True)
    # 划分特征和标签
    X = data.drop(["SalePrice"], axis=1)
    y = data["SalePrice"]
    # 筛选出数值型特征
    print(X.dtypes)
    numberical_features = X.select_dtypes(exclude="object").columns
    # 筛选类别信息
    categorical_features = X.select_dtypes(include=["object", "str"]).columns
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 特征预处理
    numberical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy="mean")),
            ("std", StandardScaler()),
        ]
    )
    # 类别型特征先将缺失值替换为字符串"NaN"，再进行独热编码
    categorical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy="constant", fill_value="NaN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    #  组合特征预处理器
    preprossor = ColumnTransformer(
        transformers=[
            ("num", numberical_transformer, numberical_features),
            ("category", categorical_transformer, categorical_features),
        ]
    )
    x_train = pd.DataFrame(preprossor.fit_transform(x_train).toarray(), columns=preprossor.get_feature_names_out())
    x_test = pd.DataFrame(preprossor.transform(x_test).toarray(), columns=preprossor.get_feature_names_out())
    ## 构建Tensor数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return train_dataset, test_dataset, x_train.shape[1]


train_dataset, test_dataset, feature_num = process_dataset()
# 构建模型
model = nn.Sequential(
    nn.Linear(in_features=feature_num, out_features=128),
    # 批量标准化
    nn.BatchNorm1d(128),
    nn.ReLU(),
    # 加一个dropout层
    nn.Dropout(0.2),  # 0.2概率失活
    nn.Linear(128, 1),  # 输出为 预测值
)


# 定义损失函数，使用对数损失函数
def log_rmse(pred, target):
    mse = nn.MSELoss()
    pred.squeeze_()  # 分类模型输出 (batch, 1)，压缩成 (batch,)
    pred = torch.clamp(pred, min=1, max=float("inf"))
    return torch.sqrt(mse(torch.log(pred), torch.log(target)))


if __name__ == "__main__":
    process_dataset()
    import pandas as pd

    print(pd.__version__)

    # 看看你的列实际是什么 dtype
    # print(X.dtypes)
