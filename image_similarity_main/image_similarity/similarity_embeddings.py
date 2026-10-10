import re
import os
import torch
import numpy as np
import chromadb
import pandas as pd
from math import ceil
from tqdm import tqdm
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import Image
from PIL import Image as PILImage
import torchvision.transforms.transforms as T
from similarity_data import sorted_alphanumeric
from similarity_model import ConvEncoder
from similarity_config import (
    ENCODER_MODEL_NAME,
    CHROMA_BACKEND_PATH,
    CHROMA_INSERT_BATCH,
    PACKAGE_NAME,
    CHROMA_COLLECTION_NAME,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_PATH,
    USE_HTTP_SERVICE,
)

# 训练好的Encoder模型，可以直接将图片通过encoder进行编码，然后存放进入向量数据库
"""
功能1 ： 将所有图片经过 encoder之后 存入向量数据库
功能2 ： 输入新的图片，去向量数据库中进行查询，返回最相似的向量
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，否则使用CPU


# 1.加载模型
def _load_encoder() -> ConvEncoder:
    model = ConvEncoder().to(device)
    model.load_state_dict(torch.load(ENCODER_MODEL_NAME, map_location=device))  # 加载模型参数
    return model


def _load_id_to_image(main_dir: str, transform):
    id2image = {}
    image_names = sorted_alphanumeric(os.listdir(main_dir))
    # classification = pd.read_csv(label_path)  # 读取分类标签CSV文件
    with tqdm(total=len(image_names)) as pbar:
        for i, image_name in enumerate(image_names):
            # 拼接完整图像路径
            img_loc = os.path.join(main_dir, image_name)
            # 打开图像并转换为RGB格式
            image = PILImage.open(img_loc).convert("RGB")

            # img_flag_id = self.label_dict[idx]  # 在分类数据中查找对应的标签

            # 对图像进行预处理（如果定义了 transform）
            if transform is not None:
                tensor_image = transform(image)  # 应用预处理操作，将图像转换为张量
            else:
                raise RuntimeError("transform参数不能为None，需指定预处理方法")
            # 再将张量转化为 ndarray存入字典
            id2image[str(i)] = tensor_image.numpy()
            pbar.update(1)
    return id2image


# 2. 自定义 Embedding 函数
class ImageEmbeddingFunction(EmbeddingFunction[Image]):

    def __init__(self, model):
        self.model = model

    def __call__(self, input: Image) -> Embeddings:
        # 将输入image转化为Tensor
        input_tensor = torch.tensor(np.array(input))
        # 通过model前向传播
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
        return output.numpy()


def get_collection(encoder: ConvEncoder):
    # 1.创建客户端
    client = chromadb.PersistentClient(
        path=os.path.join(
            "..",
            PACKAGE_NAME,
            CHROMA_BACKEND_PATH,
        )
    )
    if USE_HTTP_SERVICE:
        client = chromadb.HttpClient(host="localhost", port=8999)

    # 2.创建集合
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME, embedding_function=ImageEmbeddingFunction(model=encoder)
    )
    return collection


def create_embeddings():
    transform = T.Compose(
        [
            T.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小为指定高度和宽度
            T.ToTensor(),  # 将图像转换为张量
        ]
    )
    # 1.获取图像id和图像数据
    id2image = _load_id_to_image(main_dir=IMG_PATH, transform=transform)
    print("图片加载完成...")
    # 2.获取集合
    encoder = _load_encoder()
    collection = get_collection(encoder=encoder)
    # 3.分批次存入chroma数据库，自动生成嵌入向量
    print(" 正在写入向量数据库...")
    ids = list(id2image.keys())
    images = list(id2image.values())
    for i in range(ceil(len(ids) / CHROMA_INSERT_BATCH)):
        start = i * CHROMA_INSERT_BATCH
        end = (i + 1) * CHROMA_INSERT_BATCH
        collection.upsert(
            ids=ids[start : min(end, len(ids))],
            images=images[start : min(end, len(ids))],
        )
    print("向量写入完成")


def search_similarity_image_ids(collection, image_tensor, cnt):
    result = collection.query(
        query_images=[image_tensor.numpy()],
        n_results=cnt,
    )
    # 返回所有的id
    ids = [int(id) for id in result["ids"][0]]
    return ids


if __name__ == "__main__":
    # model = _load_encoder()
    # print(model)
    # transform = T.Compose(
    #     [
    #         T.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小为指定高度和宽度
    #         T.ToTensor(),  # 将图像转换为张量
    #     ]
    # )
    # id_dict = _load_id_to_image(main_dir=IMG_PATH, transform=transform)
    # print(id_dict)
    create_embeddings()