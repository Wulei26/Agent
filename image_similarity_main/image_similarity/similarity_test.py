import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms as T
from common.utils import seed_everything
from similarity_config import SEED,IMG_PATH,IMG_HEIGHT,IMG_WIDTH
from similarity_data import create_dateset
from similarity_embeddings import _load_encoder,search_similarity_image_ids,get_collection

# seed_everything(SEED)  # 设置随机种子

if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小为指定高度和宽度
            T.ToTensor(),  # 将图像转换为张量
        ]
    )
    # 获取数据集
    dataset,_,val_dataset = create_dateset()
    print(f"验证集样本数: {len(val_dataset)}")
    _,image = val_dataset[0]
    print(f"第一张图片的形状: {image.shape}")
    # 1. 加载训练好的模型
    print("加载训练好的模型...")
    model = _load_encoder()
    model.eval()  # 设置模型为评估模式
    print("模型加载完成，开始进行图像嵌入生成...")
    # 3. 获取Chroma集合
    collection = get_collection(encoder=model)
    # # 4. 查询相似图像的id
    print("正在查询相似图像...")
    similar_image_ids = search_similarity_image_ids(collection, image, cnt=5)
    print(f"查询到的相似图像id: {similar_image_ids}")

    # 5. 可视化查询结果
    # 获取原始图片ID
    original_idx = val_dataset.indices[0]
    image_name = val_dataset.dataset.total_imgs[original_idx]
    image_id = os.path.splitext(image_name)[0]

    # 绘制图片
    plt.figure(figsize=(10, 5))

    # 显示查询图片
    plt.subplot(2, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Query Image ID: {image_id}")
    plt.axis("off")

    # 显示相似图片
    for i, img_id in enumerate(similar_image_ids):
        img_path = os.path.join("../common/dataset/", f"{img_id}.jpg")

        similar_image = Image.open(img_path).convert("RGB")

        # 使用与查询图片相同的预处理
        similar_image = transform(similar_image)

        plt.subplot(2, 3, i + 2)
        plt.imshow(similar_image.permute(1, 2, 0).numpy())
        plt.title(f"Image ID: {img_id}")
        plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("similarity_results.png", dpi=150)