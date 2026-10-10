import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from common.utils import seed_everything
from image_denoising.denoising_config import *
from image_denoising.denoising_data import create_dataloader
from image_denoising.denoising_engine import val_step
from image_denoising.denoising_model import ConvDenoiser


def test_batch(model, test_loader, device, save_path="denoising_result.png"):
    model.to(device)
    model.eval()
    # 1.获取数据加载器
    test_iter = iter(test_loader)
    noisy_image, image = next(test_iter)
    # 2. 推理预测
    with torch.no_grad():
        noisy_image = noisy_image.to(device)
        # 前向传播
        outputs = model(noisy_image)
    print("输出重构图片形状：", outputs.shape)
    # 3. 数据转换
    noisy_image = noisy_image.permute(0, 2, 3, 1).cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    # 4.画图
    fig, aexs = plt.subplots(nrows=3, ncols=10, figsize=(25, 4), sharex=True, sharey=True)
    for (
        imgs,
        ax_row,
    ) in zip([image, noisy_image, outputs], aexs):
        for img, ax in zip(imgs, ax_row):
            ax.imshow(img)
            ax.set_axis_off()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)  # 关闭图形，释放内存，避免命令行卡住/不退出
    print(f"结果已保存到: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # 2.导入数据
    _, test_loader = create_dataloader()
    # 3.定义模型
    model = ConvDenoiser()
    # 加载训练好的参数
    state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    model.load_state_dict(state_dict)
    print("模型加载完成...")

    # 4.测试
    print("测试结果如下：")
    test_batch(model, test_loader, device)

    # 5.平均误差
    test_loss = val_step(model=model, device=device, test_loader=test_loader, loss_fn=torch.nn.MSELoss())
    print("测试均方误差:", test_loss)
