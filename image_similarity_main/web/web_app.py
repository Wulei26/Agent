# 导入必要的库
import base64  # Base64 编解码
from io import BytesIO  # 内存字节流
from pathlib import Path  # 路径处理库

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torchvision.transforms as T  # 图像预处理工具
from PIL import Image  # PIL图像处理库
from fastapi import FastAPI, Request  # FastAPI Web框架相关
from fastapi.responses import HTMLResponse, PlainTextResponse  # 响应类型
from fastapi.staticfiles import StaticFiles  # 静态文件服务

from image_denoising import denoising_config, denoising_model
from image_classification import classification_config, classification_model
from image_similarity import similarity_config, similarity_model, similarity_embeddings

# 当前文件所在目录（web/）及项目根目录
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# 创建 FastAPI 应用实例
app = FastAPI(title="智图寻宝 · 智能商品识别系统")

# 挂载静态资源目录
app.mount("/logo", StaticFiles(directory=str(BASE_DIR / "logo")), name="logo")
app.mount("/pictures", StaticFiles(directory=str(BASE_DIR / "pictures")), name="pictures")
app.mount("/dataset", StaticFiles(directory=str(ROOT_DIR / "common" / "dataset")), name="dataset")

# 打印启动信息
print("启动应用")

# 设备检测与设置（优先使用GPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("正在加载去噪模型")
denoiser = denoising_model.ConvDenoiser()
denoiser.load_state_dict(torch.load(
    str(ROOT_DIR / denoising_config.PACKAGE_NAME / denoising_config.DENOISER_MODEL_NAME),
    map_location=device))
denoiser.to(device)
print("去噪模型加载完毕")

print("正在加载分类模型")
classifier = classification_model.ImageClassification()
classifier.load_state_dict(torch.load(
    str(ROOT_DIR / classification_config.PACKAGE_NAME / classification_config.CLASSIFIER_MODEL_NAME),
    map_location=device))
classifier.to(device)
print("分类模型加载完毕")

print("正在加载嵌入模型")
encoder = similarity_model.ConvEncoder()  # 初始化编码器
# 加载编码器的预训练权重（自动处理设备映射）
encoder.load_state_dict(
    torch.load(
        str(ROOT_DIR / similarity_config.PACKAGE_NAME / similarity_config.ENCODER_MODEL_NAME),
        map_location=device))
encoder.to(device)  # 将模型移动到指定设备
print("嵌入模型加载完毕")

print("正在加载向量集合")
# 只需创建一次嵌入向量集合
# similarity_embeddings.create_embeddings(encoder)
collection = similarity_embeddings.get_collection(encoder)
print("向量集合加载完毕")

# 图像预处理流程（统一使用）
TRANSFORM = T.Compose([T.Resize((64, 64)), T.ToTensor()])


def _read_image(data: bytes) -> Image.Image:
    """将请求体字节流解码为 RGB 图像"""
    return Image.open(BytesIO(data)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    """将 PIL 图像编码为 base64 PNG 字符串"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# 首页路由
@app.get("/", response_class=HTMLResponse)
async def index():
    # 渲染首页模板
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


# 去噪路由：返回原始、噪声、去噪三张图的 base64
@app.post("/denoising")
async def get_denoised_image(request: Request):
    # 从请求体中读取图像字节
    data = await request.body()
    image = _read_image(data)

    # 应用预处理并转换为张量
    image_tensor = TRANSFORM(image)

    ## 向输入图像添加随机噪声
    # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
    noisy_img = image_tensor + denoising_config.NOISE_FACTOR * torch.randn(*image_tensor.shape)
    # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
    noisy_img = torch.clip(noisy_img, 0., 1.)

    # 增加批次维度
    noisy_img = noisy_img.unsqueeze(0)

    with torch.no_grad():
        # 模型推理
        noisy_img = noisy_img.to(device)
        denoised_image = denoiser(noisy_img)

    # 后处理
    denoised_image = denoised_image.squeeze(0).cpu()  # 移除批次维度
    denoised_image = denoised_image.permute(1, 2, 0).numpy() * 255  # CHW -> HWC并转换到0-255范围
    noisy_img = noisy_img.squeeze(0).cpu()
    noisy_img = noisy_img.permute(1, 2, 0).numpy() * 255

    # 转换为PIL图像
    denoised_image = Image.fromarray(denoised_image.astype("uint8"))
    noisy_img = Image.fromarray(noisy_img.astype("uint8"))

    return {
        "noisy_img": _encode_image(noisy_img),
        "denoised_image": _encode_image(denoised_image),
    }


# 分类路由：返回商品类型文本
@app.post("/classification", response_class=PlainTextResponse)
async def classification(request: Request):
    # 从请求体中读取图像字节
    data = await request.body()
    image = _read_image(data)

    # 应用预处理并转换为张量，增加批次维度
    image_tensor = TRANSFORM(image).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = classifier(image_tensor)

    label = classification_config.classification_names[np.argmax(logits.cpu().detach().numpy())]
    return "您搜索的商品类型是：" + label


# 相似图像计算路由（POST请求）
@app.post("/simimages")
async def simimages(request: Request):
    # 从请求体中读取图像字节
    data = await request.body()
    image = _read_image(data)

    # 应用预处理并转换为张量
    image_tensor = TRANSFORM(image)

    # 计算相似图像索引
    indices_list = similarity_embeddings.search_similarity_image_ids(
        collection, image_tensor, cnt=5
    )
    # 返回JSON格式的响应
    return {"indices_list": indices_list}


# 主程序入口
if __name__ == "__main__":
    import uvicorn  # ASGI服务器

    # 启动FastAPI应用，监听9000端口
    uvicorn.run(app, host="127.0.0.1", port=9000)
