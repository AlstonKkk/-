import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from glob import glob


# -------------------- 1. 模型定义 --------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


# -------------------- 2. 加载训练好的模型 --------------------
def load_trained_model(model_path):
    model = LeNet()
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# -------------------- 3. 图像预处理 --------------------
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        tensor = 1.0 - tensor  # 反色处理
        return tensor
    except Exception as e:
        print(f"预处理错误: {str(e)}")
        return None


# -------------------- 4. 执行推理 --------------------
def predict_digit(model, image_tensor):
    if image_tensor is None:
        return None, None
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        prob, pred = torch.max(outputs, 1)
        return pred.item(), prob.item(), time.time() - start_time


# -------------------- 5. 可视化结果 --------------------
def visualize_prediction(image_tensor, prediction, confidence, output_path):
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    img = image_tensor[0, 0].numpy()
    plt.imshow(img, cmap='gray')
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), [0] * 10, color='lightblue')
    plt.bar(prediction, confidence, color='red')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.title(f"Conf: {confidence:.2%}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = '/home/orangepi/daima/lenet.pth'
    IMAGE_DIR = '/home/orangepi/daima/picture_for_test1'  # 图片文件夹路径
    OUTPUT_DIR = '/home/orangepi/daima/fanhui_all/results1'  # 输出文件夹

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    model_load_start = time.time()
    model = load_trained_model(MODEL_PATH)
    print(f"模型加载耗时: {time.time() - model_load_start:.4f}s\n")

    # 获取图片列表
    image_paths = glob(os.path.join(IMAGE_DIR, '*.png')) + glob(os.path.join(IMAGE_DIR, '*.jpg'))

    # 处理每张图片
    for img_path in image_paths:
        print(f"处理图片: {os.path.basename(img_path)}")
        total_start = time.time()

        # 预处理
        preprocess_start = time.time()
        input_tensor = preprocess_image(img_path)
        preprocess_time = time.time() - preprocess_start

        if input_tensor is None:
            print("  图片预处理失败，跳过\n")
            continue

        # 推理
        pred_start = time.time()
        prediction, confidence, inference_time = predict_digit(model, input_tensor)
        pred_time = time.time() - pred_start

        # 可视化
        vis_start = time.time()
        output_name = f"{os.path.splitext(os.path.basename(img_path))[0]}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        visualize_prediction(input_tensor, prediction, confidence, output_path)
        print(f"IMAGE_SAVED:{output_path}")
        vis_time = time.time() - vis_start

        # 打印结果
        print(f"  预处理时间: {preprocess_time:.4f}s")
        print(f"  推理时间: {inference_time:.4f}s")
        print(f"  可视化时间: {vis_time:.4f}s")
        print(f"  总耗时: {time.time() - total_start:.4f}s")
        print(f"  识别完成，置信度 {confidence:.2%}\n")
        #最后只参考一下lenet1_x_piliang.py的时间就行