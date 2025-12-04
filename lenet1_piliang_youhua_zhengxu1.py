import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
import torchvision.transforms as transforms
# matplotlib.use('Agg')
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
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: 1 - x)  # 反色处理（根据训练数据特性）,如果是黑底白字需要注释掉
    ])

    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0)

    return tensor


# -------------------- 4. 执行推理 --------------------
def predict_digit(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    start_time = time.time()
    probs = torch.nn.functional.softmax(outputs, dim=1)
    prob, pred = torch.max(probs, 1)

    return pred.item(), prob.item(),  time.time() - start_time



# -------------------- 5. 可视化结果 --------------------
def visualize_prediction(image_tensor, prediction, confidence, output_path):
    plt.figure(figsize=(6, 3))

    # 显示图像
    plt.subplot(1, 2, 1)
    img = image_tensor[0, 0].numpy()
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    # 显示预测结果
    plt.subplot(1, 2, 2)
    plt.bar(range(10), [0] * 10, color='lightblue')
    plt.bar(prediction, confidence, color='red')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)
    #这下面的四个每两个互相更改，这里的逻辑很重要
    # plt.show()
    # #plt.close()

    # #plt.show()
    # plt.close()

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 配置路径
    bu1 = time.time()
    MODEL_PATH = '/home/orangepi/daima/lenet.pth'
    IMAGE_DIR = '/home/orangepi/daima/baidi_heizi'  # 图片文件夹路径
    OUTPUT_DIR = '/home/orangepi/daima/fanhui_all/results3'  # 输出文件夹

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    model_load_start = time.time()
    model = load_trained_model(MODEL_PATH)
    print(f"模型加载耗时: {time.time() - model_load_start:.4f}s\n")
    supported_formats = ('.png', '.jpg')
    # 获取图片列表
    #image_paths = glob(os.path.join(IMAGE_DIR, '*.png')) + glob(os.path.join(IMAGE_DIR, '*.jpg'))
    # 获取文件名列表并排序
    filenames = [f for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith(('.png', '.jpg'))]  # 直接写扩展名更直观


    # 定义排序函数（处理纯文件名）
    def extract_leading_number(filename):
        import re
        match = re.match(r'^(\d+)', filename)
        return int(match.group(1)) if match else float('inf')


    filenames_sorted = sorted(filenames, key=extract_leading_number)

    # 生成完整路径列表
    image_paths = [os.path.join(IMAGE_DIR, f) for f in filenames_sorted]
    # 处理每张图片
    bu2  = time.time()
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


        # 打印结果
        print(f"  预处理时间: {preprocess_time:.4f}s")
        print(f"  推理时间: {inference_time:.4f}s")
        vis_time = time.time() - vis_start
        print(f"  可视化时间: {vis_time:.4f}s")
        print(f"识别结果: 数字 {prediction}, 置信度 {confidence:.2%}")
        time.sleep(0.05)
        print(f"  总耗时: {time.time() - total_start:.4f}s")

        #最后只参考一下lenet1_x_piliang.py的时间就行
        #下面这个数字得注释掉
        # print(f"识别结果: 数字 {prediction}")
        time.sleep(1)