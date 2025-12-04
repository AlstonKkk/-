import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置
import matplotlib.pyplot as plt
import time


# -------------------- 1. 模型定义 --------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # conv1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # conv2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # conv3
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # fc1
            nn.ReLU(),
            nn.Linear(128, 10),  # fc2
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


# -------------------- 2. 加载训练好的模型 --------------------
def load_trained_model(model_path):
    # 初始化模型
    model = LeNet()

    # 加载state_dict
    checkpoint = torch.load(model_path, map_location='cpu')

    # 处理可能的检查点格式（如果保存的是整个模型而不是state_dict）
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 加载参数
    model.load_state_dict(state_dict)
    model.eval()  # 设置为评估模式
    return model


# -------------------- 3. 图像预处理 --------------------
def preprocess_image(image_path):
    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')  # 转为灰度图

    # 调整尺寸为28x28
    img = img.resize((28, 28))

    # 转换为numpy数组并归一化
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0  # [0,255] → [0,1]

    # 添加batch和channel维度 → [1, 1, 28, 28]
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    # 反色处理（假设训练时是白底黑字）
    tensor = 1.0 - tensor  # 如果输入是黑底白字则不需要这步

    return tensor


# -------------------- 4. 执行推理 --------------------
def predict_digit(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    prob, pred = torch.max(outputs, 1)
    return pred.item(), prob.item()


# -------------------- 5. 可视化结果 --------------------
def visualize_prediction(image_tensor, prediction, confidence):
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


    plt.savefig('/home/orangepi/daima/fanhui_all/prediction_result.png')
    plt.close()


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    start_time = time.time()
    # 配置参数
    #香橙派上的lenet.py的代码是这个注释的部分
    MODEL_PATH = '/home/orangepi/daima/lenet.pth'
    IMAGE_PATH = '/home/orangepi/daima/8.png'  # 替换为你的测试图片路径

    #MODEL_PATH = "lenet.pth"
    #IMAGE_PATH = "8.png"
    # 加载模型
    model = load_trained_model(MODEL_PATH)

    # 预处理图像
    input_tensor = preprocess_image(IMAGE_PATH)

    # 执行预测
    predicted_digit, confidence = predict_digit(model, input_tensor)

    # 显示结果
    print(f"识别结果: 数字 {predicted_digit}, 置信度 {confidence:.2%}")
    visualize_prediction(input_tensor, predicted_digit, confidence)
    end_time = time.time()  # 记录程序结束时间
    print(f"整个程序运行完成，总耗时 {end_time - start_time:.4f} 秒")