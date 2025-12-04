import torch
import torch.nn as nn
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

start_time = time.time()
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
    checkpoint = torch.load(model_path, map_location='cpu',weights_only=True)

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
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0  # 归一化到[0,1]

    # 添加batch和channel维度 → [1, 1, 28, 28]
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    # 反色处理（假设训练时是白底黑字）
    #tensor = 1.0 - tensor  # 如果输入是黑底白字则不需要这步

    return tensor


# -------------------- 4. 执行推理 --------------------
def predict_digit(model, image_tensor):
    """返回所有类别概率"""
    with torch.no_grad():
        outputs = model(image_tensor)
    probabilities = outputs.squeeze().numpy()
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities


# -------------------- 5. 可视化结果 --------------------
def visualize_prediction(image_tensor, probabilities, predicted_class):
    """增强版可视化函数，显示所有类别置信度

    Args:
        image_tensor: 输入图像Tensor (shape: [1,1,28,28])
        probabilities: 所有类别的概率分布 (shape: [10])
        predicted_class: 预测类别 (int)
    """
    plt.figure(figsize=(12, 5))

    # ===================== 显示输入图像 =====================
    plt.subplot(1, 2, 1)
    img = image_tensor[0, 0].numpy()
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title("Input Image\n(28x28 Grayscale)", fontsize=12, pad=10)
    plt.axis('off')

    # ===================== 显示置信度分布 =====================
    plt.subplot(1, 2, 2)

    # 生成颜色列表（预测类别用红色高亮）
    colors = ['skyblue'] * 10
    colors[predicted_class] = 'salmon'

    # 绘制柱状图
    bars = plt.bar(range(10), probabilities * 100,
                   color=colors, edgecolor='black', alpha=0.8)

    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        text_color = 'red' if i == predicted_class else 'black'
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f"{height:.1f}%",
                 ha='center', va='bottom',
                 color=text_color,
                 fontsize=10,
                 fontweight='bold' if i == predicted_class else 'normal')

    # 图表装饰
    plt.xticks(range(10), [str(i) for i in range(10)], fontsize=11)
    plt.yticks(np.arange(0, 101, 20), fontsize=10)
    plt.ylim(0, 100)
    plt.xlabel('Digit Class', fontsize=12, labelpad=8)
    plt.ylabel('Confidence (%)', fontsize=12, labelpad=8)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.title(f"Prediction: Digit {predicted_class}\n"
              f"Confidence: {probabilities[predicted_class] * 100:.2f}%",
              fontsize=14, pad=15)

    plt.tight_layout()
    plt.show()


# -------------------- 主程序 --------------------
if __name__ == "__main__":

    # 配置参数
    MODEL_PATH = "lenet.pth"
    IMAGE_PATH = "picture_for_test/0.png"  # 替换为你的测试图片路径

    # 加载模型
    model = load_trained_model(MODEL_PATH)

    # 预处理图像
    input_tensor = preprocess_image(IMAGE_PATH)

    # 执行预测
    predicted_digit, all_probs = predict_digit(model, input_tensor)


    # 显示结果
    # 显示结果（传入完整概率数组）
    print(f"识别结果: 数字 {predicted_digit}, 置信度 {all_probs[predicted_digit]:.2%}")
    visualize_prediction(input_tensor, all_probs, predicted_digit)

    # 打印详细概率分布
    print("\n详细概率分布:")
    for i, prob in enumerate(all_probs):
        print(f"数字 {i}: {prob * 100:.2f}%", end=' | ')
        if (i + 1) % 5 == 0:  # 每行显示5个结果
            print()
    end_time = time.time()  # 获取结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"代码运行时间：{elapsed_time} 秒")