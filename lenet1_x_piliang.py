# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
#
# matplotlib.use('TkAgg')  # 强制使用Tkinter后端
# import time
# import os
#
#
# # -------------------- 1. 模型定义 --------------------
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # conv1
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # conv2
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # conv3
#             nn.ReLU(),
#
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128),  # fc1
#             nn.ReLU(),
#             nn.Linear(128, 10),  # fc2
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# # -------------------- 2. 加载训练好的模型 --------------------
# def load_trained_model(model_path):
#     model = LeNet()
#     checkpoint = torch.load(model_path, map_location='cpu')
#
#     if isinstance(checkpoint, dict) and 'model' in checkpoint:
#         state_dict = checkpoint['model']
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model
#
#
# # -------------------- 3. 图像预处理 --------------------
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('L')
#     img = img.resize((28, 28))
#     img_array = np.array(img).astype(np.float32) / 255.0
#     tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
#     tensor = 1.0 - tensor  # 反色处理
#     return tensor
#
#
# # -------------------- 4. 执行推理 --------------------
# def predict_digit(model, image_tensor):
#     with torch.no_grad():
#         outputs = model(image_tensor)
#     prob, pred = torch.max(outputs, 1)
#     return pred.item(), prob.item()
#
#
# # -------------------- 5. 可视化结果 --------------------
# def visualize_prediction(image_tensor, prediction, confidence):
#     plt.figure(figsize=(6, 3))
#
#     plt.subplot(1, 2, 1)
#     img = image_tensor[0, 0].numpy()
#     plt.imshow(img, cmap='gray')
#     plt.title("Input Image")
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.bar(range(10), [0] * 10, color='lightblue')
#     plt.bar(prediction, confidence, color='red')
#     plt.xticks(range(10))
#     plt.ylim(0, 1)
#     plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
#
#     plt.tight_layout()
#     plt.show()
#
#
# # -------------------- 主程序 --------------------
# if __name__ == "__main__":
#     start_time = time.time()
#
#     # 配置参数
#     MODEL_PATH = '/home/orangepi/daima/lenet.pth'
#     IMAGE_DIR = '/home/orangepi/daima/picture_for_test'  # 替换为你的图片文件夹路径
#
#     # 支持的图片格式
#     supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
#
#     # 加载模型
#     model = load_trained_model(MODEL_PATH)
#
#     # 获取图片文件列表
#     image_files = [f for f in os.listdir(IMAGE_DIR)
#                    if f.lower().endswith(supported_formats)]
#
#     print(f"发现 {len(image_files)} 张待处理图片")
#
#     total_process_time = 0
#     processed_count = 0
#
#     # 遍历处理每张图片
#     for img_file in image_files:
#         file_start_time = time.time()
#         img_path = os.path.join(IMAGE_DIR, img_file)
#
#         try:
#             print(f"\n正在处理: {img_file}")
#
#             # 预处理
#             img_tensor = preprocess_image(img_path)
#
#             # 推理预测
#             pred, conf = predict_digit(model, img_tensor)
#
#             # 记录时间
#             process_time = time.time() - file_start_time
#             total_process_time += process_time
#             processed_count += 1
#
#             # 显示结果
#             print(f"识别结果: 数字 {pred}, 置信度 {conf:.2%}")
#             print(f"本张处理耗时: {process_time:.4f} 秒")
#
#             # 可视化
#             visualize_prediction(img_tensor, pred, conf)
#
#         except Exception as e:
#             print(f"处理 {img_file} 时出错: {str(e)}")
#             continue
#
#     # 最终统计
#     end_time = time.time()
#     print("\n" + "=" * 50)
#     print(f"成功处理 {processed_count}/{len(image_files)} 张图片")
#     print(f"单张平均耗时: {total_process_time / processed_count:.4f} 秒"
#           if processed_count > 0 else "无成功处理图片")
#     print(f"总耗时: {end_time - start_time:.4f} 秒")
#     print("=" * 50)
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 强制使用Tkinter后端
import time
import os

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
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    tensor = 1.0 - tensor  # 反色处理
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

    plt.subplot(1, 2, 1)
    img = image_tensor[0, 0].numpy()
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), [0] * 10, color='lightblue')
    plt.bar(prediction, confidence, color='red')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")

    plt.tight_layout()
    plt.draw()
    plt.show(block=False)  # 改为非阻塞持续显示
    plt.pause(0.1)  # 显示0.1秒


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    plt.ion()  # 启用交互模式
    start_time = time.time()

    # 配置参数
    MODEL_PATH = '/home/orangepi/daima/lenet.pth'
    IMAGE_DIR = '/home/orangepi/daima/picture_for_test'

    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 加载模型
    model = load_trained_model(MODEL_PATH)

    # 获取图片文件列表
    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(supported_formats)]

    print(f"发现 {len(image_files)} 张待处理图片")

    total_process_time = 0
    processed_count = 0

    # 遍历处理每张图片
    for img_file in image_files:
        file_start_time = time.time()
        img_path = os.path.join(IMAGE_DIR, img_file)

        try:
            print(f"\n正在处理: {img_file}")

            # 预处理
            img_tensor = preprocess_image(img_path)

            # 推理预测
            pred, conf = predict_digit(model, img_tensor)

            # 记录时间
            process_time = time.time() - file_start_time
            total_process_time += process_time
            processed_count += 1

            # 显示结果
            print(f"识别结果: 数字 {pred}, 置信度 {conf:.2%}")
            print(f"本张处理耗时: {process_time:.4f} 秒")

            # 可视化
            visualize_prediction(img_tensor, pred, conf)

        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")
            continue

    # 最终统计
    end_time = time.time()
    print("\n" + "=" * 50)
    print(f"成功处理 {processed_count}/{len(image_files)} 张图片")
    print(f"单张平均耗时: {total_process_time / processed_count:.4f} 秒"
          if processed_count > 0 else "无成功处理图片")
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    print("=" * 50)
    #plt.ioff()  # 关闭交互模式
    plt.show(block=True)  # ✅ 最后阻塞显示所有窗口