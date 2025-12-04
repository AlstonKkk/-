import torch
import numpy as np
import os

# 创建保存目录
save_dir = "model_parameters_3d"
os.makedirs(save_dir, exist_ok=True)

# 加载模型参数
state_dict = torch.load("Lenet.pth", map_location='cpu')

# 参数键名映射
param_mapping = {
    'model.0.weight': 'conv1.weight',
    'model.0.bias': 'conv1.bias',
    'model.3.weight': 'conv2.weight',
    'model.3.bias': 'conv2.bias',
    'model.6.weight': 'conv3.weight',
    'model.6.bias': 'conv3.bias',
    'model.9.weight': 'fc1.weight',
    'model.9.bias': 'fc1.bias',
    'model.11.weight': 'fc2.weight',
    'model.11.bias': 'fc2.bias'
}


def save_conv_3d(param_tensor, save_path):
    """保存卷积层权重为三维数组格式 (out_channels, in_channels, kernel_size)"""
    param_np = param_tensor.cpu().numpy()
    with open(save_path, 'w') as f:
        # 遍历每个输出通道
        for out_c in range(param_np.shape[0]):
            f.write(f"=== Output Channel {out_c} ===\n")
            # 遍历每个输入通道
            for in_c in range(param_np.shape[1]):
                f.write(f"Input Channel {in_c}:\n")
                kernel = param_np[out_c, in_c]
                # 写入卷积核矩阵
                np.savetxt(f, kernel, fmt='%12.8f', delimiter=' ', header='', comments='')
                f.write("\n")  # 通道间空行分隔
            f.write("\n")  # 输出通道间分隔


def save_fc_flatten(param_tensor, save_path):
    """保存全连接层参数为一维展平格式"""
    param_np = param_tensor.cpu().numpy().flatten()
    np.savetxt(save_path, param_np, fmt='%12.8f', delimiter='\n')


# 遍历所有参数
for seq_key, save_name in param_mapping.items():
    param = state_dict[seq_key]
    save_path = os.path.join(save_dir, f"{save_name}.dat")

    # 按类型保存
    if 'conv' in save_name and 'weight' in save_name:
        save_conv_3d(param, save_path)
    else:
        save_fc_flatten(param, save_path)

print(f"参数已保存至 {save_dir}，卷积层权重为三维结构！")
