# 文件名：orangepi_client.py
import socket
import pickle
import struct
import cv2
import numpy as np
import time
from PIL import Image

def preprocess_image(image_path, invert=True):
    try:
        # 1. 读取图像并转换为灰度图，缩放至28x28
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))

        # 2. 转换为numpy数组并归一化到[0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 3. 反色处理（白底黑字 → 黑底白字）
        if invert:
            img_array = 1.0 - img_array

        # 4. 重塑维度并确保内存连续
        hwc_array = img_array.reshape((28, 28, 1))
        buffer = np.ascontiguousarray(hwc_array)  # 保证内存连续性

        return buffer
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None


def send_data(batch_arrays, host='192.168.0.20', port=5000):
    """批量发送数组数据到服务端"""
    send_start = time.time()
    data = pickle.dumps(batch_arrays)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    header = struct.pack('!I', len(data))
    sock.sendall(header)
    sock.sendall(data)

    send_time = time.time() - send_start
    print(f"批量发送耗时: {send_time:.4f}s")
    return sock, send_time
# def receive_image(sock, send_time, save_paths):
#     """批量接收处理后的图片（直接保存字节流）"""
#     receive_start = time.time()
#
#     try:
#         # 接收数据头（确保完整接收4字节）
#         header = b''
#         while len(header) < 4:
#             chunk = sock.recv(4 - len(header))
#             if not chunk:
#                 raise ConnectionError("数据头接收失败")
#             header += chunk
#
#         data_size = struct.unpack('!I', header)[0]
#
#         # 接收完整数据
#         received_data = b''
#         while len(received_data) < data_size:
#             remaining = data_size - len(received_data)
#             packet = sock.recv(4096 if remaining > 4096 else remaining)
#             if not packet:
#                 raise ConnectionError("数据接收中断")
#             received_data += packet
#         receive_time = time.time() - receive_start
#         # 直接保存为图片文件（非pickle反序列化）
#         if len(save_paths) != 1:
#             raise ValueError("当前仅支持单张图片接收")
#
#         with open(save_paths[0], 'wb') as f:
#             f.write(received_data)
#         print(f"图片已保存至: {save_paths[0]}")
#         print(f"IMAGE_SAVED:{save_paths[0]}")
#         # 计算耗时
#
#         total_time = send_time + receive_time
#         print(f"接收耗时: {receive_time:.4f}s")
#         print(f"总耗时: {total_time:.4f}s")
#
#     finally:
#         sock.close()
# def receive_image(sock, send_time, save_paths):
#     """批量接收处理后的图片（直接保存字节流）"""
#     receive_start = time.time()
#
#     try:
#         # 接收数据头
#         header = sock.recv(4)
#         data_size = struct.unpack('!I', header)[0]
#
#         # 使用更大的接收块大小
#         CHUNK_SIZE = 64 * 1024  # 64KB chunks
#         received_data = bytearray()
#
#         while len(received_data) < data_size:
#             remaining = data_size - len(received_data)
#             chunk_size = min(CHUNK_SIZE, remaining)
#             chunk = sock.recv(chunk_size)
#             if not chunk:
#                 raise ConnectionError("数据接收中断")
#             received_data.extend(chunk)
#         receive_time = time.time() - receive_start
#         # 直接保存为图片文件（非pickle反序列化）
#         if len(save_paths) != 1:
#             raise ValueError("当前仅支持单张图片接收")
#
#         with open(save_paths[0], 'wb') as f:
#             f.write(received_data)
#         print(f"图片已保存至: {save_paths[0]}")
#         print(f"IMAGE_SAVED:{save_paths[0]}")
#         # 计算耗时
#
#         total_time = send_time + receive_time
#         print(f"接收耗时: {receive_time:.4f}s")
#         print(f"总耗时: {total_time:.4f}s")
#
#     finally:
#         sock.close()
def receive_image(sock, send_time, save_paths):
    """批量接收处理后的图片（直接保存字节流）"""
    try:
        # 等待开始传输标记
        start_flag = sock.recv(4)
        if struct.unpack('!I', start_flag)[0] != 0xFFFFFFFF:
            raise ConnectionError("未收到正确的开始标记")

        # 开始计时
        receive_start = time.time()

        # 接收数据头
        header = sock.recv(4)
        if not header:
            raise ConnectionError("数据头接收失败")

        data_size = struct.unpack('!I', header)[0]

        # 接收完整数据
        received_data = b''
        while len(received_data) < data_size:
            remaining = data_size - len(received_data)
            packet = sock.recv(4096 if remaining > 4096 else remaining)
            if not packet:
                raise ConnectionError("数据接收中断")
            received_data += packet

        receive_time = time.time() - receive_start

        # 保存图片
        if len(save_paths) != 1:
            raise ValueError("当前仅支持单张图片接收")

        with open(save_paths[0], 'wb') as f:
            f.write(received_data)
        print(f"图片已保存至: {save_paths[0]}")
        print(f"IMAGE_SAVED:{save_paths[0]}")

        # 只显示纯接收时间
        print(f"纯接收耗时: {receive_time:.4f}s")

    finally:
        sock.close()

if __name__ == "__main__":
    # 配置多个输入输出路径
    #IMAGE_PATHS = ["/home/orangepi/daima/8.png", "/home/orangepi/daima/9.png"]  # 修改为实际图片路径
    IMAGE_PATHS = ["/home/orangepi/daima/8.png"]  # 修改为实际图片路径
    SAVE_PATHS = [
        "/home/orangepi/daima/fanhuijiagou/from_pynq_1.png"]
    # SAVE_PATHS = [
    #     "/home/orangepi/daima/fanhuijiagou/from_pynq_1.png",
    #     "/home/orangepi/daima/fanhuijiagou/from_pynq_2.png"
    # ]
    # 批量预处理
    batch_arrays = []
    for path in IMAGE_PATHS:
        try:
            processed_array = preprocess_image(path)
            batch_arrays.append(processed_array)
        except Exception as e:
            print(f"图片 {path} 预处理失败: {e}")
            continue

    if not batch_arrays:
        print("没有可发送的有效图片数据")
        exit()

    # 发送并接收数据
    sock, send_time = send_data(batch_arrays)
    receive_image(sock, send_time, SAVE_PATHS)