# 文件名：orangepi_client.py
import socket
import numpy as np
import pickle
import struct
from PIL import Image
import time


def preprocess_image(image_path, invert=True):
    """（预处理函数保持不变）"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((28, 28))
            img_array = np.array(img, dtype=np.float32) / 255.0
            if invert:
                img_array = 1.0 - img_array
            hwc_array = img_array.reshape(28, 28, 1)
            contiguous_array = np.ascontiguousarray(hwc_array, dtype=np.float32)
            return contiguous_array
    except Exception as e:
        print(f"预处理失败: {str(e)}")
        raise


def send_data(array, host='192.168.0.111', port=5000):
    """发送数组数据到服务端，返回socket对象和发送耗时"""
    send_start = time.time()
    data = pickle.dumps(array)

    # 创建并连接socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # 发送数据
    header = struct.pack('!I', len(data))
    sock.sendall(header)
    sock.sendall(data)

    send_time = time.time() - send_start
    print(f"数组发送耗时: {send_time:.4f}s")
    return sock, send_time


def receive_image(sock, send_time, save_path='received_image.png'):
    """从服务端接收处理后的图片数据"""
    receive_start = time.time()

    try:
        # 接收图片数据头
        header_img = sock.recv(4)
        if not header_img:
            raise ConnectionError("图片头接收失败")
        img_size = struct.unpack('!I', header_img)[0]

        # 接收图片数据
        received_img = b''
        while len(received_img) < img_size:
            remaining = img_size - len(received_img)
            packet = sock.recv(4096 if remaining > 4096 else remaining)
            if not packet:
                raise ConnectionError("图片数据接收中断")
            received_img += packet

        # 保存图片
        with open(save_path, 'wb') as f:
            f.write(received_img)
        print(f"图片已保存至: {save_path}")

        # 计算时间
        receive_time = time.time() - receive_start
        total_time = send_time + receive_time
        print(f"接收时间: {receive_time:.4f}s")
        print(f"总耗时: {total_time:.4f}s")

    finally:
        sock.close()  # 确保关闭连接


if __name__ == "__main__":
    IMAGE_PATH = "8.png"
    # 预处理图像
    contiguous_array = preprocess_image(IMAGE_PATH)

    # 发送数据并获取保持连接的socket
    sock, send_time = send_data(contiguous_array)

    # 接收处理后的图片
    receive_image(sock, send_time, save_path="from_pynq.png")