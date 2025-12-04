# 文件名：orangepi_client.py
import socket
import numpy as np
import pickle
import struct
from PIL import Image
import time

def preprocess_image(image_path, invert=True):
    """（原有预处理函数保持不变）"""
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

def send_array(array, host='192.168.0.111', port=5000, save_path='received_image.png'):
    """发送数组并接收服务端返回的图片"""
    send_start = time.time()
    data = pickle.dumps(array)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        # 发送数组数据
        header = struct.pack('!I', len(data))
        s.sendall(header)
        s.sendall(data)
        send_time = time.time() - send_start
        print(f"数组发送耗时: {send_time:.4f}s")

        # 接收图片数据
        # 先接收4字节的图片数据长度头
        jieshou_start = time.time()
        header_img = s.recv(4)
        if not header_img:
            raise ConnectionError("图片头接收失败")
        img_size = struct.unpack('!I', header_img)[0]

        received_img = b''
        while len(received_img) < img_size:
            remaining = img_size - len(received_img)
            packet = s.recv(4096 if remaining > 4096 else remaining)
            if not packet:
                raise ConnectionError("图片数据接收中断")
            received_img += packet

        # 保存图片
        with open(save_path, 'wb') as f:
            f.write(received_img)
        print(f"图片已保存至: {save_path}")
        jieshou_time = time.time() - jieshou_start
        print(f"orangepi发送时间为:{send_time}")
        print(f"orangepi接收时间为:{jieshou_time}")
        print(f"orangepi总时间为:{send_time+jieshou_time}")

if __name__ == "__main__":
    IMAGE_PATH = "8.png"
    contiguous_array = preprocess_image(IMAGE_PATH)
    send_array(contiguous_array, save_path="from_pynq.png")
