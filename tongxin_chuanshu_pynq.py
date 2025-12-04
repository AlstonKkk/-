# 文件名：pynq_server.py
import socket
import numpy as np
import pickle
import struct
import time
from PIL import Image
import io
import os


def start_server(host='192.168.0.20', port=5000, image_path="/home/xilinx/bu_image/image.png"):
    """启动服务端，接收数组后返回本地图片"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        print(f"服务端已启动，等待连接... {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"已连接客户端: {addr}")
                # 接收数组部分保持不变
                jieshou1_start = time.time()
                header = conn.recv(4)
                if not header:
                    continue
                data_size = struct.unpack('!I', header)[0]
                received_data = b''
                while len(received_data) < data_size:
                    packet = conn.recv(data_size - len(received_data))
                    if not packet:
                        break
                    received_data += packet
                array = pickle.loads(received_data)
                print(f"接收数组: {array.shape}")
                jieshou1_time = time.time() - jieshou1_start

                # 改为发送本地图片
                fasong1_start = time.time()
                try:
                    # 读取本地图片文件
                    with open(image_path, 'rb') as f:
                        png_bytes = f.read()

                    # 发送图片数据（保留原有协议）
                    header_img = struct.pack('!I', len(png_bytes))
                    conn.sendall(header_img)
                    conn.sendall(png_bytes)

                except Exception as e:
                    print(f"发送图片失败: {str(e)}")
                    error_msg = f"Error: {str(e)}".encode()
                    header_img = struct.pack('!I', len(error_msg))
                    conn.sendall(header_img)
                    conn.sendall(error_msg)

                fasong1_time = time.time() - fasong1_start
                print(f"已返回图片，大小: {len(png_bytes)}字节")
                print(f"pynq发送时间为:{fasong1_time:.4f}s")
                print(f"pynq接收时间为:{jieshou1_time:.4f}s")
                print(f"pynq总时间为:{(fasong1_time + jieshou1_time):.4f}s")


if __name__ == "__main__":
    # 修改为你的实际图片路径
    start_server(image_path="/home/xilinx/bu_image/image.png")