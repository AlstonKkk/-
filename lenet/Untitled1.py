# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

def generate_random_matrices(size=4096):

    matrix_a = np.random.rand(size, size).astype(np.float32)
    matrix_b = np.random.rand(size, size).astype(np.float32)
    return matrix_a, matrix_b

def multiply_matrices_numpy(matrix_a, matrix_b):

    return np.dot(matrix_a, matrix_b)

def multiply_matrices_python(matrix_a, matrix_b):

    rows, cols = matrix_a.shape[0], matrix_b.shape[1]
    result = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            for k in range(matrix_a.shape[1]):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
    
    return result

def measure_time(func, *args, **kwargs):

    start_time = time.time()  # 兼容Python 2的time.perf_counter
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def run_experiment(matrix_size=256, num_runs=5):

    numpy_times = []
    python_times = []
    
    for i in range(num_runs):
        print("运行实验 %d/%d..." % (i+1, num_runs))
        matrix_a, matrix_b = generate_random_matrices(matrix_size)
        
        # 测量NumPy方法
        numpy_time, _ = measure_time(multiply_matrices_numpy, matrix_a, matrix_b)
        numpy_times.append(numpy_time)
        
        # 仅在矩阵较小时运行纯Python方法，避免耗时过长
        if matrix_size <= 128:
            python_time, _ = measure_time(multiply_matrices_python, matrix_a, matrix_b)
            python_times.append(python_time)
    
    # 计算平均时间
    avg_numpy_time = sum(numpy_times) / num_runs
    print("\nNumPy 矩阵乘法平均时间 (%dx%d): %.6f 秒" % (matrix_size, matrix_size, avg_numpy_time))
    
    if python_times:
        avg_python_time = sum(python_times) / num_runs
        print("纯Python 矩阵乘法平均时间 (%dx%d): %.6f 秒" % (matrix_size, matrix_size, avg_python_time))
        print("NumPy 加速比: %.2fx" % (avg_python_time / avg_numpy_time))
    

if __name__ == "__main__":
    # 可以修改矩阵大小和运行次数来测试不同场景
    MATRIX_SIZE = 4096
    NUM_RUNS = 5
    run_experiment(MATRIX_SIZE, NUM_RUNS)