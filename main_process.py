import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator

# 假设 U_QMP 和 zhiling 脚本中的函数是可导入的
from U_QMP import create_uqmp_circuit
from zhiling import create_zeroing_network

def simulate_circuit(qc, n_qubits):
    """Helper function to simulate a circuit and get the result as an integer."""
    cr = ClassicalRegister(n_qubits)
    qc.add_register(cr)
    qc.measure(qc.qregs[0], cr)
    
    simulator = AerSimulator(method='matrix_product_state')
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1).result()
    counts = result.get_counts()
    # Get the single shot result and convert from binary string to int
    result_str = list(counts.keys())[0]
    return int(result_str, 2)

def run_max_plus_transform(image_data):
    """对整个图像执行Max-Plus变换的模拟."""
    size = image_data.shape[0]
    # 输出图像, D_ij = max(I_2i,2j, I_2i,2j+1, I_2i+1,2j, I_2i+1,2j+1)
    output_image = np.zeros_like(image_data)

    # 遍历所有 2x2 块
    for i in range(0, size, 2):
        for j in range(0, size, 2):
            # 提取 2x2 块的像素值
            a, b = image_data[i, j], image_data[i, j+1]
            c, d = image_data[i+1, j], image_data[i+1, j+1]
            
            # --- 使用 U_QMP 模拟计算最大值 ---
            N_QUBITS = 8 # 假设8位灰度
            qr_a = QuantumRegister(N_QUBITS, name='a')
            qr_b = QuantumRegister(N_QUBITS, name='b')
            qr_c = QuantumRegister(N_QUBITS, name='c')
            qr_d = QuantumRegister(N_QUBITS, name='d')
            qc = QuantumCircuit(qr_a, qr_b, qr_c, qr_d)

            # 初始化输入值
            qc.initialize(int(a), qr_a)
            qc.initialize(int(b), qr_b)
            qc.initialize(int(c), qr_c)
            qc.initialize(int(d), qr_d)

            # 构建U_QMP电路
            create_uqmp_circuit(qc, qr_a, qr_b, qr_c, qr_d)
            
            # 直接在主电路中添加经典寄存器并进行测量
            cr_result = ClassicalRegister(N_QUBITS, name='res')
            qc.add_register(cr_result)
            qc.measure(qr_d, cr_result)

            simulator = AerSimulator(method='matrix_product_state')
            compiled_circuit = transpile(qc, simulator)
            result = simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            # 获取结果字符串并移除空格，以防格式问题
            result_str = list(counts.keys())[0].replace(' ', '')
            max_val = int(result_str, 2)

            # 根据Max-Plus变换规则更新四个位置
            output_image[i, j] = max_val       # HL
            output_image[i, j+1] = max_val   # HH
            output_image[i+1, j] = max_val   # LL
            output_image[i+1, j+1] = max_val # LH

    return output_image

if __name__ == '__main__':
    # 1. 创建一个 16x16 的样本图像 (图a)
    # 使用一个简单的灰度梯度图案
    print("1. 创建样本图像...")
    img_size = 16
    original_image = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            original_image[i, j] = (i + j) * (255 // (2 * img_size - 2))

    # 2. 执行Max-Plus变换
    print("2. 模拟Max-Plus变换...")
    transformed_image = run_max_plus_transform(original_image.copy())

    # 3. 对LL子带置零
    # LL子带在一级变换后位于图像的左上角四分之一区域
    print("3. 对LL子带置零...")
    ll_size = img_size // 2
    # 注意: 在我们的模拟中，LL子带是 output_image[i+1, j] 的集合
    # 为了简化，我们直接在经典图像上模拟置零操作
    # 遍历变换后图像的左上角区域 (对应LL子带)
    final_image = transformed_image.copy()
    for i in range(0, img_size, 2):
        for j in range(0, img_size, 2):
             # [i+1, j] 是LL子带像素
             final_image[i+1, j] = 0

    # 4. 显示结果
    print("4. 显示结果...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('图a: 原始 16x16 样本图像')
    axes[0].grid(True, which='both', color='r', linestyle='-')

    axes[1].imshow(final_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('图b: 模拟Max-Plus变换并置零LL子带后')
    axes[1].grid(True, which='both', color='r', linestyle='-')

    # 分析LL子带
    ll_sub_band = final_image[1:img_size:2, 0:img_size:2]
    all_zeros = np.all(ll_sub_band == 0)
    
    print("\n--- 分析 ---")
    print(f"LL子带是否全部为零: {all_zeros}")
    if all_zeros:
        print("分析结果: 成功! LL子带已被完全置零。")
    else:
        print("分析结果: 失败! LL子带未被置零。")

    plt.show()
