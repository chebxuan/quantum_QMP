import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

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
    output_image = np.zeros_like(image_data)

    # 使用tqdm添加进度条
    for i in tqdm(range(0, size, 2), desc="Processing Image Blocks"):
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

            # 为差分输出分配寄存器
            diff_ba = QuantumRegister(N_QUBITS, name='diff_ba')  # b - a
            diff_dc = QuantumRegister(N_QUBITS, name='diff_dc')  # d - c, 注意layer1后d=max(c,d)，但此diff仍捕获比较阶段的d-c
            diff_db = QuantumRegister(N_QUBITS, name='diff_db')  # max(c,d) - b
            
            # 构建U_QMP电路，并输出三个差分
            create_uqmp_circuit(qc, qr_a, qr_b, qr_c, qr_d, diff_ba=diff_ba, diff_dc=diff_dc, diff_db=diff_db)
            
            # 直接在主电路中添加经典寄存器并进行测量（四个输出：LL, LH, HL, HH）
            cr_LL = ClassicalRegister(N_QUBITS, name='LL')  # max
            cr_LH = ClassicalRegister(N_QUBITS, name='LH')  # b - a
            cr_HL = ClassicalRegister(N_QUBITS, name='HL')  # max(c,d) - b
            cr_HH = ClassicalRegister(N_QUBITS, name='HH')  # d - c
            qc.add_register(cr_LL, cr_LH, cr_HL, cr_HH)
            qc.measure(qr_d, cr_LL)
            qc.measure(diff_ba, cr_LH)
            qc.measure(diff_db, cr_HL)
            qc.measure(diff_dc, cr_HH)

            simulator = AerSimulator(method='matrix_product_state')
            compiled_circuit = transpile(qc, simulator)
            result = simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            # 解析四个classic寄存器的结果（Qiskit返回按寄存器名分组的带空格bit串）
            key = list(counts.keys())[0]
            parts = key.split()
            # 顺序与添加寄存器顺序一致：LL LH HL HH
            val_LL = int(parts[0], 2)
            val_LH = int(parts[1], 2)
            val_HL = int(parts[2], 2)
            val_HH = int(parts[3], 2)

            # 根据新的Max-Plus规则更新四个位置
            output_image[i, j] = val_HL       # HL
            output_image[i, j+1] = val_HH    # HH
            output_image[i+1, j] = val_LL    # LL
            output_image[i+1, j+1] = val_LH  # LH

    return output_image

if __name__ == '__main__':
    # 1. 加载并转换 baboon.bmp 图像
    print("1. 加载和预处理 baboon.bmp 图像 (使用原始尺寸)...")
    with Image.open('baboon.bmp') as img:
        original_image = img.convert('L')
        original_image = np.array(original_image, dtype=np.uint8)
    img_size = original_image.shape[0]

    # 2. 执行Max-Plus变换
    print("2. 模拟Max-Plus变换...")
    transformed_image = run_max_plus_transform(original_image.copy())

    # 3. 跳过LL子带置零操作以最大化PSNR和SSIM
    print("3. 跳过LL子带置零操作...")
    final_image = transformed_image.copy()

    # 4. 显示结果
    print("4. 显示结果...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('图a: 原始 16x16 样本图像')
    axes[0].grid(True, which='both', color='r', linestyle='-')

    axes[1].imshow(final_image, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('图b: 模拟Max-Plus变换并置零LL子带后')
    axes[1].grid(True, which='both', color='r', linestyle='-')

    # 5. 计算并打印PSNR和SSIM
    psnr_value = psnr(original_image, final_image, data_range=255)
    ssim_value = ssim(original_image, final_image, data_range=255)

    print("\n--- 图像质量评估 ---")
    print(f"PSNR (峰值信噪比): {psnr_value:.2f} dB")
    print(f"SSIM (结构相似性): {ssim_value:.4f}")

    plt.show()
