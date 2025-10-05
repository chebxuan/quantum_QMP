def create_zeroing_network(n_pos_bits: int, n_gray_bits: int):
    """
    创建一个电路，用于将LL子带的像素灰度值置零。
    假设一级变换后，LL子带由地址最高位 y_top=0 和 x_top=0 定义。

    参数:
        n_pos_bits (int): 总的位置比特数 (例如 16 for 256x256)
        n_gray_bits (int): 灰度值的比特数 (例如 8 for 256-level grayscale)

    返回:
        QuantumCircuit: 置零电路.
    """
    # 位置寄存器
    pos_reg = QuantumRegister(n_pos_bits, name='pos')
    # 颜色寄存器
    color_reg = QuantumRegister(n_gray_bits, name='color')
    
    circuit = QuantumCircuit(pos_reg, color_reg)

    # LL子带由最高位地址 y_top=0 和 x_top=0 决定
    # Qiskit的位置寄存器顺序是 |y_top ... y0, x_top ... x0⟩
    # 所以控制位是 pos_reg[n_pos_bits-1] (y_top) 和 pos_reg[n_pos_bits//2 - 1] (x_top)
    control_y = pos_reg[n_pos_bits - 1]
    control_x = pos_reg[n_pos_bits // 2 - 1]

    # 我们需要 "00-控制"，所以在应用多控门前后用X门包裹控制位
    circuit.x(control_y)
    circuit.x(control_x)

    # 对每一个颜色比特应用一个三控门 (MCX)
    from qiskit.circuit.library import MCXGate

    for i in range(n_gray_bits):
        # 创建一个三控门，控制位是 [y_top, x_top, color_bit_i]
        # 我们需要一个 "001-controlled" 门
        # Qiskit的MCXGate可以方便地设置控制状态
        mcx = MCXGate(num_ctrl_qubits=3, ctrl_state="001")
        circuit.append(mcx, [control_y, control_x, color_reg[i], color_reg[i]])

    # 恢复控制位
    circuit.x(control_y)
    circuit.x(control_x)
    
    return circuit

# --- 示例：测试置零网络 ---
# zeroing_circuit = create_zeroing_network(n_pos_bits=16, n_gray_bits=8)
# print("\n置零网络电路图 (只显示一个颜色比特):")
# zeroing_circuit.decompose().draw('mpl') # decompose()可以看到内部的X门