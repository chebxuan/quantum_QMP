import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT


def _phase_addition(circuit: QuantumCircuit, src_reg, dst_like, n: int, sign: int = +1):
    assert sign in (+1, -1)
    for i in range(n):
        for j in range(i, n):  # 只处理寄存器位，不包括ancilla
            lam = (np.pi / (2 ** (j - i))) * (1 if sign > 0 else -1)
            circuit.cp(lam, src_reg[i], dst_like[j])
        
        # 单独处理ancilla位
        if len(dst_like) > n:  # 如果有ancilla位
            lam = (np.pi / (2 ** (n - i))) * (1 if sign > 0 else -1)
            circuit.cp(lam, src_reg[i], dst_like[n])


def u_comp(circuit: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister, anc_ab):
    """
    Comparator U_comp consistent with circuit_top.tex encapsulation:
      - Inputs: a^n, b^n, anc_ab (single qubit)
      - Compute sign(b - a) with QFT/phi/IQFT on [b, anc_ab] (anc_ab is the (n+1)-th qubit)
      - Restore b (uncompute) to keep values intact
      - If anc_ab == 1 (i.e., b - a < 0 -> a > b), swap a and b so that b outputs max(a,b)
    """
    n = len(a_reg)
    assert len(b_reg) == n, "a_reg and b_reg must have same length"

    b_extended = [*b_reg, anc_ab]

    # QFT on b || anc_ab
    circuit.append(QFT(n + 1, name='QFT'), b_extended)
    # φ_{n+1}(-a) on targets b_extended
    _phase_addition(circuit, a_reg, b_extended, n, sign=-1)
    # IQFT
    circuit.append(QFT(n + 1, name='IQFT', inverse=True), b_extended)

    # At this point anc_ab is the sign bit of (b - a) in two's complement
    # Restore b to its original value
    circuit.append(QFT(n + 1, name='QFT'), b_extended)
    _phase_addition(circuit, a_reg, b_extended, n, sign=+1)
    circuit.append(QFT(n + 1, name='IQFT', inverse=True), b_extended)

    # If anc_ab == 1, swap a and b => b holds max(a,b)
    for i in range(n):
        circuit.cswap(anc_ab, a_reg[i], b_reg[i])


if __name__ == '__main__':
    # quick self-test
    n = 3
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    anc = QuantumRegister(1, 'anc_ab')
    cr = ClassicalRegister(n * 2, 'res')
    qc = QuantumCircuit(a, b, anc, cr)

    # a=5(101), b=2(010)
    qc.x(a[0]); qc.x(a[2])
    qc.x(b[1])

    u_comp(qc, a, b, anc[0])

    qc.barrier()
    qc.measure(b, cr[0:n])  # expect max(a,b)=5 -> 101
    qc.measure(a, cr[n:2*n])

    backend = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=512).result()
    counts = result.get_counts()
    print(counts)

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram

def controlled_phase_addition(circuit, a_reg, b_like, n):
    """
    对b寄存器执行 controlled-phase addition of a: |b> -> |b+a>
    这是通过在傅里葉空間中进行相位旋轉來實現的。
    """
    # b_like can be a QuantumRegister or a list of qubits (length n or n+1)
    for i in range(n):
        for j in range(i, n):
            lambda_val = np.pi / (2 ** (j - i))
            circuit.cp(lambda_val, a_reg[i], b_like[j])

def inverse_controlled_phase_addition(circuit, a_reg, b_like, n):
    """
    controlled_phase_addition的逆操作: |b> -> |b-a>
    """
    for i in range(n):
        for j in range(i, n):
            lambda_val = -np.pi / (2 ** (j - i))
            circuit.cp(lambda_val, a_reg[i], b_like[j])


def phase_addition(circuit: QuantumCircuit, src_reg, dst_like, n: int, sign: int = +1, extra_control=None):
    """
    Add or subtract src_reg into dst_like via controlled-phase rotations in Fourier space.
    - If extra_control is provided, use multi-controlled phase (mcp) with extra_control.
    - dst_like: list-like sequence of target qubits (length >= n)
    """
    assert sign in (+1, -1)
    for i in range(n):
        for j in range(i, n):
            lam = (np.pi / (2 ** (j - i))) * (1 if sign > 0 else -1)
            if extra_control is None:
                circuit.cp(lam, src_reg[i], dst_like[j])
            else:
                # multi-controlled phase using extra control and src bit
                circuit.mcp(lam, [extra_control, src_reg[i]], dst_like[j])


def u_com(circuit: QuantumCircuit, a_reg: QuantumRegister, b_reg: QuantumRegister, anc_qubit, b_ext_qubit, control_swap_by_ext: bool = False):
    """
    In-place U_com on provided circuit and registers.
    - a_reg: n-qubit register holding value a
    - b_reg: n-qubit register holding value b (will be restored after compare)
    - anc_qubit: 1 ancilla qubit to store comparison result
    - b_ext_qubit: 1 extra qubit to extend b to n+1 for arithmetic

    Behavior (encapsulated as in the LaTeX block):
      1) Compute s = sign(b - a) into anc_qubit (via QFT +/- phase ops + IQFT)
      2) Uncompute to restore b
      3) Controlled by anc_qubit, swap a and b bitwise
    """
    n = len(a_reg)
    assert len(b_reg) == n, "b_reg must have same length as a_reg"

    # 1) comparison: put b_ext (n+1) into Fourier space
    b_extended = [*b_reg, b_ext_qubit]
    circuit.append(QFT(len(b_extended), name='QFT'), b_extended)

    # compute (b - a) in Fourier space
    inverse_controlled_phase_addition(circuit, a_reg, b_extended, n)

    # back to computational basis
    circuit.append(QFT(len(b_extended), name='IQFT', inverse=True), b_extended)

    # Highest bit (b_ext_qubit) works as the sign bit in two's complement arithmetic
    circuit.cx(b_ext_qubit, anc_qubit)
    circuit.barrier()

    # 2) uncompute: restore b
    circuit.append(QFT(len(b_extended), name='QFT'), b_extended)
    controlled_phase_addition(circuit, a_reg, b_extended, n)
    circuit.append(QFT(len(b_extended), name='IQFT', inverse=True), b_extended)
    circuit.barrier()

    # 3) Controlled swap a <-> b
    control_qubit = b_ext_qubit if control_swap_by_ext else anc_qubit
    for i in range(n):
        circuit.cswap(control_qubit, a_reg[i], b_reg[i])

    circuit.barrier()

def create_u_com_circuit(n: int):
    """
    创建 U_com 电路.
    
    参数:
        n (int): 寄存器 a 和 b 的比特数 (精度).

    返回:
        QuantumCircuit: U_com 电路.
    """
    # --- 1. 初始化寄存器 ---
    # a 和 b 是要比较的数
    a_reg = QuantumRegister(n, name='a')
    # b 寄存器需要 n+1 位来执行减法 b-a 并存储符号位
    b_reg = QuantumRegister(n + 1, name='b_ext')
    # c 是比较结果位
    c_reg = QuantumRegister(1, name='c')
    
    circuit = QuantumCircuit(a_reg, b_reg, c_reg)

    # --- 2. 比较模块 (计算 b-a) ---
    # 将 b 变换到傅里叶空间
    circuit.append(QFT(n + 1, name='QFT'), b_reg)
    
    # 在傅里叶空间中执行减法 (b - a)
    # 这里我们只使用a的n位
    a_subset = a_reg[:]
    b_subset_for_sub = b_reg[:n+1] # b的全部n+1位都参与
    inverse_controlled_phase_addition(circuit, a_subset, b_subset_for_sub, n)
    
    # 变换回计算基
    circuit.append(QFT(n + 1, name='IQFT', inverse=True), b_reg)

    # --- 3. 存储比较结果 ---
    # 减法 b-a 的结果存储在 b_reg 中。
    # 如果 b < a, b-a 会是一个负数，其在 n+1 位补码表示下的最高位 b_reg[n] 为 |1⟩.
    # 题目要求 a < b 时 c=1, 所以当 b_reg[n]=0 (b-a>=0) 时, c=1 (需要反转逻辑)
    # 或者，我们直接计算a-b, 这样当a<b时, a-b为负, 最高位为1, 符合要求
    # 为简单起见，我们这里先实现 b-a<0 (即 a>b) 时 c=1
    # 如果您需要 a<b 时 c=1, 只需在CNOT前后对 b_reg[n] 加X门
    circuit.cx(b_reg[n], c_reg[0])
    circuit.barrier()

    # --- 4. 重置模块 (Uncomputation) ---
    # 为了恢复 b 寄存器，我们需要做逆操作，即加上 a
    # 变换到傅里叶空间
    circuit.append(QFT(n + 1, name='QFT'), b_reg)

    # 在傅里叶空间中执行加法 (b-a) + a = b
    controlled_phase_addition(circuit, a_subset, b_subset_for_sub, n)

    # 变换回计算基
    circuit.append(QFT(n + 1, name='IQFT', inverse=True), b_reg)
    circuit.barrier()

    # --- 5. 位置交换模块 ---
    # 如果 c=1 (即 a > b), 则交换 a 和 b
    for i in range(n):
        circuit.cswap(c_reg[0], a_reg[i], b_reg[i])

    return circuit

if __name__ == '__main__':
    # --- 示例：测试 U_com 电路 ---
    n_bits = 3  # 例如，用3个比特比较0-7之间的数
    val_a = 5   # a = 101
    val_b = 2   # b = 010

    # 创建主电路来准备输入态并应用 U_com
    a_reg_main = QuantumRegister(n_bits, name='a')
    b_reg_main = QuantumRegister(n_bits + 1, name='b_ext')
    c_reg_main = QuantumRegister(1, name='c')
    cr = ClassicalRegister(n_bits * 2 + 2, name='result') # 测量 a, b, c

    main_circuit = QuantumCircuit(a_reg_main, b_reg_main, c_reg_main, cr)

    # 准备输入态 |a⟩ = |5⟩ 和 |b⟩ = |2⟩
    # a=5 -> 101
    main_circuit.x(a_reg_main[0])
    main_circuit.x(a_reg_main[2])

    # b=2 -> 010
    main_circuit.x(b_reg_main[1])
    main_circuit.barrier()

    # 应用 U_com 电路
    example_u_com = create_u_com_circuit(n_bits)
    # 注意：Qiskit中寄存器是逆序的，但我们这里构造正确，直接组合
    main_circuit = main_circuit.compose(example_u_com, qubits=[*a_reg_main, *b_reg_main, *c_reg_main])
    main_circuit.barrier()

    # 测量结果
    # 我们只关心 a, b 的前 n 位和 c
    main_circuit.measure(a_reg_main, cr[0:n_bits])
    main_circuit.measure(b_reg_main[0:n_bits], cr[n_bits:n_bits*2])
    main_circuit.measure(c_reg_main, cr[n_bits*2])

    # 使用模拟器运行
    backend = Aer.get_backend('aer_simulator')
    compiled = transpile(main_circuit, backend)
    job = backend.run(compiled, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("输入: a=5, b=2")
    print("电路期望输出: a_out=min(5,2)=2, b_out=max(5,2)=5, c=1 (因为 a>b)")
    print("模拟结果:", counts)
    # 结果应该是 '10100101', 逆序读: c=1, b=010(2), a=101(5)

    # 绘制电路
    print("\nU_com 电路图:")
    example_u_com.draw('mpl')
    # --- 保存电路图 ---
    example_u_com.draw('mpl', filename='U_com_circuit.png')
    main_circuit.draw('mpl', filename='U_com_main.png')
    print("电路图已保存为 U_com_circuit.png 与 U_com_main.png")

