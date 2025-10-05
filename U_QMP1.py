# main_uqmp_exact.py

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT

# 导入您已验证的 u_comp 函数
from U_comp import u_comp as u_comp_block

def phase_addition_stage2(circuit: QuantumCircuit, src_reg, dst_like, n: int, sign: int = +1, extra_control=None):
    """
    用于第二阶段的相位加/减法。
    - 支持一个额外的控制位 (extra_control)，用于实现三重控制门。
    """
    assert sign in (+1, -1)
    num_dst = len(dst_like)
    for i in range(n):
        for j in range(i, num_dst): # 目标可以是n或n+1位
            if (j - i) >= n: continue # 确保src_reg索引不出界
            
            lam = (np.pi / (2 ** (j - i))) * sign
            
            if extra_control is None:
                circuit.cp(lam, src_reg[i], dst_like[j])
            else:
                # 使用多控相位门实现: C(extra_control) - C(src_reg[i]) - Phase(dst_like[j])
                circuit.mcp(lam, [extra_control, src_reg[i]], dst_like[j])


def create_uqmp_circuit_exact(qc, qr_a, qr_b, qr_c, qr_d, anc_ab, anc_cd, anc_swap):
    """
    严格按照您的描述构建U_QMP电路。
    """
    n = len(qr_a)
    print("正在构建 U_QMP 电路，并使用您描述的精确第二阶段...")

    # === 阶段 1: 并行比较 ===
    qc.barrier(label="Stage 1 Start")
    print("阶段 1: 并行比较 (a,b) 和 (c,d)。")
    # ab 比较器: b <- max(a,b), anc_ab 被设置
    u_comp_block(qc, qr_a, qr_b, anc_ab)
    # cd 比较器: d <- max(c,d), anc_cd 被设置
    u_comp_block(qc, qr_c, qr_d, anc_cd)
    qc.barrier(label="Stage 1 End")

    # === 阶段 2: 您描述的自定义算术单元 ===
    print("阶段 2: 执行自定义相位算术单元...")
    
    # 2.1: 对 d^n 和 anc_swap 进行 (n+1) 比特 QFT
    qft_reg_n_plus_1 = qr_d[:] + [anc_swap]
    qc.append(QFT(n + 1, name='QFT(n+1)'), qft_reg_n_plus_1)
    
    # 2.2: 受控相位变换 φ_{n+1}
    # 应用 φ_{n+1}(-b)
    print("  - 应用 φ_{n+1}(-b)")
    phase_addition_stage2(qc, qr_b, qft_reg_n_plus_1, n, sign=-1)
    
    # 应用受 anc_ab 控制的 φ_{n+1}(-(a-b))，即 φ_{n+1}(b-a)
    print("  - 应用受 anc_ab 控制的 φ_{n+1}(b-a)")
    phase_addition_stage2(qc, qr_b, qft_reg_n_plus_1, n, sign=+1, extra_control=anc_ab) # 加 b
    phase_addition_stage2(qc, qr_a, qft_reg_n_plus_1, n, sign=-1, extra_control=anc_ab) # 减 a
    
    # 2.3: (n+1) 比特 IQFT
    qc.append(QFT(n + 1, name='IQFT(n+1)').inverse(), qft_reg_n_plus_1)
    
    qc.barrier()
    
    # 2.4: d^n 单独进 QFT，并进行 φ_n 相位变换
    print("  - 对 d^n 应用 QFT(n)")
    qc.append(QFT(n, name='QFT(n)'), qr_d)
    
    # 应用 φ_n(b)
    print("  - 应用 φ_n(b)")
    phase_addition_stage2(qc, qr_b, qr_d, n, sign=+1)
    
    # 应用受 anc_ab 控制的 φ_n(a-b)
    print("  - 应用受 anc_ab 控制的 φ_n(a-b)")
    phase_addition_stage2(qc, qr_a, qr_d, n, sign=+1, extra_control=anc_ab) # 加 a
    phase_addition_stage2(qc, qr_b, qr_d, n, sign=-1, extra_control=anc_ab) # 减 b
    
    qc.barrier()
    
    # 2.5: 受 anc_swap 控制的 b 与 d 的交换门
    print("  - 执行受 anc_swap 控制的 C-SWAP(b,d)")
    for i in range(n):
        qc.cswap(anc_swap, qr_b[i], qr_d[i])
        
    qc.barrier(label="U_QMP End")
    print("U_QMP 电路构建完成。")


# --- 主执行脚本 ---
if __name__ == '__main__':
    N_QUBITS = 3
    
    # 定义所有需要的寄存器
    qr_a, qr_b, qr_c, qr_d = (QuantumRegister(N_QUBITS, name=n) for n in ['a', 'b', 'c', 'd'])
    anc_ab, anc_cd, anc_swap = (QuantumRegister(1, name=n) for n in ['anc_ab', 'anc_cd', 'anc_swap'])
    cr_d = ClassicalRegister(N_QUBITS, name='res_d')
    
    qc = QuantumCircuit(qr_a, qr_b, qr_c, qr_d, anc_ab, anc_cd, anc_swap, cr_d)

    # 初始化输入值
    val_a, val_b, val_c, val_d = 5, 2, 7, 1
    qc.initialize(val_a, qr_a)
    qc.initialize(val_b, qr_b)
    qc.initialize(val_c, qr_c)
    qc.initialize(val_d, qr_d)

    # 构建U_QMP电路
    create_uqmp_circuit_exact(qc, qr_a, qr_b, qr_c, qr_d, anc_ab[0], anc_cd[0], anc_swap[0])

    # 测量寄存器 d 作为最终输出
    qc.measure(qr_d, cr_d)

    # 绘制电路图
    print("\n正在绘制精确的 U_QMP 电路图...")
    try:
        qc.draw('mpl', style='iqx', fold=-1).savefig('U_QMP_exact_diagram.png')
        print("电路图已保存为 U_QMP_exact_diagram.png")
    except Exception as e:
        print(f"绘制电路图失败 (可能需要安装 'matplotlib' 和 'pylatexenc'): {e}")

    # 模拟电路
    print("\n正在模拟电路...")
    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=4096).result()
    counts = result.get_counts()

    print("\n模拟结果 (Counts):")
    print(counts)

    # 结果分析
    if not counts:
        print("\n--- 分析 ---")
        print("电路没有测量结果，请检查实现。")
    else:
        # Qiskit 结果是 MSB 在左，LSB 在右，所以直接转换
        most_likely_binary_qiskit = max(counts, key=counts.get)
        most_likely_decimal = int(most_likely_binary_qiskit, 2)
        
        print(f"\n--- 分析 ---")
        print(f"输入值: a={val_a}, b={val_b}, c={val_c}, d={val_d}")
        # 您的 U_comp 使得 b=max(a,b), d=max(c,d)
        max_ab = max(val_a, val_b)
        max_cd = max(val_c, val_d)
        print(f"阶段1后, b 寄存器状态应为 |{max_ab}⟩, d 寄存器状态应为 |{max_cd}⟩")
        print("阶段2对这些中间值执行了您指定的复杂算术运算。")
        print(f"寄存器 'd' 的最频繁测量结果: {most_likely_decimal} (二进制: {most_likely_binary_qiskit})")