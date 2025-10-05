#!/usr/bin/env python3
"""
测试U_comp模块是否正确工作
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from U_comp import u_comp

def test_u_comp():
    """测试U_comp比较器"""
    print("=== 测试U_comp比较器 ===")
    
    # 测试用例1: a=5, b=2 (期望: max=5)
    print("\n测试用例1: a=5, b=2")
    n = 3
    a = QuantumRegister(n, 'a')
    b = QuantumRegister(n, 'b')
    anc = QuantumRegister(1, 'anc')
    cr = ClassicalRegister(n * 2, 'result')
    
    qc = QuantumCircuit(a, b, anc, cr)
    
    # 初始化: a=5(101), b=2(010)
    qc.x(a[0])  # bit 0
    qc.x(a[2])  # bit 2
    qc.x(b[1])  # bit 1
    
    qc.barrier()
    
    # 应用U_comp
    u_comp(qc, a, b, anc[0])
    
    qc.barrier()
    
    # 测量结果
    qc.measure(a, cr[0:n])
    qc.measure(b, cr[n:2*n])
    
    # 模拟
    simulator = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=1024).result()
    counts = result.get_counts()
    
    print("模拟结果:", counts)
    
    # 分析结果
    most_likely = max(counts, key=counts.get)
    a_result = int(most_likely[:n], 2)
    b_result = int(most_likely[n:], 2)
    
    print(f"a的结果: {a_result}, b的结果: {b_result}")
    print(f"期望: max(5,2)=5, 实际: max({a_result},{b_result})={max(a_result, b_result)}")
    
    if max(a_result, b_result) == 5:
        print("✅ U_comp测试通过!")
        return True
    else:
        print("❌ U_comp测试失败!")
        return False

if __name__ == "__main__":
    test_u_comp()
