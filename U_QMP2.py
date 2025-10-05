# main_uqmp_split.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from U_comp import u_comp          # 您的模块
from qiskit.circuit.library import QFT

N = 3
a,b,c,d = 5,2,7,1

# --- 寄存器 ---
qa = QuantumRegister(N,'a');  qb = QuantumRegister(N,'b')
qc = QuantumRegister(N,'c');  qd = QuantumRegister(N,'d')
anc_ab = QuantumRegister(1,'anc_ab'); anc_cd = QuantumRegister(1,'anc_cd')
anc_s  = QuantumRegister(1,'anc_swap')
# 经典寄存器：先测 b、d，再测最终 d
cb = ClassicalRegister(N,'res_b'); cd = ClassicalRegister(N,'res_d')
cd_final = ClassicalRegister(N,'res_d_final')

qcirc = QuantumCircuit(qa,qb,qc,qd,anc_ab,anc_cd,anc_s,cb,cd,cd_final)

# 初始化
for reg,val in [(qa,a),(qb,b),(qc,c),(qd,d)]:
    for k in range(N):
        if (val>>k)&1: qcirc.x(reg[k])

# ===== 阶段 1 =====
u_comp(qcirc, qa, qb, anc_ab[0])
u_comp(qcirc, qc, qd, anc_cd[0])
qcirc.barrier()

# ---- 第一次测量：看 b、d 是不是 5 和 7 ----
qcirc.measure(qb, cb)
qcirc.measure(qd, cd)
# 为了继续跑阶段 2，再把它们放回量子寄存器（用初始化即可）
qcirc.barrier()

# ===== 阶段 2：您手绘的完整算术单元 =====
# 2.1  (n+1)-QFT on d||anc_s
qcirc.append(QFT(N+1), [*qd, anc_s[0]])
# 2.2  φ_{n+1}(-b)
from U_comp import _phase_addition
_phase_addition(qcirc, qb, [*qd, anc_s[0]], N, sign=-1)
# 2.3  φ_{n+1}(b-a) 受 anc_ab 控制
_phase_addition(qcirc, qb, [*qd, anc_s[0]], N, sign=+1)   # +b
_phase_addition(qcirc, qa, [*qd, anc_s[0]], N, sign=-1)  # -a
# 2.4  IQFT(n+1)
qcirc.append(QFT(N+1, inverse=True), [*qd, anc_s[0]])
qcirc.barrier()

# 2.5  QFT(n) on d
qcirc.append(QFT(N), qd)
# 2.6  φ_n(b)
_phase_addition(qcirc, qb, qd, N, sign=+1)
# 2.7  φ_n(a-b) 受 anc_ab 控制
_phase_addition(qcirc, qa, qd, N, sign=+1)
_phase_addition(qcirc, qb, qd, N, sign=-1)
# 2.8  IQFT(n)
qcirc.append(QFT(N, inverse=True), qd)
qcirc.barrier()

# 2.9  C-SWAP(b,d) 受 anc_s 控制
for i in range(N):
    qcirc.cswap(anc_s[0], qb[i], qd[i])

# ===== 最终测量 =====
qcirc.measure(qd, cd_final)

# 运行
backend = Aer.get_backend('aer_simulator')
counts = backend.run(transpile(qcirc, backend), shots=4096).result().get_counts()

print('分段测量结果 (cb= b 寄存器, cd= d 寄存器, cd_final= 阶段 2 输出):')
print(counts)