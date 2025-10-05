# main_uqmp_circuit.py

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from math import pi

# Use the encapsulated comparator consistent with circuit_top.tex
from U_comp import u_comp as u_comp_block

def create_uqmp_circuit(circuit, qr_a, qr_b, qr_c, qr_d, diff_ba: QuantumRegister | None = None, diff_dc: QuantumRegister | None = None, diff_db: QuantumRegister | None = None):
    """
    Builds the U_QMP circuit architecture using the provided U_com module.
    This function finds max(a, b, c, d) and stores it in qr_d.
    """
    print("Building U_QMP circuit with the correct U_com module...")
    
    circuit.barrier()
    
    # Allocate ancillas per comparator and the new short anc for stage-2
    n = len(qr_a)
    reg_anc_ab = QuantumRegister(1, 'anc_ab')
    reg_anc_cd = QuantumRegister(1, 'anc_cd')
    reg_anc_swap = QuantumRegister(1, 'anc_swap')  # pairs with d in QFT and controls SWAP
    circuit.add_register(reg_anc_ab, reg_anc_cd, reg_anc_swap)
    # optional diff output registers
    if diff_ba is not None:
        circuit.add_register(diff_ba)
    if diff_dc is not None:
        circuit.add_register(diff_dc)
    if diff_db is not None:
        circuit.add_register(diff_db)
    anc_ab = reg_anc_ab[0]
    anc_cd = reg_anc_cd[0]
    anc_swap = reg_anc_swap[0]

    # === Layer 1: Parallel Comparison ===
    print("Layer 1: Comparing a,b and c,d in parallel.")
    # ab comparator: makes b <- max(a,b), sets anc_ab, optionally capture (b-a)
    if diff_ba is not None:
        u_comp_block(circuit, qr_a, qr_b, anc_ab, diff_ba)
    else:
        u_comp_block(circuit, qr_a, qr_b, anc_ab)
    # cd comparator: makes d <- max(c,d), sets anc_cd, optionally capture (d-c)
    if diff_dc is not None:
        u_comp_block(circuit, qr_c, qr_d, anc_cd, diff_dc)
    else:
        u_comp_block(circuit, qr_c, qr_d, anc_cd)
    
    circuit.barrier()
    
    # === Layer 2: Final Comparison using QFT Subtractor ===
    print("Layer 2: Comparing max(a,b) and max(c,d) using QFT subtraction.")
    
    # Qubits for the QFT part (d and the ancilla)
    qft_qubits = qr_d[:] + reg_anc_swap[:]  # d register + anc_swap

    # 1. Compute d - b to get the sign bit
    # QFT
    circuit.append(QFT(n + 1, do_swaps=True), qft_qubits)

    # Controlled phase additions for subtraction d - b
    for i in range(n):
        for j in range(i, n):
            angle = -pi / (2**(j - i))
            circuit.cp(angle, qr_b[i], qr_d[j])
        # Phase for the anc_swap (MSB)
        angle_anc = -pi / (2**(n - i))
        circuit.cp(angle_anc, qr_b[i], anc_swap)

    # IQFT
    circuit.append(QFT(n + 1, do_swaps=True).inverse(), qft_qubits)
    # optionally capture (d - b) into diff_db (low n bits)
    if diff_db is not None:
        for i in range(n):
            circuit.cx(qr_d[i], diff_db[i])
    circuit.barrier()

    # 2. Controlled SWAP based on the sign bit (anc_swap)
    # If anc_swap is 1 (d < b), swap b and d to make d the maximum
    for i in range(n):
        circuit.cswap(anc_swap, qr_b[i], qr_d[i])
    circuit.barrier()

    # 3. Corrective Subtraction: If a swap happened, d now holds b's old value.
    # We must compute d-b again to restore the original d, which is now in b.
    # This is equivalent to b - (d-b) mod 2^n, which restores b.
    # The logic is complex; a simpler way is to uncompute the first subtraction
    # conditioned on the ancilla.

    # We apply the inverse of the initial subtraction, controlled by anc_swap.
    # This effectively does b -> b - d if anc_swap=1
    
    # QFT for corrective step
    circuit.append(QFT(n + 1, do_swaps=True), qft_qubits)

    # Controlled phase additions for subtraction d - b (inverse op)
    for i in range(n):
        for j in range(i, n):
            angle = pi / (2**(j - i))
            circuit.cp(angle, qr_b[i], qr_d[j], ctrl_state='0') # Apply if anc_swap is 0
        angle_anc = pi / (2**(n - i))
        circuit.cp(angle_anc, qr_b[i], anc_swap, ctrl_state='0')

    # IQFT
    circuit.append(QFT(n + 1, do_swaps=True).inverse(), qft_qubits)
    circuit.barrier()
    
    # Reset the ancilla for future use (good practice)
    circuit.reset(anc_swap)

    circuit.barrier()
    print("U_QMP circuit built.")


if __name__ == '__main__':
    print("--- Script starting ---")
    # --- 1. Setup ---
    N_QUBITS = 3  # For numbers 0-7

    qr_a = QuantumRegister(N_QUBITS, name='a')
    qr_b = QuantumRegister(N_QUBITS, name='b')
    qr_c = QuantumRegister(N_QUBITS, name='c')
    qr_d = QuantumRegister(N_QUBITS, name='d')
    cr_d = ClassicalRegister(N_QUBITS, name='res_d')
    qc = QuantumCircuit(qr_a, qr_b, qr_c, qr_d, cr_d)
    print("--- 1. Setup complete ---")

    # --- 2. Initialize Inputs ---
    # Test case: a=5, b=2, c=7, d=1
    # Expected final result in register 'd': max(5,2,7,1) = 7 (binary 111)
    val_a, val_b, val_c, val_d = 5, 2, 7, 1
    
    # Initialize via X gates (binary LSB->MSB)
    for idx, val in enumerate([val_a, val_b, val_c, val_d]):
        reg = [qr_a, qr_b, qr_c, qr_d][idx]
        for bit in range(N_QUBITS):
            if (val >> bit) & 1:
                qc.x(reg[bit])
    print("--- 2. Initialization complete ---")

    # --- 3. Build the U_QMP Circuit ---
    create_uqmp_circuit(qc, qr_a, qr_b, qr_c, qr_d)
    print("--- 3. Circuit build complete ---")

    # --- 4. Measure the Result ---
    # The final maximum value is in register 'd' per spec
    qc.measure(qr_d, cr_d)
    print("--- 4. Measurement configured ---")

    # --- 5. Draw the Circuit (skip if matplotlib not available) ---
    print("\nSkipping circuit diagram drawing (requires pylatexenc)...")
    # qc.draw('mpl', style='iqx', fold=-1).savefig('U_QMP_final_circuit.png')
    # print("Circuit diagram saved as U_QMP_final_circuit.png")
    
    # --- 6. Simulate using qiskit_aer ---
    print("\nSimulating the circuit...")
    simulator = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=1024).result()
    counts = result.get_counts()

    print("\nSimulation Results:")
    print(counts)
    
    most_likely_binary = max(counts, key=counts.get)
    most_likely_decimal = int(most_likely_binary, 2)
    
    print(f"\n--- Analysis ---")
    print(f"Inputs: a={val_a}, b={val_b}, c={val_c}, d={val_d}")
    print(f"Expected maximum: {max(val_a, val_b, val_c, val_d)}")
    print(f"Most frequent simulation result: {most_likely_decimal} (binary {most_likely_binary})")
    
    if most_likely_decimal == max(val_a, val_b, val_c, val_d):
        print("\nSUCCESS: The U_QMP circuit correctly calculated the maximum value.")
    else:
        print("\nFAILURE: The circuit did not produce the expected result.")