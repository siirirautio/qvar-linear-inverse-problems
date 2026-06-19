from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Initialize
# --------------------------------------------------

# Set backend
DEVICE_CORTEX_URL = os.getenv('Q50_CORTEX_URL')
provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
backend = provider.get_backend()

# --------------------------------------------------
# Problem dimensions
# --------------------------------------------------

# Set number of qubits used
n_qubits = 5
N = 2**n_qubits

# Discretize solution space
L, U = 0.6, 2.0
dx = (U - L) / (N - 1)
x_vals = np.linspace(L, U, N)

# Inverse problem: forward operator "a", measurement "g", regularization parameter "lam"
a, b, lam = 1.7, 2.3, 0.8

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

# Quadratic objective, data misfit + L2 regularization
def f_cost(x):
    return (a*x - b)**2 + lam*x**2

# Turning the grid objective into a Quadratic Unconstrained Binary Optimization (QUBO) in the bits
def compute_binary_cost_coefficients(n_qubits, L, dx, a, b, lam):
    alpha = (a**2 + lam)
    beta  = -2*a*b
    const = b**2

    A = alpha * (dx**2)
    B = 2*alpha*L*dx + beta*dx
    C = alpha*(L**2) + beta*L + const

    h = np.zeros(n_qubits, dtype=float)
    J = np.zeros((n_qubits, n_qubits), dtype=float)

    for i in range(n_qubits):
        h[i] = A*(2**(2*i)) + B*(2**i)
        for j in range(i+1, n_qubits):
            J[i, j] = 2*A*(2**(i+j))

    return C, h, J # C: constant shift
                   # h: linear terms
                   # J: quadratic couplings, upper triangular

# Initial state, creating the uniform superposition
def prepare_initial_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    return qc

# Mixer unitary
def apply_mixer_unitary(qc, beta):
    for i in range(qc.num_qubits):
        qc.rx(-2.0 * beta, i)

# Cost unitary from the QUBO, “problem Hamiltonian” phase separator
def apply_cost_unitary_from_qubo(qc, gamma, C0, h, J):
    n = qc.num_qubits

    z_coeff = -0.5 * h.astype(float) # qubits naturally give you Pauli Z eigenvalues, so need to convert
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                z_coeff[i] += -0.25 * J[i, j]
                z_coeff[j] += -0.25 * J[i, j]

    # Apply single-qubit Z phases: exp(-i * gamma * z_coeff[i] * Z_i)
    # Qiskit: RZ(theta) = exp(-i * theta/2 * Z), so choose theta = 2 * gamma * z_coeff
    for i in range(n):
        theta = 2.0 * gamma * z_coeff[i]
        if abs(theta) > 0:
            qc.rz(theta, i)

    # Apply ZZ phases: exp(-i * gamma * (J_ij/4) * Z_i Z_j)
    # Implement with CX-RZ-CX, where:
    #   CX(i->j); RZ(theta) on j; CX(i->j) gives exp(-i * theta/2 * Z_i Z_j)
    # So need theta = 2 * gamma * (J_ij/4) = gamma * J_ij / 2
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                theta_zz = gamma * J[i, j] / 2.0
                qc.rzz(theta_zz, i, j)
    
# --------------------------------------------------
# QVaR method
# --------------------------------------------------

# The exact solution
x_star = (a*b) / (a**2 + lam) 

# Build the QUBO coefficients for the discretized problem
C0, h, J = compute_binary_cost_coefficients(n_qubits, L, dx, a, b, lam)

# Rescale the Hamiltonian coefficients (so the biggest coefficient is about 1)
scale = max(np.max(np.abs(h)), np.max(np.abs(J)))
h = h / scale
J = J / scale
C0 = C0 / scale

# Sparsify: drop weak couplings
eps = 0.15  
J[np.abs(J) < eps] = 0.0

# “Annealing” schedule and Trotterization parameters
# (This is classic Trotterized time evolution: approximate a continuous-time process with many small steps)
# --------------------------------------------------
# Choose from T = p: 2, 100, 200
# --------------------------------------------------
T = 200     # total evolution time, Large T → slow evolution → closer to adiabatic behavior
p = 200     # number of discrete time steps (layers), The number of alternating cost/mixer layers, The circuit depth multiplier
dt = T / p

# Sampling budget: After building the circuit, you measure it shots times to get an empirical distribution over bitstrings (thus over x_k)
shots = 10000  # number of samples

# Build the circuit
qc = prepare_initial_state(n_qubits)

# The annealing schedule: slowly turn on the cost, turn off the mixer
for k in range(1, p + 1):
    #s = k / p                 # linear schedule in [0,1]
    tau = k / p                # normalized time
    s = np.sin(0.5*np.pi*tau)**2 # smoothly goes from 0 to 1 with zero slope at endpoints (nice “adiabatic-ish” schedule)
    gamma_k = s * dt           # cost time slice, controls how much cost phase you apply that step 
    beta_k  = (1.0 - s) * dt   # mixer time slice, controls how much mixer you apply that step

    # Trotter step: exp(-i s H_C dt) then exp(-i (1-s) H_M dt), i.e. cost evolution then mixer evolution
    apply_cost_unitary_from_qubo(qc, gamma_k, C0, h, J)
    apply_mixer_unitary(qc, beta_k)

# Measure the final state
qc.measure_all()
qc_t = transpile(
    qc,
    backend=backend,
    optimization_level=1,
    seed_transpiler=1234
)
result = backend.run(
    qc_t,
    shots=shots
).result()
counts = result.get_counts()  # dictionary like {bitstring: count} giving observed measurement frequencies

# --------------------------------------------------
# Build full sample list
# --------------------------------------------------
all_samples = []

for bs, c in counts.items():
    k_idx = int(bs, 2)   
    x     = float(x_vals[k_idx])
    cost  = float(f_cost(x))
    all_samples.append({
        "bitstring": bs,
        "count":     int(c),
        "x":         x,
        "cost":      cost,
    })

# Sort by cost (cheapest first)
all_samples.sort(key=lambda d: d["cost"])

# --------------------------------------------------
# Expand samples weighted by count
# --------------------------------------------------
x_exp    = np.array([s["x"]    for s in all_samples for _ in range(s["count"])])
cost_exp = np.array([s["cost"] for s in all_samples for _ in range(s["count"])])

best = all_samples[0]
k_map    = max(counts, key=counts.get)
x_map    = float(x_vals[int(k_map, 2)])

# --------------------------------------------------
# Assemble output dict
# --------------------------------------------------
ops = qc_t.count_ops()
output = {
    "run_info": {
        "n_qubits":    int(np.log2(N)),
        "N":           N,
        "L":           L,
        "U":           U,
        "dx":          float((U - L) / (N - 1)),
        "a":           a,
        "b":           b,
        "lam":         lam,
        "T":           T,
        "p":           p,
        "eps":         eps,
        "scale":       float(scale),
        "total_shots": int(sum(s["count"] for s in all_samples)),
        "unique_bitstrings": len(all_samples),
        "gate_counts": dict(ops),
        "circuit_depth": qc_t.depth(),
    },
    "ground_truth": {
        "x_star": float(x_star),
    },
    "best_sample": {
        "bitstring": best["bitstring"],
        "count":     best["count"],
        "x":         best["x"],
        "cost":      best["cost"],
    },
    "map_estimate": {
        "bitstring": k_map,
        "x":         x_map,
        "cost":      float(f_cost(x_map)),
    },
    "mean_estimate": {
        "x":    float(x_exp.mean()),
        "cost": float(cost_exp.mean()),
    },
    "cost_statistics": {
        "min":    float(cost_exp.min()),
        "max":    float(cost_exp.max()),
        "mean":   float(cost_exp.mean()),
        "median": float(np.median(cost_exp)),
        "std":    float(cost_exp.std()),
    },
    "samples": all_samples,
}

# Save results to JSON-file
tag = f"T{T}_p{p}_eps{eps}"
filename = f"samples_q50_{tag}.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved {len(all_samples)} unique bitstrings to {filename}")

# Endianness sanity check
top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
print("\nTop bitstrings (check decoding):")
for bs, c in top:
    k_no = int(bs, 2)        # treats the leftmost char as MSB (standard binary)
    k_rev = int(bs[::-1], 2) # treats the rightmost char as MSB (i.e., flips endianness)
    print(f"{bs}  count={c:5d}  x(no-rev)={x_vals[k_no]: .4f}   x(rev)={x_vals[k_rev]: .4f}")

# Build a histogram over decoded x values
x_samples = []
for bs, c in counts.items():
    k_idx = int(bs, 2)
    x = x_vals[k_idx]
    x_samples.extend([x] * c)

# Create and save histogram
plt.figure(figsize=(6,4))
plt.hist(x_samples, bins=N, density=True, range=(L, U))
plt.xlim(L, U)
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Histogram of sampled x values")
plt.tight_layout()
plt.savefig("histogram_q50_T{T}_p{p}_eps{eps}.png", dpi=300)
plt.close()

# Print results and information
print(x_star)
print(np.mean(x_samples))
k_map = max(counts, key=counts.get)
x_map = x_vals[int(k_map, 2)]
print(x_map)

print ('Total gate count:', sum(qc.count_ops().values()))

print("\n=== ORIGINAL CIRCUIT OPS ===")
ops = qc.count_ops()
print(ops)
print("depth:", qc.depth())
print("rzz count:", ops.get("rzz", 0))
