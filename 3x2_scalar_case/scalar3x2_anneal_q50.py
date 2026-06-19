from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
from collections import Counter
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats

# Backend
DEVICE_CORTEX_URL = os.getenv('Q50_CORTEX_URL')
provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
backend = provider.get_backend()

# ----------------------------
# Problem: min ||A x - g||^2 + alpha ||x||^2
# ----------------------------
A = np.array([[1, -1],
              [1, -2],
              [2,  1]], dtype=float)

g = np.array([2.1, 2.9, 1.1], dtype=float)
alpha = 0.5

bits_per_var = 6
bounds = [(0.5, 1.5), (-1.5, -0.5)]

n_vars = 2
n_qubits = n_vars * bits_per_var
n_levels = 2**bits_per_var

print("n_qubits =", n_qubits)

# ----------------------------
# Encoding: x_r = L_r + delta_r * integer
# ----------------------------
L = np.array([b[0] for b in bounds], dtype=float)
U = np.array([b[1] for b in bounds], dtype=float)
delta = (U - L) / (n_levels - 1)

# Per-variable settings
n1 = bits_per_var
n2 = bits_per_var

L1, U1 = bounds[0]
L2, U2 = bounds[1]

dx1 = delta[0]
dx2 = delta[1]

lam = alpha
eps = 0.01

def bitstring_to_x(bitstring):
    q = np.array([int(c) for c in bitstring[::-1]], dtype=int)  # LSB-first
    x = np.zeros(n_vars)

    for r in range(n_vars):
        k = 0
        for b in range(bits_per_var):
            k += q[r * bits_per_var + b] * (2**b)
        x[r] = L[r] + delta[r] * k

    return x

def objective(x):
    return np.linalg.norm(A @ x - g)**2 + alpha * np.dot(x, x)

# ----------------------------
# Build QUBO for objective
# ----------------------------
def build_qubo(A, g, alpha):
    H = A.T @ A + alpha * np.eye(n_vars)
    c = -2.0 * (A.T @ g)
    C0 = float(g @ g)

    S = np.zeros((n_vars, n_qubits))
    for r in range(n_vars):
        for b in range(bits_per_var):
            S[r, r * bits_per_var + b] = delta[r] * (2**b)

    # x = L + S q
    C0 = C0 + float(L @ H @ L + c @ L)
    ell = S.T @ (2.0 * H @ L + c)
    Q = S.T @ H @ S

    h = np.zeros(n_qubits)
    J = np.zeros((n_qubits, n_qubits))

    for i in range(n_qubits):
        h[i] = Q[i, i] + ell[i]
        for j in range(i + 1, n_qubits):
            J[i, j] = 2.0 * Q[i, j]

    return C0, h, J

C0, h, J = build_qubo(A, g, alpha)

def sparsify_qubo(h, J, eps=5e-3):
    h_sparse = h.copy()
    h_sparse[np.abs(h_sparse) < eps] = 0.0

    pairs = []
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) >= eps:
                pairs.append((i, j, J[i, j]))

    return h_sparse, pairs


eps = 0.05
h_sparse, pairs = sparsify_qubo(h, J, eps=eps)

print(f"Number of active pair terms after sparsification: {len(pairs)}")


# --------------------------------------------------
# Scale sparse QUBO coefficients
# --------------------------------------------------
max_h = np.max(np.abs(h_sparse)) if np.any(h_sparse) else 0.0
max_J = max((abs(Jij) for _, _, Jij in pairs), default=0.0)
scale = max(max_h, max_J)

if scale == 0.0:
    scale = 1.0

h_sparse = h_sparse / scale

pairs = [(i, j, Jij / scale) for i, j, Jij in pairs]
C0 = C0 / scale

# ----------------------------
# Quantum circuit
# ----------------------------
def prepare_initial_state(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    return qc
   
def apply_cost_unitary_sparse(qc, gamma, h, pairs, eps=1e-12):
    n = qc.num_qubits
    z_coeff = -0.5 * h.astype(float)

    for i, j, Jij in pairs:
        z_coeff[i] += -0.25 * Jij
        z_coeff[j] += -0.25 * Jij

    for i in range(n):
        theta = 2.0 * gamma * z_coeff[i]
        if abs(theta) > eps:
            qc.rz(theta, i)

    for i, j, Jij in pairs:
        theta_zz = gamma * Jij / 2.0
        if abs(theta_zz) > eps:
            qc.rzz(theta_zz, i, j)

def apply_mixer_unitary(qc, beta):
    for i in range(qc.num_qubits):
        qc.rx(-2.0 * beta, i)

def bitstring_to_x1_x2(bitstring, n1, n2, L1, dx1, L2, dx2):
    bits = bitstring[::-1]  # if keeping current LSB-first convention

    bits1 = bits[:n1]
    bits2 = bits[n1:n1 + n2]

    k1 = sum(int(b) << i for i, b in enumerate(bits1))
    k2 = sum(int(b) << i for i, b in enumerate(bits2))

    x1 = L1 + k1 * dx1
    x2 = L2 + k2 * dx2

    return x1, x2
    
def true_cost(x1, x2, A, g, lam):
    f = np.array([x1, x2], dtype=float)
    r = A @ f - g
    return r @ r + lam * (f @ f)

T = 2.0 #3.0
p = 4 #3 #6
dt = T / p
shots = 10000 

# ----------------------------
# Run QUBO
# ----------------------------
qc = prepare_initial_state(n_qubits)

for k in range(1, p + 1):
    tau = k / p
    s = np.sin(0.5 * np.pi * tau)**2
    gamma = s * dt
    beta = (1.0 - s) * dt

    apply_cost_unitary_sparse(qc, gamma, h_sparse, pairs,eps)
    apply_mixer_unitary(qc, beta)

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

counts = result.get_counts()

# ----------------------------
# Read result
# ----------------------------
best_bs = min(counts, key=lambda bs: objective(bitstring_to_x(bs)))
x_best = bitstring_to_x(best_bs)

print("\nBest bitstring:", best_bs)
print("x_best =", x_best)
print("objective =", objective(x_best))

x_ridge = np.linalg.solve(A.T @ A + alpha * np.eye(2), A.T @ g)
print("\nContinuous ridge solution:", x_ridge)

print("\nTop samples:")
for bs, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
    x = bitstring_to_x(bs)
    print(bs, "count =", c, "x =", np.round(x, 4), "cost =", objective(x))
    
print("\nTop samples by cost:")
for bs, c in sorted(counts.items(), key=lambda kv: objective(bitstring_to_x(kv[0])))[:10]:
    x = bitstring_to_x(bs)
    print(bs, "count =", c, "x =", np.round(x, 4), "cost =", objective(x))

# ----------------------------
# Endianness sanity check
# ----------------------------

top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

print("\nTop bitstrings (check decoding):")

for bs, c in top:
    x1, x2 = bitstring_to_x1_x2(
        bs, n1, n2,
        L1, dx1,
        L2, dx2
    )

    print(
        f"{bs}  count={c:5d}  "
        f"x1={x1:.4f}  x2={x2:.4f}"
    )

# ----------------------------
# Build sample arrays
# ----------------------------

x1_samples = []
x2_samples = []
cost_samples = []

for bs, c in counts.items():

    x1, x2 = bitstring_to_x1_x2(
        bs, n1, n2,
        L1, dx1,
        L2, dx2
    )

    val = true_cost(x1, x2, A, g, lam)

    x1_samples.extend([x1] * c)
    x2_samples.extend([x2] * c)
    cost_samples.extend([val] * c)

# ----------------------------
# Basic statistics
# ----------------------------

# The exact solution
f_star = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ g)

print("\nContinuous optimum:")
print(f_star)

print("\nSample statistics:")
print("mean x1:", np.mean(x1_samples))
print("mean x2:", np.mean(x2_samples))

# MAP estimate
bs_map = max(counts, key=counts.get)

x1_map, x2_map = bitstring_to_x1_x2(
    bs_map,
    n1, n2,
    L1, dx1,
    L2, dx2,
)

print("MAP estimate:", x1_map, x2_map)

# Circuit statistics
print("\nCircuit statistics")

print(qc_t.count_ops())
print("depth =", qc_t.depth())

ops = qc_t.count_ops()

print('Total gate count:', sum(ops.values()))
print("rzz count:", ops.get("rzz", 0))

# ----------------------------
# Histogram x1
# ----------------------------

plt.figure(figsize=(6, 4))

plt.hist(
    x1_samples,
    bins=32,
    range=(L1 - dx1 / 2, U1 + dx1 / 2),
    density=True,
)

plt.xlim(L1, U1)

plt.xlabel("x1")
plt.ylabel("Density")

plt.title("Histogram of sampled x1 values")

plt.tight_layout()

filename = (
    f"scalar3x2_x1_histogram_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# ----------------------------
# Histogram x2
# ----------------------------

plt.figure(figsize=(6, 4))

plt.hist(
    x2_samples,
    bins=32,
    range=(L2 - dx2 / 2, U2 + dx2 / 2),
    density=True,
)

plt.xlim(L2, U2)

plt.xlabel("x2")
plt.ylabel("Density")

plt.title("Histogram of sampled x2 values")

plt.tight_layout()

filename = (
    f"scalar3x2_x2_histogram_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# ----------------------------
# Histogram of objective values
# ----------------------------

plt.figure(figsize=(6, 4))

plt.hist(
    cost_samples,
    bins=60,
    density=True
)

plt.title("Histogram of sampled objective values")

plt.xlabel("cost")
plt.ylabel("density")

plt.grid(alpha=0.3)

filename = (
    f"scalar3x2_cost_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# ----------------------------
# Objective contour
# ----------------------------

def f_cost_contour(x1, x2):

    x = np.array([x1, x2])

    Q = A.T @ A + lam * np.eye(2)
    c = -2 * (A.T @ g)

    return x @ Q @ x + c @ x + g @ g

x1_plot = np.linspace(0.5, 1.5, 200)
x2_plot = np.linspace(-1.5, -0.5, 200)

X1, X2 = np.meshgrid(x1_plot, x2_plot)

Z = np.zeros_like(X1)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f_cost_contour(
            X1[i, j],
            X2[i, j]
        )

# Mean estimate
mean_est = np.array([
    np.mean(x1_samples),
    np.mean(x2_samples),
])

# Top samples
top10 = sorted(
    counts.items(),
    key=lambda kv: kv[1],
    reverse=True,
)[:10]

samples = []

for bs, _ in top10:

    x1, x2 = bitstring_to_x1_x2(
        bs,
        n1, n2,
        L1, dx1,
        L2, dx2,
    )

    samples.append([x1, x2])

samples = np.array(samples)


# ----------------------------
# Contour plot
# ----------------------------

plt.figure(figsize=(7, 6))

plt.contour(
    X1,
    X2,
    Z,
    levels=15,
    colors='black',
    linewidths=1,
)


# True minimum
plt.plot(
    f_star[0],
    f_star[1],
    'kp',
    markersize=10,
    markerfacecolor='k',
    label="True minimum",
)

# MAP
plt.plot(
    x1_map,
    x2_map,
    'ro',
    markersize=8,
    linewidth=2,
    label="MAP",
)

# Mean
plt.plot(
    mean_est[0],
    mean_est[1],
    'bx',
    markersize=8,
    linewidth=2,
    label="Mean",
)

# Top samples
plt.plot(
    samples[:, 0],
    samples[:, 1],
    '.',
    color=[0.2, 0.5, 0.8],
    markersize=12,
    label="Top samples",
)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.legend(loc='best')

plt.grid(True)

plt.axis('equal')

plt.tight_layout()

filename = (
    f"scalar3x2_objectivecontour_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# ----------------------------
# 2D histogram with eigendirections
# ----------------------------

plt.figure(figsize=(6, 5))

plt.hist2d(
    x1_samples,
    x2_samples,
    bins=32,
    range=[[L1, U1], [L2, U2]],
    density=True,
    cmap="viridis",
)

plt.colorbar(label="Density")

Q = A.T @ A + lam * np.eye(2)

eigvals, eigvecs = np.linalg.eigh(Q)

center = f_star

arrow_scale = 0.6

for k in range(2):

    v = eigvecs[:, k]

    length = arrow_scale / np.sqrt(eigvals[k])

    plt.arrow(
        center[0],
        center[1],
        length * v[0],
        length * v[1],
        width=0.01,
        head_width=0.05,
        length_includes_head=True,
    )

    plt.arrow(
        center[0],
        center[1],
        -length * v[0],
        -length * v[1],
        width=0.01,
        head_width=0.05,
        length_includes_head=True,
    )

plt.scatter(
    f_star[0],
    f_star[1],
    color="red",
    marker="x",
    s=100,
    label="optimum",
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title("2D histogram of sampled (x1, x2)")

plt.legend()

plt.tight_layout()

filename = (
    f"scalar32_x1x2_hist2d_eigendirs_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# ----------------------------
# Credible-region plot
# ----------------------------

plt.figure(figsize=(6, 5))

plt.hist2d(
    x1_samples,
    x2_samples,
    bins=32,
    range=[[L1, U1], [L2, U2]],
    density=True,
    cmap="viridis",
)

plt.colorbar(label="Density")

Sigma = np.linalg.inv(Q)

eigvals, eigvecs = np.linalg.eigh(Sigma)

order = np.argsort(eigvals)[::-1]

eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

angle = np.degrees(
    np.arctan2(
        eigvecs[1, 0],
        eigvecs[0, 0]
    )
)

center = f_star

levels = [
    (2.30, "68% credible region", "--"),
    (5.99, "95% credible region", "-"),
]

for c, label, linestyle in levels:

    width = 2 * np.sqrt(eigvals[0] * c)
    height = 2 * np.sqrt(eigvals[1] * c)

    ellipse = patches.Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
        linestyle=linestyle,
        label=label,
    )

    plt.gca().add_patch(ellipse)

plt.scatter(
    center[0],
    center[1],
    color='red',
    marker='x',
    s=100,
    label="posterior mean",
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title("2D histogram with posterior credible regions")

plt.legend()

plt.tight_layout()

filename = (
    f"scalar32_x1x2_hist2d_credible_"
    f"T{T}_p{p}_eps{eps}_"
    f"nvars{n_vars}_bits{bits_per_var}.png"
)

plt.savefig(filename, dpi=300)
plt.show()

# Convert counts dict to compact arrays
bitstrings = np.array(list(counts.keys()))
count_values = np.array(list(counts.values()), dtype=np.int64)

# Decode once
decoded = np.array([
    bitstring_to_x1_x2(bs, n1, n2, L1, dx1, L2, dx2)
    for bs in bitstrings
])
x1_values = decoded[:, 0]
x2_values = decoded[:, 1]

cost_values = np.array([
    true_cost(x1, x2, A, g, lam)
    for x1, x2 in zip(x1_values, x2_values)
])

metadata = {
    "n1": n1,
    "n2": n2,
    "L1": L1,
    "U1": U1,
    "L2": L2,
    "U2": U2,
    "dx1": dx1,
    "dx2": dx2,
    "T": T,
    "p": p,
    "dt": dt,
    "eps": eps,
    "shots": shots,
    "scale": scale,
    "f_star": f_star.tolist(),
    "gate_counts": dict(qc.count_ops()),
    "backend": str(backend),
}

np.savez_compressed(
    f"scalar3x2_results_T{T}_p{p}_eps{eps}.npz",
    bitstrings=bitstrings,
    counts=count_values,
    x1=x1_values,
    x2=x2_values,
    cost=cost_values,
    h=h,
    J=J,
    C0=C0,
    A=A,
    g=g,
    lam=lam,
    metadata=json.dumps(metadata),
)

