from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

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

# Allowed:
#   bits_per_var     in {2, 3, 4}
#   n_wavelet_coeffs in {3, 4, 5}
n_wavelet_coeffs = 5
bits_per_var = 2

n_split_vars = 2 * n_wavelet_coeffs
n_qubits = n_split_vars * bits_per_var

print(f"n_wavelet_coeffs = {n_wavelet_coeffs}")
print(f"n_split_vars     = {n_split_vars}")
print(f"bits_per_var     = {bits_per_var}")
print(f"n_qubits         = {n_qubits}")


# --------------------------------------------------
# Discretization of each positive variable t in [0,1]
# chosen consistently with bits_per_var
# --------------------------------------------------
L, U = 0.0, 1.0
delta = (U - L) / (2**bits_per_var - 1)

t_grid = np.linspace(L, U, 2**bits_per_var)
n_levels = len(t_grid)

print(f"delta    = {delta}")
print(f"t_grid   = {t_grid}")
print(f"n_levels = {n_levels}")

# --------------------------------------------------
# Haar basis and blur operator
# --------------------------------------------------
def haar_wavelet(N, j, k):
    x = np.zeros(N)
    block = N // (2**j)
    half = block // 2
    start = k * block
    x[start:start+half] = 1.0
    x[start+half:start+block] = -1.0
    x /= np.linalg.norm(x)
    return x

def gaussian_blur_matrix(N, sigma):
    A = np.zeros((N, N))
    idx = np.arange(N)

    for i in range(N):
        d = np.minimum(np.abs(idx - i), N - np.abs(idx - i))
        kernel = np.exp(-(d**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        A[i, :] = kernel

    return A


# --------------------------------------------------
# Minimal selectors for basis and compatible w_true
# --------------------------------------------------
def build_basis(N_signal, n_wavelet_coeffs):
    if n_wavelet_coeffs == 3:
        basis = [
            haar_wavelet(N_signal, 0, 0),
            haar_wavelet(N_signal, 1, 0),
            haar_wavelet(N_signal, 1, 1),
        ]
    elif n_wavelet_coeffs == 4:
        basis = [
            haar_wavelet(N_signal, 0, 0),
            haar_wavelet(N_signal, 1, 0),
            haar_wavelet(N_signal, 1, 1),
            haar_wavelet(N_signal, 2, 1),
        ]
    elif n_wavelet_coeffs == 5:
        basis = [
            haar_wavelet(N_signal, 0, 0),
            haar_wavelet(N_signal, 1, 0),
            haar_wavelet(N_signal, 1, 1),
            haar_wavelet(N_signal, 2, 1),
            haar_wavelet(N_signal, 2, 2),
        ]
    else:
        raise ValueError("n_wavelet_coeffs must be one of {3, 4, 5}")
    return basis


def suggested_w_true(n_wavelet_coeffs, bits_per_var):
    # Compatible grids:
    # bits_per_var = 2 -> multiples of 1/3
    # bits_per_var = 3 -> multiples of 1/7
    # bits_per_var = 4 -> multiples of 1/15
    candidates = {
        2: {
            3: np.array([1/3, -2/3, 1/3]),
            4: np.array([1/3, -2/3, 1/3, 2/3]),
            5: np.array([1/3, -2/3, 1/3, 2/3, 2/3]),
        },
        3: {
            3: np.array([1/7, -2/7, 1/7]),
            4: np.array([1/7, -2/7, 1/7, 6/7]),
            5: np.array([1/7, -2/7, 1/7, 6/7, 6/7]),
        },
        4: {
            3: np.array([2/15, -3/15, 3/15]),
            4: np.array([2/15, -3/15, 3/15, 13/15]),
            5: np.array([2/15, -3/15, 3/15, 13/15, 13/15]),
        },
    }

    if bits_per_var not in candidates:
        raise ValueError("bits_per_var must be one of {2, 3, 4}")
    if n_wavelet_coeffs not in candidates[bits_per_var]:
        raise ValueError("n_wavelet_coeffs must be one of {3, 4, 5}")

    return candidates[bits_per_var][n_wavelet_coeffs].copy()


# --------------------------------------------------
# Build reduced inverse problem
# x = C w, with w = t1 - t2
# --------------------------------------------------
N_signal = 64
A = gaussian_blur_matrix(N_signal, sigma=2.0)

basis = build_basis(N_signal, n_wavelet_coeffs)
C = np.column_stack(basis)

M = A @ C

# --------------------------------------------------
# Synthetic true signal
# --------------------------------------------------
w_true = suggested_w_true(n_wavelet_coeffs, bits_per_var)

print(f"w_true   = {w_true}")

x_true = C @ w_true

rng = np.random.default_rng(0)
noise_level = 0.02
y_clean = A @ x_true
y = y_clean + noise_level * rng.standard_normal(N_signal)


# --------------------------------------------------
# Regularization parameter
# --------------------------------------------------
lam = 0.05


# --------------------------------------------------
# Cost function in split variables
# --------------------------------------------------
def f_cost(z):
    t1 = z[:n_wavelet_coeffs]
    t2 = z[n_wavelet_coeffs:]
    w = t1 - t2
    data_term = np.linalg.norm(M @ w - y)**2
    reg_term = lam * np.sum(t1) + lam * np.sum(t2)
    return data_term + reg_term

def compute_binary_cost_coefficients(bits_per_var, delta, G, y, lam):
    """
    Build QUBO coefficients for

        ||G z - y||_2^2 + lam * sum(z)

    where each scalar z_r is encoded as

        z_r = delta * sum_{b=0}^{bits_per_var-1} 2^b q_{r,b}

    with q_{r,b} in {0,1}.
    """
    n_vars = G.shape[1]
    n_qubits_local = n_vars * bits_per_var

    Hz = G.T @ G
    cz = lam * np.ones(n_vars) - 2.0 * (G.T @ y)
    C0 = float(y @ y)

    S = np.zeros((n_vars, n_qubits_local), dtype=float)

    for r in range(n_vars):
        for b in range(bits_per_var):
            q_idx = r * bits_per_var + b
            S[r, q_idx] = delta * (2**b)

    Q = S.T @ Hz @ S
    ell = S.T @ cz

    h = np.zeros(n_qubits_local, dtype=float)
    J = np.zeros((n_qubits_local, n_qubits_local), dtype=float)

    for i in range(n_qubits_local):
        h[i] = Q[i, i] + ell[i]
        for j in range(i + 1, n_qubits_local):
            J[i, j] = 2.0 * Q[i, j]

    return C0, h, J


G = np.hstack([M, -M])
C0, h, J = compute_binary_cost_coefficients(bits_per_var, delta, G, y, lam)


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


eps = 0.03 #0.02
h_sparse, pairs = sparsify_qubo(h, J, eps=eps)


print(f"Number of active pair terms after sparsification: {len(pairs)}")


# --------------------------------------------------
# helper: convert z -> binary vector q
# --------------------------------------------------
def z_to_binary(z, bits_per_var, delta):
    n_vars = len(z)
    q = np.zeros(n_vars * bits_per_var)

    for r in range(n_vars):
        k = int(round(z[r] / delta))
        k = max(0, min(k, 2**bits_per_var - 1))
        for b in range(bits_per_var):
            q[r*bits_per_var + b] = (k >> b) & 1

    return q


# --------------------------------------------------
# helper: compute QUBO cost from q (sparse)
# --------------------------------------------------
def qubo_cost(q, C0, h, pairs):
    cost = C0 + np.dot(h, q)
    for i, j, Jij in pairs:
        cost += Jij * q[i] * q[j]
    return cost


# --------------------------------------------------
# test consistency
# --------------------------------------------------
print("\nChecking QUBO vs continuous cost\n")

for _ in range(5):
    t1 = delta * np.random.randint(0, 2**bits_per_var, size=n_wavelet_coeffs)
    t2 = delta * np.random.randint(0, 2**bits_per_var, size=n_wavelet_coeffs)

    z = np.concatenate([t1, t2])
    q = z_to_binary(z, bits_per_var, delta)

    cost_cont = f_cost(z)
    cost_qubo = qubo_cost(q, C0, h, pairs)

    print("t1 =", t1)
    print("t2 =", t2)
    print("continuous cost =", cost_cont)
    print("qubo cost       =", cost_qubo)
    print("difference      =", abs(cost_cont - cost_qubo))
    print()


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

T = 4.0
p = 3
dt = T / p

print(f"T     = {T}")
print(f"p     = {p}")

shots = 10000

qc = prepare_initial_state(n_qubits)

for k in range(1, p + 1):
    tau = k / p
    s = np.sin(0.5 * np.pi * tau)**2
    gamma_k = s * dt
    beta_k  = (1.0 - s) * dt

    apply_cost_unitary_sparse(qc, gamma_k, h_sparse, pairs)
    apply_mixer_unitary(qc, beta_k)

qc.measure_all()
tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=1234) # add seed 

print("transpiled depth:", tqc.depth())
print("transpiled size:", tqc.size())
print("transpiled ops:", tqc.count_ops())
print("size:", tqc.size())
print('Computing...')

result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

def bitstring_to_z_lsb_first(bitstring, bits_per_var, delta, n_split_vars):
    q = np.array([int(c) for c in bitstring[::-1]], dtype=float)  # LSB-first
    z = np.zeros(n_split_vars)

    for r in range(n_split_vars):
        k = 0
        for b in range(bits_per_var):
            k += int(q[r * bits_per_var + b]) * (2**b)
        z[r] = delta * k

    return z


def z_to_t1_t2(z, n_wavelet_coeffs):
    t1 = z[:n_wavelet_coeffs]
    t2 = z[n_wavelet_coeffs:]
    return t1, t2

def z_to_w(z, n_wavelet_coeffs):
    t1, t2 = z_to_t1_t2(z, n_wavelet_coeffs)
    return t1 - t2


def z_to_signal(z, n_wavelet_coeffs, C):
    w = z_to_w(z, n_wavelet_coeffs)
    return C @ w

top_by_cost = sorted(
    counts.items(),
    key=lambda kv: f_cost(
        bitstring_to_z_lsb_first(kv[0], bits_per_var, delta, n_split_vars)
    )
)
print("\nTop bitstrings by lowest cost:")
for bs, c in top_by_cost[:20]:
    z = bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars)
    t1, t2 = z_to_t1_t2(z, n_wavelet_coeffs)
    w = z_to_w(z, n_wavelet_coeffs)
    cost = f_cost(z)

    print(f"{bs}  count={c:5d}")
    print(f"   t1 = {np.round(t1, 3)}")
    print(f"   t2 = {np.round(t2, 3)}")
    print(f"   w  = {np.round(w, 3)}")
    print(f"   cost = {cost:.6f}")

# -----------------------------------------
# Collect samples
# -----------------------------------------
cost_samples = []
w_samples = []
x_samples = []

for bs, c in counts.items():
    z = bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars)
    w = z_to_w(z, n_wavelet_coeffs)
    x = z_to_signal(z, n_wavelet_coeffs, C)
    cost = f_cost(z)

    cost_samples.extend([cost] * c)
    w_samples.extend([w] * c)
    x_samples.extend([x] * c)


# -----------------------------------------
# Best sample = minimum cost, not maximum count
# -----------------------------------------
best_bs = min(
    counts,
    key=lambda bs: f_cost(bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars))
)

z_best = bitstring_to_z_lsb_first(best_bs, bits_per_var, delta, n_split_vars)
w_best = z_to_w(z_best, n_wavelet_coeffs)
x_best = z_to_signal(z_best, n_wavelet_coeffs, C)

print("max |h|:", np.max(np.abs(h_sparse)))
print("max |J|:", max(abs(Jij) for _,_,Jij in pairs))

# --------------------------------------------------
# Re-collect samples (safe to call again)
# --------------------------------------------------
cost_samples = []
w_samples    = []
x_samples    = []

for bs, c in counts.items():
    z    = bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars)
    w    = z_to_w(z, n_wavelet_coeffs)
    x    = z_to_signal(z, n_wavelet_coeffs, C)
    cost = f_cost(z)
    cost_samples.extend([cost] * c)
    w_samples.extend([w]    * c)
    x_samples.extend([x]    * c)

cost_arr = np.array(cost_samples)
w_arr    = np.array(w_samples)    # shape (n_shots, n_wavelet_coeffs)
x_arr    = np.array(x_samples)    # shape (n_shots, N_signal)

# Best sample
best_bs = min(
    counts,
    key=lambda bs: f_cost(
        bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars)
    )
)
z_best = bitstring_to_z_lsb_first(best_bs, bits_per_var, delta, n_split_vars)
w_best = z_to_w(z_best, n_wavelet_coeffs)
x_best = z_to_signal(z_best, n_wavelet_coeffs, C)
cost_best = f_cost(z_best)

# Mean estimate
w_mean = w_arr.mean(axis=0)
x_mean = x_arr.mean(axis=0)

# --------------------------------------------------
# Figure 1: Histogram of sampled objective values
# --------------------------------------------------
tag = f"bits{bits_per_var}_ncoeffs{n_wavelet_coeffs}_T{T}_p{p}_lambda{lam}_eps{eps}"

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(cost_arr, bins=80, density=True, color="#4C72B0", alpha=0.85)
ax.axvline(cost_best, color="red",    linestyle="--", linewidth=1.5,
           label=f"best-by-cost = {cost_best:.3f}")
ax.axvline(cost_arr.mean(), color="orange", linestyle=":",  linewidth=1.5,
           label=f"mean cost = {cost_arr.mean():.3f}")
ax.set_xlabel("Objective value", fontsize=12)
ax.set_ylabel("Density",         fontsize=12)
ax.set_title("Distribution of sampled objective values", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"cost_histogram_{tag}.png", dpi=300)
plt.close()
print("Saved cost_histogram")

# --------------------------------------------------
# Figure 2: True vs recovered signal (best + mean)
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_true, color="black",  linewidth=2,   label="True signal")
ax.plot(x_best, color="red",    linewidth=1.5,
        linestyle="--", label=f"Best recovered (cost={cost_best:.3f})")
ax.plot(x_mean, color="#4C72B0", linewidth=1.5,
        linestyle=":",  label="Mean recovered")
ax.set_xlabel("Signal index", fontsize=12)
ax.set_ylabel("Amplitude",    fontsize=12)
ax.set_title("True vs recovered signal", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"signal_recovery_{tag}.png", dpi=300)
plt.close()
print("Saved signal_recovery")

# --------------------------------------------------
# Figure 3: Wavelet coefficients — true vs best vs mean
# --------------------------------------------------
if n_wavelet_coeffs == 3:
    xlabels = [r'$\psi_{0,0}$', r'$\psi_{1,0}$', r'$\psi_{1,1}$']
elif n_wavelet_coeffs == 4:
    xlabels = [r'$\psi_{0,0}$', r'$\psi_{1,0}$',
               r'$\psi_{1,1}$', r'$\psi_{2,1}$']
else:
    xlabels = [r'$\psi_{0,0}$', r'$\psi_{1,0}$', r'$\psi_{1,1}$',
               r'$\psi_{2,1}$', r'$\psi_{2,2}$']

idx = np.arange(n_wavelet_coeffs)
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(idx - width, w_true, width, color="black",   alpha=0.85, label="True")
ax.bar(idx,         w_best, width, color="red",     alpha=0.85, label="Best recovered")
ax.bar(idx + width, w_mean, width, color="#4C72B0", alpha=0.85, label="Mean recovered")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(idx)
ax.set_xticklabels(xlabels, fontsize=12)
ax.set_ylabel("Coefficient value", fontsize=12)
ax.set_title("Wavelet coefficients: true vs recovered", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"wavelet_coeffs_{tag}.png", dpi=300)
plt.close()
print("Saved wavelet_coeffs")

 
# --------------------------------------------------
# Build full sample list
# --------------------------------------------------
all_samples = []
 
for bs, c in counts.items():
    z    = bitstring_to_z_lsb_first(bs, bits_per_var, delta, n_split_vars)
    t1   = z[:n_wavelet_coeffs].tolist()
    t2   = z[n_wavelet_coeffs:].tolist()
    w    = (z[:n_wavelet_coeffs] - z[n_wavelet_coeffs:]).tolist()
    x    = (C @ np.array(w)).tolist()
    cost = float(f_cost(z))
 
    all_samples.append({
        "bitstring": bs,
        "count":     c,
        "t1":        t1,
        "t2":        t2,
        "w":         w,
        "x":         x,
        "cost":      cost,
    })
 
# Sort by cost (cheapest first)
all_samples.sort(key=lambda d: d["cost"])
 
# --------------------------------------------------
# Best and mean estimates
# --------------------------------------------------
best = all_samples[0]
 
w_arr   = np.array([s["w"] for s in all_samples for _ in range(s["count"])])
x_arr   = np.array([s["x"] for s in all_samples for _ in range(s["count"])])
c_arr   = np.array([s["cost"] for s in all_samples for _ in range(s["count"])])
 
w_mean  = w_arr.mean(axis=0).tolist()
x_mean  = x_arr.mean(axis=0).tolist()
w_std   = w_arr.std(axis=0).tolist()
x_std   = x_arr.std(axis=0).tolist()
 
# --------------------------------------------------
# Assemble output dict
# --------------------------------------------------
output = {
    "run_info": {
        "n_qubits":         n_qubits,
        "n_wavelet_coeffs": n_wavelet_coeffs,
        "bits_per_var":     bits_per_var,
        "n_split_vars":     n_split_vars,
        "T":                T,
        "p":                p,
        "lam":              lam,
        "eps":              eps,
        "delta":            delta,
        "total_shots":      sum(c["count"] for c in all_samples),
        "unique_bitstrings": len(all_samples),
    },
    "ground_truth": {
        "w_true":  w_true.tolist(),
        "x_true":  x_true.tolist(),
        "y":       y.tolist(),
    },
    "best_sample": {
        "bitstring": best["bitstring"],
        "count":     best["count"],
        "t1":        best["t1"],
        "t2":        best["t2"],
        "w":         best["w"],
        "x":         best["x"],
        "cost":      best["cost"],
        "w_error":   float(np.linalg.norm(np.array(best["w"]) - w_true)),
        "x_error":   float(np.linalg.norm(np.array(best["x"]) - x_true)),
    },
    "mean_estimate": {
        "w":       w_mean,
        "x":       x_mean,
        "w_std":   w_std,
        "x_std":   x_std,
        "w_error": float(np.linalg.norm(np.array(w_mean) - w_true)),
        "x_error": float(np.linalg.norm(np.array(x_mean) - x_true)),
    },
    "cost_statistics": {
        "min":    float(c_arr.min()),
        "max":    float(c_arr.max()),
        "mean":   float(c_arr.mean()),
        "median": float(np.median(c_arr)),
        "std":    float(c_arr.std()),
    },
    "samples": all_samples,
}
 
# --------------------------------------------------
# Save to JSON
# --------------------------------------------------
filename = f"samples_{tag}.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)
 
print(f"Saved {len(all_samples)} unique bitstrings to {filename}")

