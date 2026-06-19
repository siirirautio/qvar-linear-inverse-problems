import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
})

def haar_wavelet(N, j, k):
    x = np.zeros(N)
    block = N // (2**j)
    half = block // 2
    start = k * block
    x[start:start+half] = 1.0
    x[start+half:start+block] = -1.0
    x /= np.linalg.norm(x)
    return x

N_signal = 64
basis = [
    haar_wavelet(N_signal, 0, 0),
    haar_wavelet(N_signal, 1, 0),
    haar_wavelet(N_signal, 1, 1),
    haar_wavelet(N_signal, 2, 1),
    haar_wavelet(N_signal, 2, 2),
]
C = np.column_stack(basis)

xlabels = [r'$\psi_{0,0}$', r'$\psi_{1,0}$', r'$\psi_{1,1}$',
           r'$\psi_{2,1}$', r'$\psi_{2,2}$']

files = {
    2: "samples_bits2_ncoeffs5_T4_0_p3_lambda0_05_eps0_03.json",
    3: "samples_bits3_ncoeffs5_T4_0_p3_lambda0_05_eps0_03.json",
    4: "samples_bits4_ncoeffs5_T4_0_p3_lambda0_05_eps0_03.json",
}

for nb, path in files.items():
    with open(path) as f:
        data = json.load(f)

    w_true = np.array(data["ground_truth"]["w_true"])
    w_best = np.array(data["best_sample"]["w"])
    w_mean = np.array(data["mean_estimate"]["w"])
    x_true = C @ w_true
    x_best = C @ w_best
    x_mean = C @ w_mean
    cost_best = data["best_sample"]["cost"]

    cost_samples = []
    for s in data["samples"]:
        cost_samples.extend([s["cost"]] * s["count"])
    cost_arr = np.array(cost_samples)

    tag = f"bits{nb}_ncoeffs5_T4.0_p3_lambda0.05_eps0.03"

    # --------------------------------------------------
    # Figure 1: Cost histogram
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(cost_arr, bins=80, density=True, color="#4C72B0", alpha=0.85)
    ax.axvline(cost_best, color="red", linestyle="--", linewidth=2.0,
               label="Best-by-cost")
    ax.axvline(cost_arr.mean(), color="orange", linestyle=":", linewidth=2.0,
               label="Mean cost")
    ax.set_xlabel("Objective value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cost_histogram_{tag}.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 2: Wavelet coefficients
    # --------------------------------------------------
    idx = np.arange(5)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(idx - width, w_true, width, color="black",   alpha=0.85, label="True")
    ax.bar(idx,         w_best, width, color="red",     alpha=0.85, label="Best-by-cost")
    ax.bar(idx + width, w_mean, width, color="#4C72B0", alpha=0.85, label="Mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Coefficient value")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"wavelet_coeffs_{tag}.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 3: Signal recovery
    # Thicker lines; legend placed below xlabel at figure bottom
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_true, color="black",   linewidth=3.5, label="True")
    ax.plot(x_best, color="red",     linewidth=3.0, linestyle="--",
            label="Best-by-cost")
    ax.plot(x_mean, color="#4C72B0", linewidth=3.0, linestyle=":",
            label="Mean")
    ax.set_xlabel("Signal index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=True,
               fontsize=22)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(f"signal_recovery_{tag}.png", dpi=300)
    plt.close()

    print(f"Saved figures for n_b={nb}")

print("Done.")
