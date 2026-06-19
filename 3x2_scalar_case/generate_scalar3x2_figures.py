import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.sparse import diags as sp_diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
})

OUT = "."  # change to output directory if needed

def load(fname):
    d = np.load(fname, allow_pickle=True)
    meta = json.loads(str(d['metadata']))
    counts = d['counts']
    x1 = d['x1']; x2 = d['x2']; cost = d['cost']
    x1_exp = np.repeat(x1, counts)
    x2_exp = np.repeat(x2, counts)
    best_idx = np.argmin(cost)
    map_idx  = np.argmax(counts)
    return dict(
        x1=x1, x2=x2, cost=cost, counts=counts,
        x1_exp=x1_exp, x2_exp=x2_exp,
        cost_exp=np.repeat(cost, counts),
        map_x1=x1[map_idx], map_x2=x2[map_idx],
        mean_x1=x1_exp.mean(), mean_x2=x2_exp.mean(),
        f_star=np.array(meta['f_star']),
        L1=meta['L1'], U1=meta['U1'], L2=meta['L2'], U2=meta['U2'],
        dx1=meta['dx1'],
        A=d['A'], g=d['g'], lam=float(d['lam']),
        best_cost=cost[best_idx],
        h=d['h'], J=d['J'], scale=meta['scale'],
    )

qpu = load("scalar3x2_results_T2_0_p4_eps0_05.npz")
sim = load("scalar3x2_results_simulated_T2_0_p4_eps0_05.npz")

A = qpu['A']; g = qpu['g']; lam = qpu['lam']
f_star = qpu['f_star']
L1,U1,L2,U2 = qpu['L1'],qpu['U1'],qpu['L2'],qpu['U2']
dx1 = qpu['dx1']

Q      = A.T @ A + lam * np.eye(2)
Sigma  = np.linalg.inv(Q)
eigvals_Q, eigvecs_Q = np.linalg.eigh(Q)
eigvals_S, eigvecs_S = np.linalg.eigh(Sigma)
order = np.argsort(eigvals_S)[::-1]
eigvals_S = eigvals_S[order]; eigvecs_S = eigvecs_S[:, order]
angle = np.degrees(np.arctan2(eigvecs_S[1,0], eigvecs_S[0,0]))

cred_levels = [
    (0.20, "20%",  ":",  2.0, "#CC00CC"),
    (0.40, "40%",  "--", 2.0, "#fdae61"),
    (0.60, "60%",  "-",  2.2, "#d7191c"),
]

def top10_samples(d):
    idx = np.argsort(d['counts'])[::-1][:10]
    return d['x1'][idx], d['x2'][idx]

def add_credible_ellipses(ax):
    for p, label, ls, lw, col in cred_levels:
        chi2 = -2*np.log(1-p)
        w   = 2*np.sqrt(eigvals_S[0]*chi2)
        h_e = 2*np.sqrt(eigvals_S[1]*chi2)
        ellipse = patches.Ellipse(xy=f_star, width=w, height=h_e,
                                  angle=angle, fill=False,
                                  linewidth=lw, linestyle=ls, color=col)
        ax.add_patch(ellipse)

# ── Cost histograms (2 figures) ─────────────────────────────────────────────
for d, tag in [(sim,"sim"), (qpu,"qpu")]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(d['cost_exp'], bins=60, density=True, color="#4C72B0", alpha=0.85)
    ax.axvline(d['best_cost'], color="red", linestyle="--", linewidth=2.5,
               label="Best-by-cost")
    ax.axvline(d['cost_exp'].mean(), color="orange", linestyle=":", linewidth=2.5,
               label="Mean cost")
    ax.set_xlabel("Objective value"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/scalar3x2_cost_{tag}.png", dpi=300)
    plt.close()

# ── x1 and x2 histograms (4 figures) ───────────────────────────────────────
# Legend placed below the axes via fig.legend so it never overlaps histogram
# or xlabel. The axvline calls add the lines; dummy Line2D handles go to legend.
for d, tag in [(sim,"sim"), (qpu,"qpu")]:
    for xdata, xmap, xmean, xtrue, xlab, xl, xu, vtag in [
        (d['x1_exp'], d['map_x1'], d['mean_x1'], f_star[0], r"$x_1$", L1, U1, "x1"),
        (d['x2_exp'], d['map_x2'], d['mean_x2'], f_star[1], r"$x_2$", L2, U2, "x2"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(xdata, bins=32, range=(xl-dx1/2, xu+dx1/2),
                density=True, color="#4C72B0", alpha=0.85)
        ax.axvline(xtrue, color="black",  linestyle="-",  linewidth=2.5)
        ax.axvline(xmean, color="orange", linestyle=":",  linewidth=2.5)
        ax.axvline(xmap,  color="red",    linestyle="--", linewidth=2.5)
        ax.set_xlim(xl, xu)
        ax.set_xlabel(xlab); ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        # Build legend handles separately so fig.legend can place them below
        leg_handles = [
            Line2D([0],[0], color="black",  linestyle="-",  linewidth=2.5,
                   label="True value"),
            Line2D([0],[0], color="orange", linestyle=":",  linewidth=2.5,
                   label="Mean"),
            Line2D([0],[0], color="red",    linestyle="--", linewidth=2.5,
                   label="MAP"),
        ]
        fig.legend(handles=leg_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.0), ncol=3,
                   frameon=True, fontsize=18, borderaxespad=0.2)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.30)
        plt.savefig(f"{OUT}/scalar3x2_{vtag}_histogram_{tag}.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

# ── Posterior level sets + eigendirections (2 figures) ──────────────────────
# Legend outside on the right in 1 column — never overlaps axes or xlabel
# Abbreviated labels: short enough to fit 4 per row below a square plot
legend_contour = [
    Line2D([0],[0], color="#CC00CC", ls=":",  lw=2.0, label="20%"),
    Line2D([0],[0], color="#fdae61", ls="--", lw=2.0, label="40%"),
    Line2D([0],[0], color="#d7191c", ls="-",  lw=2.2, label="60%"),
    Line2D([0],[0], color='navy',    ls='-',  lw=1.5, label="Eig."),
    Line2D([0],[0], color=[0.2,0.5,0.8], marker='.', ls='None',
           markersize=12, label="Samp."),
    Line2D([0],[0], color='black', marker='*', ls='None',
           markersize=12, label="Min."),
    Line2D([0],[0], color='red',   marker='o', ls='None',
           markersize=9,  label="MAP"),
    Line2D([0],[0], color='blue',  marker='x', ls='None',
           markersize=9, markeredgewidth=2.5, label="Mean"),
]

for d, tag in [(sim,"sim"), (qpu,"qpu")]:
    fig, ax = plt.subplots(figsize=(7, 7))
    add_credible_ellipses(ax)
    arrow_scale = 0.25
    for k in range(2):
        v = eigvecs_S[:, k]
        length = arrow_scale * np.sqrt(eigvals_S[k])
        for sign in [1, -1]:
            ax.arrow(f_star[0], f_star[1],
                     sign*length*v[0], sign*length*v[1],
                     width=0.007, head_width=0.028,
                     length_includes_head=True, color='navy')
    tx, ty = top10_samples(d)
    ax.plot(tx, ty, '.', color=[0.2,0.5,0.8], markersize=12)
    ax.plot(f_star[0], f_star[1], 'k*', markersize=16)
    ax.plot(d['map_x1'],  d['map_x2'],  'ro', markersize=10)
    ax.plot(d['mean_x1'], d['mean_x2'], 'bx', markersize=10, markeredgewidth=2.5)
    ax.set_xlim(L1, U1); ax.set_ylim(L2, U2)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.legend(handles=legend_contour, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=4,
               frameon=True, fontsize=16, borderaxespad=0.2,
               columnspacing=0.8, handlelength=1.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(f"{OUT}/scalar3x2_posterior_contours_{tag}.png", dpi=300,
                bbox_inches='tight')
    plt.close()

# ── 2D histogram + eigendirections + credible ellipses (2 figures) ──────────
legend_eigendirs = [
    Line2D([0],[0], color="#CC00CC", linestyle=":",  linewidth=2.0, label="20% cred."),
    Line2D([0],[0], color="#fdae61", linestyle="--", linewidth=2.0, label="40% cred."),
    Line2D([0],[0], color="#d7191c", linestyle="-",  linewidth=2.2, label="60% cred."),
    Line2D([0],[0], color='navy',    linestyle='-',  linewidth=1.5, label="Eigendirs."),
    Line2D([0],[0], color='white', marker='x', linestyle='None',
           markersize=10, markeredgewidth=2.5,
           markeredgecolor='white', label="True min."),
]

for d, tag in [(sim,"sim"), (qpu,"qpu")]:
    fig, ax = plt.subplots(figsize=(7, 7))
    h2d = ax.hist2d(d['x1_exp'], d['x2_exp'], bins=32,
                    range=[[L1,U1],[L2,U2]], density=True, cmap="viridis")
    plt.colorbar(h2d[3], ax=ax, label="Density")
    add_credible_ellipses(ax)
    arrow_scale = 0.55
    for k in range(2):
        v = eigvecs_Q[:, k]
        length = arrow_scale / np.sqrt(eigvals_Q[k])
        for sign in [1, -1]:
            ax.arrow(f_star[0], f_star[1],
                     sign*length*v[0], sign*length*v[1],
                     width=0.012, head_width=0.045,
                     length_includes_head=True, color='navy')
    ax.scatter(f_star[0], f_star[1], color='white', marker='x', s=180,
               linewidths=2.5, zorder=5)
    ax.set_xlim(L1, U1); ax.set_ylim(L2, U2)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_aspect('equal')
    fig.legend(handles=legend_eigendirs, loc="lower center",
               bbox_to_anchor=(0.45, 0.0), ncol=3, frameon=True, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(f"{OUT}/scalar3x2_hist2d_eigendirs_{tag}.png", dpi=300,
                bbox_inches='tight')
    plt.close()

# ── Spectral gap (2 figures) ────────────────────────────────────────────────
n_qubits = 12; eps = 0.05
h_raw = qpu['h']; J_raw = qpu['J']; scale_val = qpu['scale']
h_s = h_raw.copy()
h_s[np.abs(h_s) < eps * scale_val] = 0.0
h_s /= scale_val
pairs = [(i,j, J_raw[i,j]/scale_val)
         for i in range(n_qubits)
         for j in range(i+1, n_qubits)
         if abs(J_raw[i,j]/scale_val) >= eps]

dim = 2**n_qubits
z_coeff = -0.5 * h_s.copy()
for i,j,Jij in pairs:
    z_coeff[i] += -0.25*Jij; z_coeff[j] += -0.25*Jij

diag_HC = np.zeros(dim)
for k in range(dim):
    bits = np.array([(k>>i)&1 for i in range(n_qubits)], dtype=float)
    z = 1.0 - 2.0*bits
    val = np.dot(z_coeff, z)
    for i,j,Jij in pairs: val += (Jij/4.0)*z[i]*z[j]
    diag_HC[k] = val

HB = lil_matrix((dim,dim), dtype=float)
for i in range(n_qubits):
    mask = 1<<i
    for k in range(dim): HB[k, k^mask] -= 1.0
HB = csr_matrix(HB)

n_s = 51; s_vals = np.linspace(0,1,n_s)
E0 = np.zeros(n_s); E1 = np.zeros(n_s); gaps = np.zeros(n_s)
for idx, s in enumerate(s_vals):
    HC_sp = sp_diags(s*diag_HC, 0, format='csr')
    H_s = (1.0-s)*HB + HC_sp
    vals = eigsh(H_s, k=2, which='SA', return_eigenvectors=False,
                 tol=1e-8, maxiter=10000)
    vals = np.sort(vals)
    E0[idx]=vals[0]; E1[idx]=vals[1]; gaps[idx]=vals[1]-vals[0]

i_min = np.argmin(gaps); Delta_min = gaps[i_min]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(s_vals, E0, color="#2166AC", linewidth=2.5, label=r"$E_0(s)$")
ax.plot(s_vals, E1, color="#D6604D", linewidth=2.5, label=r"$E_1(s)$")
ax.fill_between(s_vals, E0, E1, alpha=0.15, color="#92C5DE", label="Gap region")
ax.set_xlabel(r"Interpolation parameter $s$")
ax.set_ylabel("Energy")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/scalar3x2_spectral_gap_energies.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(s_vals, gaps, color="black", linewidth=2.5)
ax.axvline(s_vals[i_min], color="red", linestyle="--", linewidth=1.8,
           label=rf"$s^* = {s_vals[i_min]:.2f}$")
ax.axhline(Delta_min, color="red", linestyle=":", linewidth=1.8,
           label=rf"$\Delta_{{\min}} = {Delta_min:.4f}$")
ax.set_xlabel(r"Interpolation parameter $s$")
ax.set_ylabel(r"Spectral gap $E_1(s) - E_0(s)$")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/scalar3x2_spectral_gap_gap.png", dpi=300)
plt.close()

print("All figures saved.")
