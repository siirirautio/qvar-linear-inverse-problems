import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 36,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 26,
})

OUT = "."

# Load results: choose simulated or Q50
files = {

#    2:   "samples_simu_T2_p2_eps0.04.json",
#    100: "samples_simu_T100_p100_eps0.04.json",
#    1000: "samples_simu_T1000_p1000_eps0.04.json",

    2:   "samples_q50_T2_p2_eps0.15.json",
    100: "samples_q50_T100_p100_eps0.15.json",
    200: "samples_q50_T200_p200_eps0.15.json",
}

for Tp, path in files.items():
    with open(path) as f:
        data = json.load(f)

    L      = data["run_info"]["L"]
    U      = data["run_info"]["U"]
    N      = data["run_info"]["N"]
    x_star = data["ground_truth"]["x_star"]
    x_map  = data["map_estimate"]["x"]
    x_mean = data["mean_estimate"]["x"]

    x_exp = np.array([s["x"] for s in data["samples"]
                      for _ in range(s["count"])])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(x_exp, bins=N, density=True, range=(L, U),
            color="#4C72B0", alpha=0.85)
    ax.axvline(x_star, color="black",  linestyle="-",  linewidth=3.0)
    ax.axvline(x_map,  color="red",    linestyle="--", linewidth=3.0)
    ax.axvline(x_mean, color="orange", linestyle=":",  linewidth=3.0)
    ax.set_xlim(L, U)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    leg_handles = [
        Line2D([0],[0], color="black",  linestyle="-",  linewidth=3.0,
               label="True"),
        Line2D([0],[0], color="red",    linestyle="--", linewidth=3.0,
               label="MAP"),
        Line2D([0],[0], color="orange", linestyle=":",  linewidth=3.0,
               label="Mean"),
    ]
    fig.legend(handles=leg_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=True, fontsize=26, borderaxespad=0.2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(f"{OUT}/scalar1d_histogram_T{Tp}_p{Tp}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scalar1d_histogram_T{Tp}_p{Tp}.png")
    print(f"T=p={Tp}: x_star={x_star:.5f}, x_mean={x_mean:.5f}, x_MAP={x_map:.5f}")

print("All done.") 
