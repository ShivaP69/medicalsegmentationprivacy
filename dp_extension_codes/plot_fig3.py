import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_dir = "gradient_alignment_logs"
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================
log_dir = "gradient_alignment_logs"
#log_dir="/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/Inf-Net/gradient_alignment_logs"
dataset = "UMN"
model_name = "NestedUNet"
#epsilon = 200.0 # for Covid
epsilon=200
common_seeds = [10] # double check # for Covid
#common_seeds = [11,12] # double check
# exact config to plot
morphology = True
retinal_layer_wise = False # when this is true the morphology also must be true

methods = ["flat", "automatic", "normalized_sgd", "psac"]

# plotting style
use_density = True   # True -> paper-style density histogram
                   # False -> true per-bin probabilities

save_pdf = True
show_plot = False


# =========================================================
# Helper: build filename pattern
# =========================================================
def build_pattern(log_dir, dataset, model_name, method, epsilon):
    return os.path.join(
        log_dir,
        f"{dataset}_{model_name}_{method}_eps{epsilon}_seed*_morphology*_retinal_layer_wise_*.csv"
    )


# =========================================================
# Helper: filter files by morphology and retinal_layer_wise
# =========================================================
def file_matches_config(filepath, morphology, retinal_layer_wise):
    fname = os.path.basename(filepath)
    morph_tag = f"morphology{morphology}"
    rlw_tag = f"retinal_layer_wise_{retinal_layer_wise}"
    ok_morph = morph_tag in fname
    ok_rlw = rlw_tag in fname
    print(fname, "->", ok_morph, ok_rlw)
    return (morph_tag in fname) and (rlw_tag in fname)


# =========================================================
# Load matching CSV files
# =========================================================
all_dfs = []

for method in methods:
    pattern = build_pattern(log_dir, dataset, model_name, method, epsilon)
    print(pattern)
    files = sorted(glob.glob(pattern))

    method_dfs = []
    for f in files:
        if not file_matches_config(f, morphology, retinal_layer_wise):
            continue

        df = pd.read_csv(f)

        # safety filters from content too
        if "dataset" in df.columns:
            df = df[df["dataset"] == dataset]
        if "model_name" in df.columns:
            df = df[df["model_name"] == model_name]
        if "clipping" in df.columns:
            df = df[df["clipping"] == method]

        if "cosine" not in df.columns:
            print(f"[WARNING] No 'cosine' column in {f}, skipping.")
            continue

        df = df.dropna(subset=["cosine"]).copy()
        df["method"] = method
        df["source_file"] = os.path.basename(f)

        if len(df) > 0:
            method_dfs.append(df)

    if method_dfs:
        combined_method_df = pd.concat(method_dfs, ignore_index=True)
        all_dfs.append(combined_method_df)
        print(f"[INFO] Loaded {len(combined_method_df)} rows for method '{method}' from {len(method_dfs)} file(s).")
    else:
        print(f"[INFO] No matching files found for method '{method}'.")


if not all_dfs:
    raise ValueError("No matching CSV files found for the requested configuration.")

df_all = pd.concat(all_dfs, ignore_index=True)

print("\n[SUMMARY] Loaded rows per method:")
print(df_all["method"].value_counts())

if "seed" in df_all.columns:
    print("\n[SUMMARY] Unique seeds per method:")
    print(df_all.groupby("method")["seed"].nunique())

print(df_all.groupby("method")["seed"].unique())

df_all = df_all[df_all["seed"].isin(common_seeds)]

# =========================================================
# Plot
# =========================================================
"""bins = np.linspace(-1.0, 1.0, 41)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_width = bins[1] - bins[0]

plt.figure(figsize=(7, 4.5))

for method in methods:
    vals = df_all.loc[df_all["method"] == method, "cosine"].values
    if len(vals) == 0:
        continue

    if use_density:
        # Figure-3-like style
        plt.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.4,
            label=method
        )
    else:
        # true per-bin probability
        counts, _ = np.histogram(vals, bins=bins)
        prob = counts / counts.sum()

        plt.plot(
            bin_centers,
            prob,
            marker="o",
            linewidth=2,
            label=method
        )

plt.xlabel("Cosine similarity")
plt.ylabel("Density" if use_density else "Probability")
plt.xlim(-1.0, 1.0)

title = (
    f"{dataset} | {model_name} | eps={epsilon} | "
    f"morphology={morphology} | retinal_layer_wise={retinal_layer_wise}"
)
plt.title(title, fontsize=10)

plt.legend(frameon=False)
plt.tight_layout()"""

# =========================================================
# Plot: one row (1x4), one subplot per method
# =========================================================
# =========================================================
# Plot: density (paper-style, Figure 3-like)
# =========================================================
use_density = True          # paper uses density (not probability)
bins = np.linspace(-1.0, 1.0, 31)   # paper uses ~30 bins


fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), sharex=True, sharey=True)
fig.patch.set_facecolor("white")

color_map = {
    "flat": "#1f77b4",
    "automatic": "#ff7f0e",
    "normalized_sgd": "#2ca02c",
    "psac": "#d62728",
}
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
      })
method_labels = {
    "flat": "Flat",
    "automatic": "Auto-S",
    "normalized_sgd": "NSGD",
    "psac": "PSAC",
}
for ax, method in zip(axes, methods):
    ax.set_facecolor("white")
    vals = df_all.loc[df_all["method"] == method, "cosine"].values

    if len(vals) == 0:
        ax.set_title(f"{method}\n(no data)")
        ax.set_xlim(-1.0, 1.0)
        continue

    weights = np.ones_like(vals) / len(vals)

    ax.hist(
        vals,
        bins=bins,
        weights=weights,
        alpha=0.75,
        color=color_map[method],
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title(method_labels.get(method, method))
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])

axes[0].set_ylabel("Probability", fontsize=13)
axes[0].set_yticks([0,0.2,0.4,0.6,0.8,1])
#axes[0].set_ylabel("Density",fontsize=13)
for ax in axes:
    ax.set_xlabel("Cosine similarity",fontsize=13)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

plt.tight_layout()

out_name = (
    f"{dataset}_{model_name}_eps{epsilon}"
    f"_morphology{morphology}_retinal_layer_wise_{retinal_layer_wise}"
    f"_gradient_cosine_density_row.pdf"
)

if save_pdf:
    plt.savefig(out_name, bbox_inches="tight")
    print(f"\n[INFO] Saved figure to: {out_name}")

plt.close()