import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# =========================================================
# User settings
# =========================================================
#log_dir = "gradient_alignment_logs"
log_dir = "/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/Inf-Net/gradient_alignment_logs"
dataset = "LungInfection"
model_name = "Inf_Net"
epsilon = 200.0
#epsilon = 200
retinal_layer_wise = False
methods = ["flat", "automatic", "normalized_sgd", "psac"]

save_pdf = True
show_plot = False

# deterministic selection of runs
random_seed_for_selection = 42

method_labels = {
    "flat": "Flat",
    "automatic": "Auto-S",
    "normalized_sgd": "NSGD",
    "psac": "PSAC",
}

# =========================================================
# Helpers
# =========================================================
def build_pattern(log_dir, dataset, model_name, method, epsilon):
    return os.path.join(
        log_dir,
        f"{dataset}_{model_name}_{method}_eps{epsilon}_seed*_morphology*_retinal_layer_wise_*.csv"
    )

def file_matches_config(filepath, morphology, retinal_layer_wise):
    fname = os.path.basename(filepath)
    morph_tag = f"morphology{morphology}"
    rlw_tag = f"retinal_layer_wise_{retinal_layer_wise}"
    return (morph_tag in fname) and (rlw_tag in fname)

def load_config_df(morphology_value):
    all_dfs = []

    for method in methods:
        pattern = build_pattern(log_dir, dataset, model_name, method, epsilon)
        files = sorted(glob.glob(pattern))

        method_dfs = []
        for f in files:
            if not file_matches_config(f, morphology_value, retinal_layer_wise):
                continue

            df = pd.read_csv(f)

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
            df["morphology"] = morphology_value
            df["source_file"] = os.path.basename(f)

            if len(df) > 0:
                method_dfs.append(df)

        if method_dfs:
            combined_method_df = pd.concat(method_dfs, ignore_index=True)
            all_dfs.append(combined_method_df)
            seeds_here = sorted(combined_method_df["seed"].unique()) if "seed" in combined_method_df.columns else []
            print(f"[INFO] Loaded {len(combined_method_df)} rows for method '{method}' | morphology={morphology_value} | seeds={seeds_here}")
        else:
            print(f"[INFO] No matching files found for method '{method}' | morphology={morphology_value}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def balance_runs_by_method(df, methods, rng_seed=42):
    """
    Balance the dataframe so each method has the same number of runs (unique seeds).
    Keeps all rows from the selected runs.
    """
    if df.empty:
        return df

    if "seed" not in df.columns:
        raise ValueError("The dataframe does not contain a 'seed' column, so runs cannot be balanced.")

    rng = np.random.default_rng(rng_seed)

    # collect available seeds per method
    seeds_per_method = {}
    for method in methods:
        seeds = sorted(df.loc[df["method"] == method, "seed"].dropna().unique().tolist())
        if len(seeds) > 0:
            seeds_per_method[method] = seeds

    if not seeds_per_method:
        return pd.DataFrame()

    print("\n[INFO] Available runs per method before balancing:")
    for method, seeds in seeds_per_method.items():
        print(f"  {method}: {len(seeds)} runs -> {seeds}")

    # use the minimum number of runs available across methods
    min_runs = min(len(seeds) for seeds in seeds_per_method.values())
    print(f"[INFO] Balancing to {min_runs} run(s) per method.")

    selected_parts = []
    selected_seed_map = {}

    for method in methods:
        if method not in seeds_per_method:
            continue

        available_seeds = np.array(seeds_per_method[method])

        if len(available_seeds) > min_runs:
            chosen = sorted(rng.choice(available_seeds, size=min_runs, replace=False).tolist())
        else:
            chosen = sorted(available_seeds.tolist())

        selected_seed_map[method] = chosen

        part = df[(df["method"] == method) & (df["seed"].isin(chosen))].copy()
        selected_parts.append(part)

    print("[INFO] Selected runs per method after balancing:")
    for method, seeds in selected_seed_map.items():
        print(f"  {method}: {seeds}")

    if not selected_parts:
        return pd.DataFrame()

    return pd.concat(selected_parts, ignore_index=True)

# =========================================================
# Load both No-Morph and Morph
# =========================================================
df_nomorph = load_config_df(False)
df_morph = load_config_df(True)

if df_nomorph.empty and df_morph.empty:
    raise ValueError("No matching CSV files found for either morphology=False or morphology=True.")

# =========================================================
# Balance runs across methods within each morphology condition
# =========================================================
df_nomorph_bal = balance_runs_by_method(df_nomorph, methods, rng_seed=random_seed_for_selection)
df_morph_bal = balance_runs_by_method(df_morph, methods, rng_seed=random_seed_for_selection)

# Optional: if you also want the same number of runs between Morph and No-Morph,
# you can balance both again to the smaller min_runs, but usually this is not necessary.

df_all = pd.concat([df_nomorph_bal, df_morph_bal], ignore_index=True)

print("\n[SUMMARY] Rows per method and morphology after balancing:")
print(df_all.groupby(["method", "morphology"]).size())

if "seed" in df_all.columns:
    print("\n[SUMMARY] Seeds per method and morphology after balancing:")
    print(df_all.groupby(["method", "morphology"])["seed"].unique())

# =========================================================
# Prepare violin data
# =========================================================
positions_nomorph = np.arange(len(methods)) - 0.18
positions_morph = np.arange(len(methods)) + 0.18

data_nomorph = []
data_morph = []

for method in methods:
    vals_nomorph = df_all.loc[
        (df_all["method"] == method) & (df_all["morphology"] == False), "cosine"
    ].dropna().values

    vals_morph = df_all.loc[
        (df_all["method"] == method) & (df_all["morphology"] == True), "cosine"
    ].dropna().values

    data_nomorph.append(vals_nomorph)
    data_morph.append(vals_morph)

# =========================================================
# Plot violin
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(5, 3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

valid_nomorph_positions = [p for p, d in zip(positions_nomorph, data_nomorph) if len(d) > 0]
valid_nomorph_data = [d for d in data_nomorph if len(d) > 0]

valid_morph_positions = [p for p, d in zip(positions_morph, data_morph) if len(d) > 0]
valid_morph_data = [d for d in data_morph if len(d) > 0]

if valid_nomorph_data:
    vp1 = ax.violinplot(
        valid_nomorph_data,
        positions=valid_nomorph_positions,
        widths=0.30,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )
    for body in vp1["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("black")
        body.set_alpha(0.9)
    vp1["cmedians"].set_color("black")
    vp1["cmedians"].set_linewidth(1.2)

if valid_morph_data:
    vp2 = ax.violinplot(
        valid_morph_data,
        positions=valid_morph_positions,
        widths=0.30,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )
    for body in vp2["bodies"]:
        body.set_facecolor("#1f77b4")
        body.set_edgecolor("black")
        body.set_alpha(0.9)
    vp2["cmedians"].set_color("black")
    vp2["cmedians"].set_linewidth(1.2)

ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels([method_labels[m] for m in methods])
ax.set_xlabel("Clipping strategy")
ax.set_ylabel("Cosine similarity")
ax.set_ylim(-1.0, 1.0)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

legend_handles = [
    Patch(facecolor="#9ecae1", edgecolor="black", label="No-Morph"),
    Patch(facecolor="#1f77b4", edgecolor="black", label="Morph"),
]
#ax.legend(handles=legend_handles, frameon=False, loc="best")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

out_name = (
    f"{dataset}_{model_name}_eps{epsilon}"
    f"_gradient_cosine_violin_balanced_runs.pdf"
)

if save_pdf:
    plt.savefig(out_name, bbox_inches="tight")
    print(f"\n[INFO] Saved figure to: {out_name}")

if show_plot:
    plt.show()
else:
    plt.close()