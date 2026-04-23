import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# =========================================================
# CONFIG
# =========================================================
model = "LFUNet"
conditional = "None"
dataset = "Duke"

base_non_dpsgd = f"/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/morph_debug_events/{dataset}/{model}/cond_{conditional}/rlw_True"
base_dpsgd     = f"/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/morph_debug_events/{dataset}/{model}_DPSGD/cond_None/rlw_True"

policy_pattern_non_dpsgd = os.path.join(base_non_dpsgd, "train_policy_debug_seed_clipping*_seed*.csv")
policy_pattern_dpsgd     = os.path.join(base_dpsgd,     "train_policy_debug_seed_clipping*_seed*.csv")

results_csv = "/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/csv_results/test_oct_results_UMN_Duke_lr_max_gradient.csv"

out_dir = f"./train_policy_grid_{dataset}_{model}"
os.makedirs(out_dir, exist_ok=True)

layer_ids = [1, 2, 3, 4, 5, 6, 7]
layer_labels = {
    1: "Layer 1",
    2: "Layer 2",
    3: "Layer 3",
    4: "Layer 4",
    5: "Layer 5",
    6: "Layer 6",
    7: "Layer 7",
}

# if you want anatomical labels, replace with:
# layer_labels = {1:"RNFL",2:"GCL",3:"IPL",4:"INL",5:"OPL",6:"ONL",7:"RPE"}

clipping_name_map = {
    "None": "Non-private",
    "automatic": "Auto-S",
    "flat": "Flat",
    "normalized_sgd": "NSGD",
    "psac": "PSAC",
}

op_colors = {
    "open":  "#1f77b4",
    "close": "#e53935",
    "both":  "#3cb44b",
}

policy_preferred_order = ["v1", "v2", "v3"]
clip_preferred_order = ["None", "automatic", "flat", "normalized_sgd", "psac"]

min_marker_size = 18
max_marker_size = 120

selection_metric = "Dice"   # "Dice", "HD95", "MAE"


# =========================================================
# HELPERS
# =========================================================
def parse_clipping_from_filename(path):
    fname = os.path.basename(path)

    # works for:
    # train_policy_debug_seed_clipping_none_seed5.csv
    # train_policy_debug_seed_clipping_flat_seed5.csv
    # train_policy_debug_seed_clipping_normalized_sgd_seed5.csv
    m = re.search(r"clipping_([^_]+(?:_[^_]+)*)_seed(\d+)\.csv$", fname)
    if not m:
        raise ValueError(f"Could not parse clipping from filename: {fname}")

    clip = m.group(1).strip().lower()
    if clip in ["none", "non-private", "non_private", "nan", ""]:
        return "None"
    if clip == "nsgd":
        return "normalized_sgd"
    return clip

def parse_seed_from_filename(path):
    fname = os.path.basename(path)
    m = re.search(r"_seed(\d+)\.csv$", fname)
    if not m:
        raise ValueError(f"Could not parse seed from filename: {fname}")
    return int(m.group(1))

def normalize_policy(x):
    if pd.isna(x):
        return "None"
    x = str(x).strip()
    return x if x else "None"

def normalize_clipping(x):
    if pd.isna(x):
        return "None"
    x = str(x).strip().lower()
    if x in ["none", "non-private", "non_private", "nan", ""]:
        return "None"
    if x == "nsgd":
        return "normalized_sgd"
    return x

def safe_mode(series):
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return "unknown"
    c = Counter(vals)
    max_count = max(c.values())
    winners = sorted([k for k, v in c.items() if v == max_count])
    return winners[0]

def choose_metric_columns(df):
    candidates = {
        "Dice": ["Validation_Dice", "Dice", "dice", "val_dice"],
        "HD95": ["HD95", "hd95", "Validation_HD95", "val_hd95", "best_hd95"],
        "MAE":  ["mae", "MAE", "Validation_MAE", "val_mae"],
    }
    found = {}
    for key, opts in candidates.items():
        for c in opts:
            if c in df.columns:
                found[key] = c
                break
    return found

def infer_policy_from_results_row(row):
    for col in ["policy_type", "Policy_Type", "policy"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                return val
    return "None"

def parse_np_array_string(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    arr = np.fromstring(s.strip("[]"), sep=" ")
    return arr if arr.size > 0 else None

def add_marker_sizes(df):
    out = df.copy()
    th = out["thickness"].astype(float).values

    if len(th) == 0:
        out["marker_size"] = min_marker_size
        return out

    tmin, tmax = np.nanmin(th), np.nanmax(th)
    if np.isclose(tmin, tmax):
        out["marker_size"] = (min_marker_size + max_marker_size) / 2
    else:
        scaled = (th - tmin) / (tmax - tmin)
        out["marker_size"] = min_marker_size + (max_marker_size - min_marker_size) * np.sqrt(scaled)

    return out

def load_and_reshape_one(path, source):
    df = pd.read_csv(path)

    clipping = parse_clipping_from_filename(path)
    file_seed = parse_seed_from_filename(path)

    if "seed" not in df.columns:
        df["seed"] = file_seed
    else:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(file_seed).astype(int)

    if "policy_type" not in df.columns:
        df["policy_type"] = "None"

    df["policy_type"] = df["policy_type"].map(normalize_policy)
    df["clipping"] = clipping
    df["source"] = source
    df["file_path"] = path

    rows = []
    for _, row in df.iterrows():
        epoch = pd.to_numeric(row.get("epoch", np.nan), errors="coerce")
        if pd.isna(epoch):
            continue

        seed = int(row["seed"])
        policy = normalize_policy(row.get("policy_type", "None"))
        clipping = normalize_clipping(row["clipping"])

        for c in layer_ids:
            tcol = f"thickness_mean_c{c}"
            ocol = f"op_c{c}"

            if tcol not in df.columns or ocol not in df.columns:
                continue

            thickness = pd.to_numeric(row.get(tcol, np.nan), errors="coerce")
            operation = str(row.get(ocol, "unknown")).strip().lower()

            if pd.isna(thickness):
                continue

            if operation not in op_colors:
                continue

            rows.append({
                "epoch": int(epoch),
                "seed": seed,
                "layer_id": c,
                "layer_label": layer_labels[c],
                "thickness": float(thickness),
                "operation": operation,
                "policy_type": policy,
                "clipping": clipping,
                "source": source,
                "file_path": path,
            })

    return pd.DataFrame(rows)

def aggregate_epoch_level(long_df):
    return (
        long_df
        .groupby(
            ["clipping", "policy_type", "seed", "epoch", "layer_id", "layer_label"],
            as_index=False
        )
        .agg(
            thickness=("thickness", "mean"),
            operation=("operation", safe_mode),
        )
    )


# =========================================================
# LOAD RESULTS CSV AND PICK BEST SEEDS
# =========================================================
res = pd.read_csv(results_csv).copy()

metric_cols = choose_metric_columns(res)
if selection_metric not in metric_cols:
    raise ValueError(
        f"Could not find metric column for {selection_metric}. "
        f"Available mappings: {metric_cols}"
    )

metric_col = metric_cols[selection_metric]

if "dataset" in res.columns:
    res = res[res["dataset"].astype(str) == dataset].copy()

if "seed" not in res.columns:
    raise ValueError("results_csv must contain a 'seed' column")

if "Model_Name" in res.columns:
    model_col = "Model_Name"
elif "model_name" in res.columns:
    model_col = "model_name"
else:
    raise ValueError("Could not find model name column in results_csv")

res["seed"] = pd.to_numeric(res["seed"], errors="coerce")
res = res.dropna(subset=["seed"])
res["seed"] = res["seed"].astype(int)

res[metric_col] = pd.to_numeric(res[metric_col], errors="coerce")
res = res.dropna(subset=[metric_col])

res["policy_type_norm"] = res.apply(infer_policy_from_results_row, axis=1).map(normalize_policy)

if "clipping_strategy" in res.columns:
    res["clipping_norm"] = res["clipping_strategy"].map(normalize_clipping)
else:
    res["clipping_norm"] = np.where(
        res[model_col].astype(str).str.contains("DPSGD", case=False, na=False),
        "unknown",
        "None"
    )

# restrict to morphology-enabled RLW=True rows for this comparison
if "retinal_layer_wise" in res.columns:
    res = res[res["retinal_layer_wise"] == True].copy()
if "initial_use_morph" in res.columns:
    res = res[res["initial_use_morph"] == True].copy()
if "final_use_morph" in res.columns:
    res = res[res["final_use_morph"] == True].copy()

# exact model names in your results csv for this experiment
model_non_dp = f"{model}_FixedR3"
model_dp     = f"{model}_DPSGD_FixedR3"

res_non = res[res[model_col].astype(str) == model_non_dp].copy()
res_dp  = res[res[model_col].astype(str) == model_dp].copy()

if len(res_non) > 0:
    res_non["clipping_norm"] = "None"

ascending = True if selection_metric in ["HD95", "MAE"] else False
files_non = sorted(glob.glob(policy_pattern_non_dpsgd))
files_dp  = sorted(glob.glob(policy_pattern_dpsgd))

all_dfs = []

for f in files_non:
    try:
        all_dfs.append(load_and_reshape_one(f, source="non_private"))
    except Exception as e:
        print(f"[WARN] Failed reading {f}: {e}")

for f in files_dp:
    try:
        all_dfs.append(load_and_reshape_one(f, source="dpsgd"))
    except Exception as e:
        print(f"[WARN] Failed reading {f}: {e}")

if not all_dfs:
    raise RuntimeError("No train debug files found")

long_df = pd.concat(all_dfs, ignore_index=True)
epoch_df = aggregate_epoch_level(long_df)

available_debug_seeds = (
    epoch_df[["clipping", "policy_type", "seed"]]
    .drop_duplicates()
    .copy()
)
res_non = res_non.merge(
    available_debug_seeds,
    left_on=["clipping_norm", "policy_type_norm", "seed"],
    right_on=["clipping", "policy_type", "seed"],
    how="inner"
)

res_dp = res_dp.merge(
    available_debug_seeds,
    left_on=["clipping_norm", "policy_type_norm", "seed"],
    right_on=["clipping", "policy_type", "seed"],
    how="inner"
)

for df_ in [res_non, res_dp]:
    for col in ["clipping", "policy_type"]:
        if col in df_.columns:
            df_.drop(columns=[col], inplace=True)
# keep only best seeds
best_non = (
    res_non
    .sort_values(metric_col, ascending=ascending)
    .groupby(["clipping_norm", "policy_type_norm"], as_index=False)
    .first()[["clipping_norm", "policy_type_norm", "seed", metric_col]]
)

best_dp = (
    res_dp
    .sort_values(metric_col, ascending=ascending)
    .groupby(["clipping_norm", "policy_type_norm"], as_index=False)
    .first()[["clipping_norm", "policy_type_norm", "seed", metric_col]]
)

best_seeds = pd.concat([best_non, best_dp], ignore_index=True)
best_seeds = best_seeds.rename(columns={
    "clipping_norm": "clipping",
    "policy_type_norm": "policy_type",
    metric_col: "selection_score",
})

best_seeds["clipping"] = best_seeds["clipping"].map(normalize_clipping)
best_seeds["policy_type"] = best_seeds["policy_type"].map(normalize_policy)

print("\n[DEBUG] available debug seeds:")
print(
    available_debug_seeds
    .sort_values(["policy_type", "clipping", "seed"])
    .to_string(index=False)
)

print("\n[DEBUG] best seeds after restricting to debug files:")
print(
    best_seeds
    .sort_values(["policy_type", "clipping"])
    .to_string(index=False)
)

best_seeds.to_csv(os.path.join(out_dir, "best_seeds_per_setting.csv"), index=False)


# =========================================================
# LOAD TRAIN DEBUG FILES
# =========================================================

epoch_best = epoch_df.merge(
    best_seeds[["clipping", "policy_type", "seed"]],
    on=["clipping", "policy_type", "seed"],
    how="inner"
).copy()

if epoch_best.empty:
    raise RuntimeError("No matched training debug rows after filtering to best seeds")
print("\n[DEBUG] epoch_best combos:")
print(
    epoch_best[["clipping", "policy_type", "seed"]]
    .drop_duplicates()
    .sort_values(["policy_type", "clipping", "seed"])
    .to_string(index=False)
)
epoch_best.to_csv(os.path.join(out_dir, "epoch_summary_best_seeds_only.csv"), index=False)


# =========================================================
# PLOT: ONE GRID FOR ALL CLIPPINGS
# =========================================================
def make_policy_grid(df_plot, save_base, epoch_step=5):
    sub = df_plot.copy()
    if sub.empty:
        raise ValueError("Empty dataframe for plotting")

    sub = add_marker_sizes(sub)

    policy_order = [p for p in ["v1", "v2", "v3"] if p in sub["policy_type"].astype(str).unique()]
    clip_order = [c for c in ["None", "flat", "automatic", "normalized_sgd", "psac"]
                  if c in sub["clipping"].astype(str).unique()]

    nrows = len(policy_order)
    ncols = len(clip_order)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.8 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    fig.patch.set_facecolor("white")
    y_order = layer_ids
    y_map = {lid: i for i, lid in enumerate(y_order)}

    for r, policy_type in enumerate(policy_order):
        for c, clipping in enumerate(clip_order):
            ax = axes[r, c]

            cell = sub[
                (sub["policy_type"] == policy_type) &
                (sub["clipping"] == clipping)
            ].copy()

            if cell.empty:
                ax.axis("off")
                continue

            # keep only every N epochs
            cell = cell[(cell["epoch"] == 1) | (cell["epoch"] % epoch_step == 0)].copy()
            cell["y"] = cell["layer_id"].map(y_map)

            # faint row guides
            for lid in y_order:
                y = y_map[lid]
                ax.hlines(
                    y,
                    xmin=cell["epoch"].min(),
                    xmax=cell["epoch"].max(),
                    colors="lightgray",
                    linewidth=0.6,
                    alpha=0.4,
                    zorder=1
                )

            for op_name, color in op_colors.items():
                cur = cell[cell["operation"] == op_name]
                if cur.empty:
                    continue

                ax.scatter(
                    cur["epoch"],
                    cur["y"],
                    s=cur["marker_size"],
                    c=color,
                    alpha=0.78,
                    edgecolors="black",
                    linewidths=0.25,
                    zorder=3
                )

            ax.set_yticks(range(len(y_order)))
            ax.set_yticklabels([layer_labels[l] for l in y_order], fontsize=14)
            ax.invert_yaxis()
            #ax.grid(axis="x", linestyle="--", alpha=0.18)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=14)

            if r == 0:
                ax.set_title(clipping_name_map.get(clipping, clipping), fontsize=14, pad=6)

            if c == 0:
                ax.set_ylabel("Layers", fontsize=14)
                ax.text(
                    -0.38, 0.5,
                    policy_type.upper(),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    rotation=90,
                    fontweight="bold"
                )

            if r == nrows - 1:
                ax.set_xlabel("Epoch", fontsize=14)

    handles_op = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markeredgecolor='black',
               markersize=8, label=op.capitalize())
        for op, color in op_colors.items()
    ]

    fig.legend(
        handles=handles_op,
        loc="lower center",
        bbox_to_anchor=(0.34, -0.02),
        ncol=3,
        frameon=False,
        title="Operation",
        fontsize=14,
        title_fontsize=14
    )

    thvals = sub["thickness"].dropna().astype(float).values
    if len(thvals) > 0:
        qs = np.quantile(thvals, [0.25, 0.5, 0.75])
        qs = np.unique(np.round(qs, 2))
        temp = add_marker_sizes(pd.DataFrame({"thickness": qs}))

        size_handles = [
            plt.scatter([], [], s=s, c="lightgray", edgecolors="black", alpha=0.85)
            for s in temp["marker_size"].values
        ]

        fig.legend(
            handles=size_handles,
            labels=[str(q) for q in qs],
            loc="lower center",
            bbox_to_anchor=(0.78, -0.02),
            ncol=min(3, len(qs)),
            frameon=False,
            title="Thickness",
            fontsize=14,
            title_fontsize=14
        )

    """fig.suptitle(
        f"{dataset}, {model}: layer-wise policy evolution during training (best seed per setting)",
        y=0.99,
        fontsize=14
    )"""

    plt.tight_layout(rect=[0.03, 0.07, 1.00, 0.95])
    plt.savefig(f"{save_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_base}.pdf")
    print(f"Saved: {save_base}.png")


save_base = os.path.join(out_dir, f"{dataset}_{model}_train_policy_grid_bestseed")
make_policy_grid(epoch_best, save_base)