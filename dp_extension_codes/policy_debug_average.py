import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter

# =========================================
# CONFIG
# =========================================
model = "LFUNet"
dataset = "Duke"

results_csv = "../../csv_results/test_oct_results_UMN_Duke_lr_max_gradient.csv"

base_non_dpsgd = (
    f"/scicore/home/wagner0024/parsar0000/miniconda3/"
    f"2025_shiva_dp_morph_extension/morph_debug_events/{dataset}/{model}/cond_None/rlw_True"
)
base_dpsgd = (
    f"/scicore/home/wagner0024/parsar0000/miniconda3/"
    f"2025_shiva_dp_morph_extension/morph_debug_events/{dataset}/{model}_DPSGD/cond_None/rlw_True"
)

policy_pattern_non_dpsgd = os.path.join(
    base_non_dpsgd, "test_policy_debug_seed_clipping*_seed*.csv"
)
policy_pattern_dpsgd = os.path.join(
    base_dpsgd, "test_policy_debug_seed_clipping*_seed*.csv"
)

layers = list(range(1, 8))
main_metric = "Validation_Dice"

MODEL_NON_DP = f"{model}_FixedR3"
MODEL_DP = f"{model}_DPSGD_FixedR3"
MODEL_NON_DP_NOMORPH = f"{model}_NoMorph"
MODEL_DP_NOMORPH = f"{model}_DPSGD_NoMorph"

preferred_policy_order = ["v1", "v2", "v3"]
preferred_clip_order = ["None", "flat", "automatic", "normalized_sgd", "psac"]

label_map = {
    "None": "Non-private",
    "automatic": "Auto-S",
    "normalized_sgd": "NSGD",
    "flat": "Flat",
    "psac": "PSAC",
}

color_map = {
    "open": "#1f77b4",
    "close": "#d62728",
    "both": "#2ca02c",
    "none": "#7f7f7f",
}

setting_cols = [
    "dataset",
    "Model_Name",
    "Batch_Size",
    "Operation",
    "Kernel",
    "conditional_morph",
    "conditional_point",
    "retinal_layer_wise",
    "clipping_strategy",
    "max_grad_norm",
    "Privacy_Epsilons",
    "policy_type",
]

# =========================================
# HELPERS
# =========================================
def sample_k_per_setting(df, setting_cols, k=3, random_state=42):
    if df.empty:
        return df.copy()

    cols = [c for c in setting_cols if c in df.columns]
    return (
        df.groupby(cols, dropna=False, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), k), random_state=random_state))
          .reset_index(drop=True)
    )

def get_best_setting_runs_sample_first(df, metric_col, setting_cols, k=3, random_state=42):
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=[metric_col]).copy()
    if df.empty:
        return pd.DataFrame()

    # 1) sample first
    df_sampled = sample_k_per_setting(df, setting_cols, k=k, random_state=random_state)

    # 2) aggregate
    df_sampled = add_setting_id(df_sampled, setting_cols)
    setting_summary = summarize_settings(df_sampled, metric_col)

    if setting_summary.empty:
        return pd.DataFrame()

    # 3) select best
    best_setting_id = setting_summary.sort_values(
        "metric_mean", ascending=False
    ).iloc[0]["_setting_id"]

    return df_sampled[df_sampled["_setting_id"] == best_setting_id].copy()

def sample_runs_to_n(df_runs, n):
    if df_runs is None or df_runs.empty or n <= 0:
        return pd.DataFrame()

    df_runs = df_runs.copy()

    # deterministic
    if "seed" in df_runs.columns:
        df_runs["seed"] = pd.to_numeric(df_runs["seed"], errors="coerce")
        df_runs = df_runs.sort_values("seed", na_position="last")

    return df_runs.head(n).copy()

def get_best_single_run(df, metric_col):
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=[metric_col]).copy()
    if df.empty:
        return pd.DataFrame()

    best_idx = df[metric_col].idxmax()
    return df.loc[[best_idx]].copy()

def normalize_clipping(x):
    if pd.isna(x):
        return "None"
    x = str(x).strip()
    if x == "" or x.lower() in ["nan", "none"]:
        return "None"
    return x

def parse_np_array_string(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    arr = np.fromstring(s.strip("[]"), sep=" ")
    return arr if arr.size > 0 else None

def parse_clipping_from_filename(path: str) -> str:
    m = re.search(r"clipping(.+?)_seed\d+\.csv$", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse clipping from filename: {path}")
    return normalize_clipping(m.group(1))

def parse_seed_from_filename(path: str) -> int:
    m = re.search(r"_seed(\d+)\.csv$", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse seed from filename: {path}")
    return int(m.group(1))

def clean_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    if "clipping_strategy" in df.columns:
        df["clipping_strategy_norm"] = df["clipping_strategy"].map(normalize_clipping)
    else:
        df["clipping_strategy_norm"] = "None"

    if "Operation" in df.columns:
        df["Operation_norm"] = df["Operation"].astype(str).str.strip().str.lower()
    else:
        df["Operation_norm"] = ""

    for col in [main_metric, "Batch_Size", "Kernel", "max_grad_norm", "Privacy_Epsilons", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def valid_dice_row(row):
    arr = parse_np_array_string(row["dice_all"])
    return arr is not None and len(arr) >= 8

def add_setting_id(df, cols):
    cols = [c for c in cols if c in df.columns]
    out = df.copy()
    out["_setting_id"] = out[cols].astype(str).agg(" | ".join, axis=1)
    return out

def summarize_settings(df, metric_col):
    grp = (
        df.groupby("_setting_id", dropna=False)
        .agg(
            metric_mean=(metric_col, "mean"),
            metric_std=(metric_col, "std"),
            n_runs=(metric_col, "count"),
        )
        .reset_index()
    )
    return grp

def get_best_setting_runs(df, metric_col):
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=[metric_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df = add_setting_id(df, setting_cols)
    setting_summary = summarize_settings(df, metric_col)

    if setting_summary.empty:
        return pd.DataFrame()

    best_setting_id = setting_summary.sort_values(
        "metric_mean", ascending=False
    ).iloc[0]["_setting_id"]

    return df[df["_setting_id"] == best_setting_id].copy()

def layerwise_mean_std_from_runs(df_runs, layers_):
    vals = []
    for _, row in df_runs.iterrows():
        arr = parse_np_array_string(row["dice_all"])
        if arr is None or len(arr) < 8:
            continue
        vals.append([arr[c] for c in layers_])

    if len(vals) == 0:
        return None, None, 0

    vals = np.array(vals, dtype=float)
    return vals.mean(axis=0), vals.std(axis=0), len(vals)

def get_last_policy_rows_per_type(df):
    sort_cols = [c for c in ["epoch", "batch_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.groupby("policy_type", dropna=False, as_index=False).tail(1)

def safe_majority(values, default="none"):
    values = [str(v) for v in values if pd.notna(v)]
    if len(values) == 0:
        return default
    return Counter(values).most_common(1)[0][0]

def get_matching_policy_rows(best_runs_df, policy_df, policy_type, clipping):
    if best_runs_df.empty or policy_df.empty:
        return pd.DataFrame()

    seeds = (
        pd.to_numeric(best_runs_df["seed"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    sub = policy_df[
        (policy_df["seed"].isin(seeds)) &
        (policy_df["policy_type"] == policy_type) &
        (policy_df["clipping"] == normalize_clipping(clipping))
    ].copy()

    return sub

def filter_results_to_debug_available_runs(results_sub, policy_df, policy_type, clipping):
    """
    Keep only RLW result rows that have matching debug rows.
    Matching is done by seed, because debug files exist per run.
    """
    if results_sub.empty or policy_df.empty:
        return pd.DataFrame()

    debug_sub = policy_df[
        (policy_df["policy_type"] == policy_type) &
        (policy_df["clipping"] == normalize_clipping(clipping))
    ].copy()

    if debug_sub.empty:
        return pd.DataFrame()

    debug_seeds = set(pd.to_numeric(debug_sub["seed"], errors="coerce").dropna().astype(int).tolist())

    out = results_sub.copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out = out.dropna(subset=["seed"])
    out["seed"] = out["seed"].astype(int)

    out = out[out["seed"].isin(debug_seeds)].copy()
    return out
# =========================================
# LOAD RESULTS CSV
# =========================================
results_df = pd.read_csv(results_csv)
results_df = clean_df(results_df)
results_df = results_df[results_df["dataset"] == dataset].copy()
results_df = results_df.dropna(subset=[main_metric]).copy()
results_df = results_df[results_df.apply(valid_dice_row, axis=1)].copy()

print(f"[INFO] results rows after filtering: {len(results_df)}")

# =========================================
# LOAD POLICY DEBUG CSVs
# =========================================
policy_rows = []

for path in sorted(glob.glob(policy_pattern_non_dpsgd)):
    pdf = pd.read_csv(path)
    if pdf.empty:
        continue
    last_rows = get_last_policy_rows_per_type(pdf)
    for _, row in last_rows.iterrows():
        row = row.copy()
        row["seed"] = parse_seed_from_filename(path)
        row["clipping"] = parse_clipping_from_filename(path)
        row["source_file"] = os.path.basename(path)
        policy_rows.append(row)

for path in sorted(glob.glob(policy_pattern_dpsgd)):
    pdf = pd.read_csv(path)
    if pdf.empty:
        continue
    last_rows = get_last_policy_rows_per_type(pdf)
    for _, row in last_rows.iterrows():
        row = row.copy()
        row["seed"] = parse_seed_from_filename(path)
        row["clipping"] = parse_clipping_from_filename(path)
        row["source_file"] = os.path.basename(path)
        policy_rows.append(row)

policy_df = pd.DataFrame(policy_rows)

if not policy_df.empty:
    policy_df = policy_df[
        (policy_df["dataset"] == dataset) &
        (policy_df["model_name"].astype(str).str.contains(model, na=False))
    ].copy()

    policy_df["seed"] = pd.to_numeric(policy_df["seed"], errors="coerce")
    policy_df = policy_df.dropna(subset=["seed"])
    policy_df["seed"] = policy_df["seed"].astype(int)
    policy_df["clipping"] = policy_df["clipping"].map(normalize_clipping)

print(f"[INFO] policy debug rows after filtering: {len(policy_df)}")

# =========================================
# MAIN COLLECTION
# =========================================
records = []

for policy_type in preferred_policy_order:
    for clipping in preferred_clip_order:

        # -------------------------
        # RLW=True adaptive
        # -------------------------
        if clipping == "None":
            rlw_candidates = results_df[
                (results_df["Model_Name"] == MODEL_NON_DP) &
                (results_df["retinal_layer_wise"] == True) &
                (results_df["policy_type"] == policy_type)
            ].copy()
        else:
            rlw_candidates = results_df[
                (results_df["Model_Name"] == MODEL_DP) &
                (results_df["retinal_layer_wise"] == True) &
                (results_df["policy_type"] == policy_type) &
                (results_df["clipping_strategy_norm"] == clipping)
            ].copy()

        rlw_candidates_debug_ok = filter_results_to_debug_available_runs(
            rlw_candidates, policy_df, policy_type, clipping
        )

        best_rlw_runs = get_best_single_run(rlw_candidates_debug_ok, main_metric)
        if best_rlw_runs.empty:
            print(f"[WARN] no RLW runs for policy={policy_type}, clipping={clipping}")
            continue

        rlw_mean, rlw_std, rlw_n = layerwise_mean_std_from_runs(best_rlw_runs, layers)
        if rlw_mean is None:
            print(f"[WARN] RLW dice parse failed for policy={policy_type}, clipping={clipping}")
            continue

        # thickness + operations from debug files
        matched_policy_rows = get_matching_policy_rows(
            best_rlw_runs, policy_df, policy_type, clipping
        )

        if matched_policy_rows.empty:
            print(f"[WARN] no matching policy debug rows for policy={policy_type}, clipping={clipping}")

        thickness_vals = []
        ops_vals = []

        for c in layers:
            tcol = f"thickness_mean_c{c}"
            ocol = f"op_c{c}"

            if not matched_policy_rows.empty and tcol in matched_policy_rows.columns:
                tvals = pd.to_numeric(matched_policy_rows[tcol], errors="coerce").dropna()
                thickness_vals.append(tvals.mean() if len(tvals) > 0 else np.nan)
            else:
                thickness_vals.append(np.nan)

            if not matched_policy_rows.empty and ocol in matched_policy_rows.columns:
                ovals = matched_policy_rows[ocol].dropna().astype(str).tolist()
                ops_vals.append(safe_majority(ovals, default="none"))
            else:
                ops_vals.append("none")


        # -------------------------
        # Morph RLW=False, operation=both
        # -------------------------
        if clipping == "None":
            fixed_candidates = results_df[
                (results_df["Model_Name"] == MODEL_NON_DP) &
                (results_df["retinal_layer_wise"] == False) &
                (results_df["initial_use_morph"] == True) &
                (results_df["final_use_morph"] == True) &
                (results_df["Operation_norm"] == "both")
            ].copy()
        else:
            fixed_candidates = results_df[
                (results_df["Model_Name"] == MODEL_DP) &
                (results_df["retinal_layer_wise"] == False) &
                (results_df["initial_use_morph"] == True) &
                (results_df["final_use_morph"] == True) &
                (results_df["Operation_norm"] == "both") &
                (results_df["clipping_strategy_norm"] == clipping)
            ].copy()

        # -------------------------
        # NoMorph
        # -------------------------
        if clipping == "None":
            nomorph_candidates = results_df[
                (results_df["Model_Name"] == MODEL_NON_DP_NOMORPH)
            ].copy()
        else:
            nomorph_candidates = results_df[
                (results_df["Model_Name"] == MODEL_DP_NOMORPH) &
                (results_df["clipping_strategy_norm"] == clipping)
            ].copy()

        # best setting for each baseline
        k_table = 3

        best_fixed_runs_full = get_best_setting_runs_sample_first(
            fixed_candidates,
            main_metric,
            setting_cols,
            k=k_table,
            random_state=42,
        )

        best_nomorph_runs_full = get_best_setting_runs_sample_first(
            nomorph_candidates,
            main_metric,
            setting_cols,
            k=k_table,
            random_state=42,
        )

        # balance ONLY between Morph and NoMorph
        if best_fixed_runs_full.empty or best_nomorph_runs_full.empty:
            fixed_mean, fixed_std, fixed_n = None, None, 0
            nomorph_mean, nomorph_std, nomorph_n = None, None, 0
        else:
            n_balance = min(len(best_fixed_runs_full), len(best_nomorph_runs_full))

            best_fixed_runs = sample_runs_to_n(best_fixed_runs_full, n_balance)
            best_nomorph_runs = sample_runs_to_n(best_nomorph_runs_full, n_balance)

            fixed_mean, fixed_std, fixed_n = layerwise_mean_std_from_runs(best_fixed_runs, layers)
            nomorph_mean, nomorph_std, nomorph_n = layerwise_mean_std_from_runs(best_nomorph_runs, layers)



        # -------------------------
        # Save per-layer rows
        # -------------------------
        for i, layer in enumerate(layers):
            records.append({
                "policy_type": policy_type,
                "clipping": clipping,
                "layer": layer,

                "rlw_mean": rlw_mean[i],
                "rlw_std": rlw_std[i],
                "rlw_n": rlw_n,

                "fixed_mean": np.nan if fixed_mean is None else fixed_mean[i],
                "fixed_std": np.nan if fixed_std is None else fixed_std[i],
                "fixed_n": fixed_n,

                "nomorph_mean": np.nan if nomorph_mean is None else nomorph_mean[i],
                "nomorph_std": np.nan if nomorph_std is None else nomorph_std[i],
                "nomorph_n": nomorph_n,

                "delta_rlw_vs_fixed": np.nan if fixed_mean is None else rlw_mean[i] - fixed_mean[i],
                "delta_rlw_vs_nomorph": np.nan if nomorph_mean is None else rlw_mean[i] - nomorph_mean[i],

                "thickness_mean": thickness_vals[i],
                "op_majority": ops_vals[i],
            })

panel_df = pd.DataFrame(records)
if panel_df.empty:
    raise ValueError("No matched panels found.")

panel_df.to_csv(f"{model}_best_setting_layerwise_meanstd_deltas.csv", index=False)
print(f"Saved: {model}_best_setting_layerwise_meanstd_deltas.csv")

# =========================================
# PRINT TABLE
# =========================================
print("\n===== BEST-SETTING LAYER-WISE MEAN±STD DELTAS =====")
for policy_type in preferred_policy_order:
    for clipping in preferred_clip_order:
        sub = panel_df[
            (panel_df["policy_type"] == policy_type) &
            (panel_df["clipping"] == clipping)
        ].sort_values("layer")

        if sub.empty:
            continue

        print(f"\n--- Policy={policy_type}, Clipping={clipping} ---")
        print(
            sub[[
                "layer",
                "nomorph_mean", "nomorph_std", "nomorph_n",
                "fixed_mean", "fixed_std", "fixed_n",
                "rlw_mean", "rlw_std", "rlw_n",
                "delta_rlw_vs_fixed",
                "delta_rlw_vs_nomorph",
                "thickness_mean",
                "op_majority",
            ]].to_string(index=False)
        )

# =========================================
# PLOT 1: MAIN PLOT
# =========================================
policy_order = [p for p in preferred_policy_order if p in panel_df["policy_type"].astype(str).unique()]
clip_order = [c for c in preferred_clip_order if c in panel_df["clipping"].astype(str).unique()]

nrows = len(policy_order)
ncols = len(clip_order)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(3.5 * ncols, 3.5 * nrows),
    sharex=True,
    sharey=True
)
fig.patch.set_facecolor("white")

if nrows == 1:
    axes = np.array([axes])
if ncols == 1:
    axes = axes[:, np.newaxis]

panel_df["color"] = panel_df["op_majority"].map(color_map).fillna("black")

size_min, size_max = 45, 190
thickness_numeric = pd.to_numeric(panel_df["thickness_mean"], errors="coerce")
valid_th = np.isfinite(thickness_numeric)

panel_df["bubble_size"] = 90.0
if valid_th.any():
    t_valid = thickness_numeric[valid_th]
    tmin_global = t_valid.min()
    tmax_global = t_valid.max()

    if np.isclose(tmin_global, tmax_global):
        panel_df.loc[valid_th, "bubble_size"] = (size_min + size_max) / 2
    else:
        panel_df.loc[valid_th, "bubble_size"] = size_min + (size_max - size_min) * (
            (thickness_numeric[valid_th] - tmin_global) / (tmax_global - tmin_global + 1e-8)
        )

for r, policy_type in enumerate(policy_order):
    for c, clipping in enumerate(clip_order):
        ax = axes[r, c]

        sub = panel_df[
            (panel_df["policy_type"] == policy_type) &
            (panel_df["clipping"] == clipping)
        ].sort_values("layer")

        if sub.empty:
            ax.axis("off")
            continue

        ax.plot(
            sub["layer"],
            sub["nomorph_mean"],
            color="dimgray",
            linewidth=1.8,
            linestyle="--",
            alpha=0.7,
            zorder=1,
            label="No Morph"
        )

        ax.plot(
            sub["layer"],
            sub["fixed_mean"],
            color="black",
            linewidth=1.8,
            linestyle="-",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="black",
            alpha=0.9,
            zorder=2,
            label="DP-Morph (non-adaptive)"
        )

        ax.scatter(
            sub["layer"],
            sub["rlw_mean"],
            s=sub["bubble_size"],
            c=sub["color"],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            zorder=3
        )

        ax.set_xticks(layers)
        ax.tick_params(axis="both", labelsize=12)

        if r == 0:
            ax.set_title(label_map.get(clipping, clipping), fontsize=10, pad=5)

        if c == 0:
            ax.set_ylabel("Dice", fontsize=12)
            ax.text(
                -0.52, 0.5,
                policy_type.upper(),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                rotation=90,
                fontweight="bold"
            )

        if r == nrows - 1:
            ax.set_xlabel("Layers", fontsize=14)

legend_ops = [
    Line2D([0], [0], marker='o', color='w', label='Open',
           markerfacecolor=color_map["open"], markeredgecolor='black', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Close',
           markerfacecolor=color_map["close"], markeredgecolor='black', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Both',
           markerfacecolor=color_map["both"], markeredgecolor='black', markersize=6),
    Line2D([0], [0], color='dimgray', linestyle='--', label='No Morph',
           markersize=5, linewidth=1.8),
    Line2D([0], [0], color='black', linestyle='-', label="DP-Morph (non-adaptive)",
           markersize=5, linewidth=1.8),
]

fig.legend(
    handles=legend_ops,
    title="Operation",
    loc="lower center",
    bbox_to_anchor=(0.33, -0.02),
    ncol=3,
    frameon=False,
    fontsize=14,
    title_fontsize=14
)

valid_legend_th = thickness_numeric[np.isfinite(thickness_numeric)]
if len(valid_legend_th) > 0:
    tmin = valid_legend_th.min()
    tmid = valid_legend_th.median()
    tmax = valid_legend_th.max()

    if np.isclose(tmin, tmax):
        size_vals = [(size_min + size_max) / 2] * 3
    else:
        def scale_size(v):
            return size_min + (size_max - size_min) * ((v - tmin) / (tmax - tmin + 1e-8))
        size_vals = [scale_size(v) for v in [tmin, tmid, tmax]]

    legend_sizes = [
        plt.scatter([], [], s=s, color="lightgray", edgecolors="black")
        for s in size_vals
    ]

    fig.legend(
        legend_sizes,
        ["Thin", "Medium", "Thick"],
        title="Thickness",
        loc="lower center",
        bbox_to_anchor=(0.78, -0.02),
        ncol=3,
        frameon=False,
        fontsize=14,
        title_fontsize=14
    )
else:
    print("[WARN] No valid thickness values found; thickness legend skipped.")

plt.tight_layout(rect=[0.03, 0.06, 1.00, 0.90])
plt.savefig(f"{model}_policy_grid_best_final.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{model}_policy_grid_best_final.pdf", bbox_inches="tight")
plt.close()

print(f"Saved: {model}_policy_grid_best_final.png")
print(f"Saved: {model}_policy_grid_best_final.pdf")

# =========================================
# PLOT 2: DELTA PLOT
# =========================================
policy_order_present = [p for p in preferred_policy_order if p in panel_df["policy_type"].unique()]
clip_order_present = [c for c in preferred_clip_order if c in panel_df["clipping"].unique()]

fig, axes = plt.subplots(
    nrows=len(policy_order_present),
    ncols=len(clip_order_present),
    figsize=(2.8 * len(clip_order_present), 2.8 * len(policy_order_present)),
    sharex=True,
    sharey=True
)

fig.patch.set_facecolor("white")

if len(policy_order_present) == 1:
    axes = np.array([axes])
if len(clip_order_present) == 1:
    axes = axes[:, np.newaxis]

for r, policy_type in enumerate(policy_order_present):
    for c, clipping in enumerate(clip_order_present):
        ax = axes[r, c]

        sub = panel_df[
            (panel_df["policy_type"] == policy_type) &
            (panel_df["clipping"] == clipping)
        ].sort_values("layer")

        if sub.empty:
            ax.axis("off")
            continue

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.8)

        ax.plot(
            sub["layer"],
            sub["delta_rlw_vs_fixed"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Adative - DP-Morph",
            zorder=3
        )

        ax.plot(
            sub["layer"],
            sub["delta_rlw_vs_nomorph"],
            marker="^",
            linewidth=2.0,
            markersize=5,
            label="Adaptive - NoMorph",
            zorder=3
        )

        ax.set_xticks(layers)
        ax.tick_params(axis="both", labelsize=11)

        if r == 0:
            ax.set_title(label_map.get(clipping, clipping), fontsize=11, pad=6)

        if c == 0:
            ax.set_ylabel("Dice difference", fontsize=11)
            ax.text(
                -0.50, 0.5,
                policy_type.upper(),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                rotation=90,
                fontweight="bold"
            )

        if r == len(policy_order_present) - 1:
            ax.set_xlabel("Layers", fontsize=11)

legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='#1f77b4',
           markerfacecolor='#1f77b4',
           markeredgecolor='#1f77b4',
           label='Adaptive - DP-Morph',
           markersize=6,
           linewidth=2),
    Line2D([0], [0],
           marker='^',
           color='#ff7f0e',
           markerfacecolor='#ff7f0e',
           markeredgecolor='#ff7f0e',
           label='Adaptive - NoMorph',
           markersize=6,
           linewidth=2),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=False,
    fontsize=14
)

plt.tight_layout(rect=[0.03, 0.06, 1.00, 0.96])
plt.savefig(f"{model}_best_setting_meanstd_deltas.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{model}_best_setting_meanstd_deltas.pdf", bbox_inches="tight")
plt.close()

print(f"Saved: {model}_best_setting_meanstd_deltas.png")
print(f"Saved: {model}_best_setting_meanstd_deltas.pdf")