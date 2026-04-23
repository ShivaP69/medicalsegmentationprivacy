import pandas as pd
import numpy as np
import os

"""
df=pd.read_csv("/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/csv_results/test_oct_results_UMN_Duke_lr_max_gradient.csv")

df["Model_Name"] = df["Model_Name"].str.replace(
    "NestedUNet",
    "deepsuper_NestedUNet",
    regex=False
)
df.to_csv('/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/csv_results/test_oct_results_UMN_Duke_lr_max_gradient.csv', index=False)
"""

# =========================
# CONFIG
# =========================
def best_per_model(dataset_filter="Duke",k=3):
    file_path = "test_oct_results_UMN_Duke_lr_max_gradient.csv"

    # main metric used to choose "best"
    main_metric = "Validation_Dice"   # higher is better

    # extra metrics to report
    report_metrics = [
        "Validation_Dice",
        "mae",
        "best_hd95",
        'wall_time_train_s',
        'seed'
    ]

    # columns that define a unique setting
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
    ]

    # optional: filter to one dataset only
    #dataset_filter = "Duke"   # or "UMN" or None


    # =========================
    # LOAD
    # =========================
    df = pd.read_csv(file_path)

    # numeric cleanup
    for col in report_metrics + ["Batch_Size", "Kernel", "max_grad_norm", "Privacy_Epsilons"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    if dataset_filter is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter].copy()

    # -------------------------
    # infer privacy setting
    # -------------------------
    if "DPSGD" in df.columns:
        dpsgd = df["DPSGD"].astype(str).str.lower()
        df["privacy_setting"] = np.where(
            dpsgd.isin(["true", "1", "yes"]),
            "private",
            "non-private"
        )
    elif "clipping_strategy" in df.columns:
        clip = df["clipping_strategy"].astype(str).str.strip().str.lower()
        df["privacy_setting"] = np.where(
            clip.isin(["", "nan", "none"]),
            "non-private",
            "private",
        )
    elif "Privacy_Epsilons" in df.columns:
        df["privacy_setting"] = np.where(
            df["Privacy_Epsilons"].fillna(0) > 0,
            "private",
            "non-private"
        )
    else:
        df["privacy_setting"] = np.where(
            df["Model_Name"].str.contains("DPSGD", case=False, na=False),
            "private",
            "non-private",
        )

    # -------------------------
    # infer morph / no morph
    # -------------------------
    has_nomorph_name = df["Model_Name"].astype(str).str.contains("NoMorph", case=False, na=False)

    if "Operation" in df.columns:
        op = df["Operation"].astype(str).str.lower()
        no_morph_from_op = op.isin(["none", "nan", "nomorph", ""])
    else:
        no_morph_from_op = pd.Series(False, index=df.index)

    df["morph_setting"] = np.where(has_nomorph_name | no_morph_from_op, "NoMorph", "Morph")

    # normalize operation labels
    if "Operation" in df.columns:
        df["Operation"] = df["Operation"].replace({
            "None": "NoMorph",
            "none": "NoMorph",
            "nan": "NoMorph",
            "": "NoMorph",
        })

    # keep only existing setting columns
    setting_cols = [c for c in setting_cols if c in df.columns]
    df = (
            df.groupby(setting_cols, dropna=False, group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), k), random_state=42))
        )# filter k random samples for each setting
    # =========================
    # AGGREGATE ALL SETTINGS
    # =========================
    agg_dict = {}
    for m in report_metrics:
        if m in df.columns:
            agg_dict[m] = ["mean", "std", "count"]

    summary = (
        df.groupby(["privacy_setting", "morph_setting"] + setting_cols, dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # -------------------------
    # Add clean model name
    # -------------------------
    summary["Model"] = (
        summary["Model_Name"]
        .astype(str)
        .str.replace("_DPSGD", "", regex=False)
        .str.replace("_NoMorph", "", regex=False)
        .str.replace("_FixedR3", "", regex=False)
        .str.replace("_FixedR5", "", regex=False)
        .str.replace("_LearnR", "", regex=False)
        .str.strip()
    )

    # =========================
    # HELPERS
    # =========================
    def fmt_metric(row, metric):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        n_col = f"{metric}_count"

        mean_val = row.get(mean_col, np.nan)
        std_val = row.get(std_col, np.nan)
        n_val = row.get(n_col, np.nan)

        if pd.isna(mean_val):
            return "NA"
        if pd.isna(std_val) or pd.isna(n_val) or int(n_val) <= 1:
            return f"{mean_val:.4f} (n={int(n_val) if not pd.isna(n_val) else 1})"
        return f"{mean_val:.4f} ± {std_val:.4f} (n={int(n_val)})"


    def print_best_block(title, best_rows, model_name):
        print("\n" + "=" * 120)
        print(f"{title} | MODEL: {model_name}")
        print("=" * 120)

        if best_rows.empty:
            print("No matching rows found.")
            return

        for _, row in best_rows.iterrows():
            print("-" * 120)

            if "clipping_strategy" in row.index:
                clip = row["clipping_strategy"]
                if pd.notna(clip) and str(clip).lower() != "nan" and str(clip) != "":
                    print(f"clipping_strategy: {clip}")

            for c in setting_cols:
                if c in row.index:
                    print(f"{c}: {row[c]}")

            print("RESULTS:")
            for m in report_metrics:
                metric_name = {
                    "Validation_Dice": "Dice",
                    "mae": "MAE",
                    "best_hd95": "HD95",
                }.get(m, m)
                print(f"  {metric_name}: {fmt_metric(row, m)}")

        print("-" * 120)


    # =========================
    # LOOP PER MODEL
    # =========================
    all_models = sorted(summary["Model"].dropna().unique())

    all_best_nonprivate_nomorph = []
    all_best_nonprivate_morph = []
    all_best_private_nomorph = []
    all_best_private_morph = []

    for model_name in all_models:
        model_df = summary[summary["Model"] == model_name].copy()

        # 1) best non-private + NoMorph
        nonprivate_nomorph = model_df[
            (model_df["privacy_setting"] == "non-private") &
            (model_df["morph_setting"] == "NoMorph")
        ].copy()

        best_nonprivate_nomorph = nonprivate_nomorph.sort_values(
            by=f"{main_metric}_mean", ascending=False
        ).head(1)

        # 2) best non-private + Morph
        nonprivate_morph = model_df[
            (model_df["privacy_setting"] == "non-private") &
            (model_df["morph_setting"] == "Morph")
        ].copy()

        best_nonprivate_morph = nonprivate_morph.sort_values(
            by=f"{main_metric}_mean", ascending=False
        ).head(1)

        # 3) best private + NoMorph per clipping strategy
        private_nomorph = model_df[
            (model_df["privacy_setting"] == "private") &
            (model_df["morph_setting"] == "NoMorph")
        ].copy()

        if "clipping_strategy" in private_nomorph.columns:
            valid_private_nomorph = private_nomorph[
                private_nomorph["clipping_strategy"].notna()
                & (private_nomorph["clipping_strategy"].astype(str).str.lower() != "nan")
                & (private_nomorph["clipping_strategy"].astype(str).str.lower() != "none")
                & (private_nomorph["clipping_strategy"].astype(str) != "")
            ].copy()

            best_private_nomorph = (
                valid_private_nomorph.sort_values(by=f"{main_metric}_mean", ascending=False)
                .groupby("clipping_strategy", dropna=False)
                .head(1)
                .reset_index(drop=True)
            )
        else:
            best_private_nomorph = private_nomorph.sort_values(
                by=f"{main_metric}_mean", ascending=False
            ).head(1)

        # 4) best private + Morph per clipping strategy
        private_morph = model_df[
            (model_df["privacy_setting"] == "private") &
            (model_df["morph_setting"] == "Morph")
        ].copy()

        if "clipping_strategy" in private_morph.columns:
            valid_private_morph = private_morph[
                private_morph["clipping_strategy"].notna()
                & (private_morph["clipping_strategy"].astype(str).str.lower() != "nan")
                & (private_morph["clipping_strategy"].astype(str).str.lower() != "none")
                & (private_morph["clipping_strategy"].astype(str) != "")
            ].copy()

            best_private_morph = (
                valid_private_morph.sort_values(by=f"{main_metric}_mean", ascending=False)
                .groupby("clipping_strategy", dropna=False)
                .head(1)
                .reset_index(drop=True)
            )
        else:
            best_private_morph = private_morph.sort_values(
                by=f"{main_metric}_mean", ascending=False
            ).head(1)

        # print
        print_best_block("BEST SETTING: NON-PRIVATE + NoMorph", best_nonprivate_nomorph, model_name)
        print_best_block("BEST SETTING: NON-PRIVATE + Morph", best_nonprivate_morph, model_name)
        print_best_block("BEST SETTINGS: PRIVATE + NoMorph (per clipping strategy)", best_private_nomorph, model_name)
        print_best_block("BEST SETTINGS: PRIVATE + Morph (per clipping strategy)", best_private_morph, model_name)

        # collect to save
        if not best_nonprivate_nomorph.empty:
            all_best_nonprivate_nomorph.append(best_nonprivate_nomorph)
        if not best_nonprivate_morph.empty:
            all_best_nonprivate_morph.append(best_nonprivate_morph)
        if not best_private_nomorph.empty:
            all_best_private_nomorph.append(best_private_nomorph)
        if not best_private_morph.empty:
            all_best_private_morph.append(best_private_morph)



def best_overall(dataset_filter="Duke",k = 3):
    # =========================
    # CONFIG
    # =========================
    file_path = "test_oct_results_UMN_Duke_lr_max_gradient.csv"

    # main metric used to choose "best"
    main_metric = "Validation_Dice"   # higher is better

    # extra metrics to report
    report_metrics = [
        "Validation_Dice",
        "mae",
        "best_hd95",
        "wall_time_train_s",
    ]

    # columns that define a unique setting
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
    ]

    # optional: filter to one dataset only, e.g. "Duke"
    #dataset_filter = "Duke"   # set to "Duke" or "UMN" if needed


    # =========================
    # LOAD
    # =========================
    df = pd.read_csv(file_path)

    # numeric cleanup
    for col in report_metrics + ["Batch_Size", "Kernel", "max_grad_norm", "Privacy_Epsilons"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # normalize strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    if dataset_filter is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter].copy()

    # -------------------------
    # infer privacy setting
    # -------------------------
    if "DPSGD" in df.columns:
        dpsgd = df["DPSGD"].astype(str).str.lower()
        df["privacy_setting"] = np.where(dpsgd.isin(["true", "1", "yes"]), "private", "non-private")
    elif "clipping_strategy" in df.columns:
        df["privacy_setting"] = np.where(
            df["clipping_strategy"].notna() & (df["clipping_strategy"] != "") & (df["clipping_strategy"].str.lower() != "nan"),
            "private",
            "non-private",
        )
    elif "Privacy_Epsilons" in df.columns:
        df["privacy_setting"] = np.where(df["Privacy_Epsilons"].fillna(0) > 0, "private", "non-private")
    else:
        df["privacy_setting"] = np.where(
            df["Model_Name"].str.contains("DPSGD", case=False, na=False),
            "private",
            "non-private",
        )

    # -------------------------
    # infer morph / no morph
    # -------------------------
    # priority: from Model_Name suffix
    has_nomorph_name = df["Model_Name"].astype(str).str.contains("NoMorph", case=False, na=False)

    # fallback using operation
    if "Operation" in df.columns:
        op = df["Operation"].astype(str).str.lower()
        no_morph_from_op = op.isin(["none", "nan", "nomorph", ""])
    else:
        no_morph_from_op = pd.Series(False, index=df.index)

    df["morph_setting"] = np.where(has_nomorph_name | no_morph_from_op, "NoMorph", "Morph")

    # normalize operation labels
    if "Operation" in df.columns:
        df["Operation"] = df["Operation"].replace({
            "None": "NoMorph",
            "none": "NoMorph",
            "nan": "NoMorph",
            "": "NoMorph",
        })


    # keep only existing setting columns
    setting_cols = [c for c in setting_cols if c in df.columns]
    df = (
        df.groupby(setting_cols, dropna=False, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), k), random_state=42))
    )# filter k random samples for each setting

    # =========================
    # AGGREGATE ALL SETTINGS
    # =========================
    agg_dict = {}
    for m in report_metrics:
        if m in df.columns:
            agg_dict[m] = ["mean", "std", "count"]

    summary = (
        df.groupby(["privacy_setting", "morph_setting"] + setting_cols, dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]


    # helper
    def fmt_metric(row, metric):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        n_col = f"{metric}_count"
        mean_val = row.get(mean_col, np.nan)
        std_val = row.get(std_col, np.nan)
        n_val = row.get(n_col, np.nan)

        if pd.isna(mean_val):
            return "NA"
        if pd.isna(std_val) or pd.isna(n_val) or int(n_val) <= 1:
            return f"{mean_val:.4f} (n={int(n_val) if not pd.isna(n_val) else 1})"
        return f"{mean_val:.4f} ± {std_val:.4f} (n={int(n_val)})"

    def print_best_block(title, best_rows):
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
        if best_rows.empty:
            print("No matching rows found.")
            return

        for i, row in best_rows.iterrows():
            print("-" * 110)
            if "clipping_strategy" in row.index:
               clip = str(row["clipping_strategy"]).strip().lower()
               if clip not in ["", "nan", "none"]:
                    print(f"clipping_strategy: {row['clipping_strategy']}")

            for c in setting_cols:
                if c in row.index:
                    print(f"{c}: {row[c]}")

            print("RESULTS:")
            for m in report_metrics:
                metric_name = {
                    "Validation_Dice": "Dice",
                    "mae": "MAE",
                    "best_hd95": "HD95",
                }.get(m, m)
                print(f"  {metric_name}: {fmt_metric(row, m)}")
        print("-" * 110)


    # =========================
    # 1) BEST NON-PRIVATE NOMORPH
    # =========================
    nonprivate_nomorph = summary[
        (summary["privacy_setting"] == "non-private") &
        (summary["morph_setting"] == "NoMorph")
    ].copy()

    best_nonprivate_nomorph = nonprivate_nomorph.sort_values(
        by=f"{main_metric}_mean", ascending=False
    ).head(1)

    # =========================
    # 2) BEST NON-PRIVATE MORPH
    # =========================
    nonprivate_morph = summary[
        (summary["privacy_setting"] == "non-private") &
        (summary["morph_setting"] == "Morph")
    ].copy()

    best_nonprivate_morph = nonprivate_morph.sort_values(
        by=f"{main_metric}_mean", ascending=False
    ).head(1)

    # =========================
    # 3) BEST PRIVATE NOMORPH PER CLIPPING STRATEGY
    # =========================
    private_nomorph = summary[
        (summary["privacy_setting"] == "private") &
        (summary["morph_setting"] == "NoMorph")
    ].copy()

    if "clipping_strategy" in private_nomorph.columns:
        best_private_nomorph = (
            private_nomorph.sort_values(by=f"{main_metric}_mean", ascending=False)
            .groupby("clipping_strategy", dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        best_private_nomorph = private_nomorph.sort_values(
            by=f"{main_metric}_mean", ascending=False
        ).head(1)

    # =========================
    # 4) BEST PRIVATE MORPH PER CLIPPING STRATEGY
    # =========================
    private_morph = summary[
        (summary["privacy_setting"] == "private") &
        (summary["morph_setting"] == "Morph")
    ].copy()

    if "clipping_strategy" in private_morph.columns:
        best_private_morph = (
            private_morph.sort_values(by=f"{main_metric}_mean", ascending=False)
            .groupby("clipping_strategy", dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        best_private_morph = private_morph.sort_values(
            by=f"{main_metric}_mean", ascending=False
        ).head(1)

    # =========================
    # PRINT
    # =========================
    print_best_block("BEST SETTING: NON-PRIVATE + NoMorph", best_nonprivate_nomorph)
    print_best_block("BEST SETTING: NON-PRIVATE + Morph", best_nonprivate_morph)
    print_best_block("BEST SETTINGS: PRIVATE + NoMorph (per clipping strategy)", best_private_nomorph)
    print_best_block("BEST SETTINGS: PRIVATE + Morph (per clipping strategy)", best_private_morph)

    # =========================
    # OPTIONAL SAVE
    # =========================
    """best_nonprivate_nomorph.to_csv("best_nonprivate_nomorph.csv", index=False)
    best_nonprivate_morph.to_csv("best_nonprivate_morph.csv", index=False)
    best_private_nomorph.to_csv("best_private_nomorph_per_clipping.csv", index=False)
    best_private_morph.to_csv("best_private_morph_per_clipping.csv", index=False)

    print("\nSaved:")
    print("  best_nonprivate_nomorph.csv")
    print("  best_nonprivate_morph.csv")
    print("  best_private_nomorph_per_clipping.csv")
    print("  best_private_morph_per_clipping.csv")"""

import pandas as pd
import numpy as np

def build_best_table(
    file_path="test_oct_results_UMN_Duke_lr_max_gradient.csv",
    dataset_filter="Duke",
    model_filter=("LFUNet", "NestedUNet"),
    k=3,
):
    main_metric = "Validation_Dice"
    report_metrics = ["Validation_Dice", "mae", "best_hd95", "wall_time_train_s"]

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
    ]

    df = pd.read_csv(file_path)

    # numeric cleanup
    for col in report_metrics + ["Batch_Size", "Kernel", "max_grad_norm", "Privacy_Epsilons"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    if dataset_filter is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter].copy()

    # infer privacy setting
    if "DPSGD" in df.columns:
        dpsgd = df["DPSGD"].astype(str).str.lower()
        df["privacy_setting"] = np.where(dpsgd.isin(["true", "1", "yes"]), "private", "non-private")
    elif "clipping_strategy" in df.columns:
        clip = df["clipping_strategy"].astype(str).str.strip().str.lower()
        df["privacy_setting"] = np.where(~clip.isin(["", "nan", "none"]), "private", "non-private")
    elif "Privacy_Epsilons" in df.columns:
        df["privacy_setting"] = np.where(df["Privacy_Epsilons"].fillna(0) > 0, "private", "non-private")
    else:
        df["privacy_setting"] = np.where(
            df["Model_Name"].str.contains("DPSGD", case=False, na=False),
            "private",
            "non-private",
        )

    # infer morph setting
    has_nomorph_name = df["Model_Name"].astype(str).str.contains("NoMorph", case=False, na=False)
    if "Operation" in df.columns:
        op = df["Operation"].astype(str).str.lower()
        no_morph_from_op = op.isin(["none", "nan", "nomorph", ""])
    else:
        no_morph_from_op = pd.Series(False, index=df.index)

    df["morph_setting"] = np.where(has_nomorph_name | no_morph_from_op, "NoMorph", "Morph")

    if "Operation" in df.columns:
        df["Operation"] = df["Operation"].replace({
            "None": "NoMorph",
            "none": "NoMorph",
            "nan": "NoMorph",
            "": "NoMorph",
        })

    setting_cols = [c for c in setting_cols if c in df.columns]

    # optional: randomly keep only k runs per setting
    df = (
        df.groupby(setting_cols, dropna=False, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), k), random_state=42))
          .reset_index(drop=True)
    )

    # aggregate
    agg_dict = {m: ["mean", "std", "count"] for m in report_metrics if m in df.columns}
    summary = (
        df.groupby(["privacy_setting", "morph_setting"] + setting_cols, dropna=False)
          .agg(agg_dict)
          .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # clean model name
    summary["Model"] = (
        summary["Model_Name"]
        .astype(str)
        .str.replace("_DPSGD", "", regex=False)
        .str.replace("_NoMorph", "", regex=False)
        .str.replace("_FixedR3", "", regex=False)
        .str.replace("_FixedR5", "", regex=False)
        .str.replace("_LearnR", "", regex=False)
        .str.strip()
    )

    summary = summary[summary["Model"].isin(model_filter)].copy()

    def format_mean_std(row, metric, decimals=4):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        n_col = f"{metric}_count"
        if mean_col not in row:
            return "NA"
        mean_val = row[mean_col]
        std_val = row.get(std_col, np.nan)
        n_val = row.get(n_col, np.nan)

        if pd.isna(mean_val):
            return "NA"
        if pd.isna(std_val) or pd.isna(n_val) or int(n_val) <= 1:
            return f"{mean_val:.{decimals}f}"
        return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"

    rows = []

    for model_name in model_filter:
        model_df = summary[summary["Model"] == model_name].copy()

        # best non-private NoMorph
        s = model_df[
            (model_df["privacy_setting"] == "non-private") &
            (model_df["morph_setting"] == "NoMorph")
        ].sort_values(by=f"{main_metric}_mean", ascending=False).head(1)

        if not s.empty:
            r = s.iloc[0]
            rows.append({
                "Model": model_name,
                "Setting": "Non-private",
                "Method": "NoMorph",
                "Dice": format_mean_std(r, "Validation_Dice"),
                "MAE": format_mean_std(r, "mae"),
                "HD95": format_mean_std(r, "best_hd95"),
                "TrainTime": format_mean_std(r, "wall_time_train_s", decimals=1),
                "Details": f"BS={r['Batch_Size']}"
            })

        # best non-private Morph
        s = model_df[
            (model_df["privacy_setting"] == "non-private") &
            (model_df["morph_setting"] == "Morph")
        ].sort_values(by=f"{main_metric}_mean", ascending=False).head(1)

        if not s.empty:
            r = s.iloc[0]
            rows.append({
                "Model": model_name,
                "Setting": "Non-private",
                "Method": "Morph",
                "Dice": format_mean_std(r, "Validation_Dice"),
                "MAE": format_mean_std(r, "mae"),
                "HD95": format_mean_std(r, "best_hd95"),
                "TrainTime": format_mean_std(r, "wall_time_train_s", decimals=1),
                "Details": f"BS={r['Batch_Size']}, Op={r['Operation']}, cond={r['conditional_point']}"
            })

        # best private NoMorph per clipping
        s = model_df[
            (model_df["privacy_setting"] == "private") &
            (model_df["morph_setting"] == "NoMorph")
        ].copy()

        if not s.empty:
            s = (
                s.sort_values(by=f"{main_metric}_mean", ascending=False)
                 .groupby("clipping_strategy", dropna=False)
                 .head(1)
            )
            for _, r in s.iterrows():
                rows.append({
                    "Model": model_name,
                    "Setting": "Private",
                    "Method": f"NoMorph + {r['clipping_strategy']}",
                    "Dice": format_mean_std(r, "Validation_Dice"),
                    "MAE": format_mean_std(r, "mae"),
                    "HD95": format_mean_std(r, "best_hd95"),
                    "TrainTime": format_mean_std(r, "wall_time_train_s", decimals=1),
                    "Details": f"BS={r['Batch_Size']}, MGN={r['max_grad_norm']}"
                })

        # best private Morph per clipping
        s = model_df[
            (model_df["privacy_setting"] == "private") &
            (model_df["morph_setting"] == "Morph")
        ].copy()

        if not s.empty:
            s = (
                s.sort_values(by=f"{main_metric}_mean", ascending=False)
                 .groupby("clipping_strategy", dropna=False)
                 .head(1)
            )
            for _, r in s.iterrows():
                rows.append({
                    "Model": model_name,
                    "Setting": "Private",
                    "Method": f"Morph + {r['clipping_strategy']}",
                    "Dice": format_mean_std(r, "Validation_Dice"),
                    "MAE": format_mean_std(r, "mae"),
                    "HD95": format_mean_std(r, "best_hd95"),
                    "TrainTime": format_mean_std(r, "wall_time_train_s", decimals=1),
                    "Details": f"BS={r['Batch_Size']}, Op={r['Operation']}, MGN={r['max_grad_norm']}"
                })

    table_df = pd.DataFrame(rows)
    return table_df

def latex_escape(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("_", "\\_")
    return text

def build_paper_style_latex(table_df, caption="", label="tab:best_results"):
    """
    Expected columns in table_df:
    - Model
    - Setting
    - Method
    - Dice
    - MAE
    - HD95
    - Details   (optional, ignored here)
    """

    # rename models for paper style if needed
    model_name_map = {
        "NestedUNet": "UNet++",
        "unet": "UNet",
        "LFUNet": "LFUNet",
    }

    # standardize setting names
    def normalize_setting_method(row):
        setting = str(row["Setting"]).strip()
        method = str(row["Method"]).strip()

        if setting == "Non-private":
            if method == "NoMorph":
                return "Non-private", "No Morph"
            elif method == "Morph":
                return "Non-private", "Morph"
        elif setting == "Private":
            if method.startswith("NoMorph + "):
                return "DP (no morph)", method.replace("NoMorph + ", "").upper()
            elif method.startswith("Morph + "):
                return "DP + Morph", method.replace("Morph + ", "").upper()

        return setting, method

    df = table_df.copy()
    df[["Setting_norm", "Method_norm"]] = df.apply(
        lambda r: pd.Series(normalize_setting_method(r)), axis=1
    )

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l l l|c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Setting & Method & Dice $\uparrow$ & \ac{HD95} $\downarrow$ & MAE $\downarrow$ \\")
    lines.append(r"\midrule")

    model_order = ["NestedUNet", "unet", "LFUNet"]
    method_order_dp = ["AUTOMATIC", "FLAT", "NORMALIZED_SGD", "PSAC"]
    pretty_method_map = {
        "AUTOMATIC": "AUTO-S",
        "FLAT": "Flat",
        "NORMALIZED_SGD": "NSGD",
        "PSAC": "PSAC",
    }

    for model in model_order:
        sub = df[df["Model"] == model].copy()
        if sub.empty:
            continue

        paper_model = model_name_map.get(model, model)

        lines.append(f"% ===================== {paper_model} =====================")
        lines.append(rf"\multicolumn{{6}}{{c}}{{\textbf{{{paper_model}}}}} \\")
        lines.append(r"\midrule")

        # ---------- Non-private ----------
        sub_np = sub[sub["Setting_norm"] == "Non-private"].copy()

        np_order = {"No Morph": 0, "Morph": 1}
        sub_np["__ord"] = sub_np["Method_norm"].map(np_order)
        sub_np = sub_np.sort_values("__ord")

        if len(sub_np) > 0:
            dice_vals = pd.to_numeric(sub_np["Dice"].str.extract(r"([0-9.]+)")[0], errors="coerce")
            hd_vals = pd.to_numeric(sub_np["HD95"].str.extract(r"([0-9.]+)")[0], errors="coerce")
            mae_vals = pd.to_numeric(sub_np["MAE"].str.extract(r"([0-9.]+)")[0], errors="coerce")

            best_dice_idx = dice_vals.idxmax() if not dice_vals.isna().all() else None
            best_hd_idx = hd_vals.idxmin() if not hd_vals.isna().all() else None
            best_mae_idx = mae_vals.idxmin() if not mae_vals.isna().all() else None

            for i, (_, row) in enumerate(sub_np.iterrows()):
                dice = row["Dice"]
                hd95 = row["HD95"]
                mae = row["MAE"]

                if row.name == best_dice_idx:
                    dice = rf"\textbf{{{dice}}}"
                if row.name == best_hd_idx:
                    hd95 = rf"\textbf{{{hd95}}}"
                if row.name == best_mae_idx:
                    mae = rf"\textbf{{{mae}}}"

                if i == 0:
                    lines.append(
                        rf"\multirow{{{len(sub_np)}}}{{*}}{{{paper_model}}} & "
                        rf"\multirow{{{len(sub_np)}}}{{*}}{{Non-private}} "
                        rf"& {latex_escape(row['Method_norm'])} & {dice} & {hd95} & {mae} \\"
                    )
                else:
                    lines.append(
                        rf"& & {latex_escape(row['Method_norm'])} & {dice} & {hd95} & {mae} \\"
                    )

        # ---------- DP (no morph) ----------
        sub_dp_nm = sub[sub["Setting_norm"] == "DP (no morph)"].copy()
        if not sub_dp_nm.empty:
            lines.append(r"\cmidrule(lr){2-6}")
            sub_dp_nm["__ord"] = sub_dp_nm["Method_norm"].map(
                lambda x: method_order_dp.index(x) if x in method_order_dp else 999
            )
            sub_dp_nm = sub_dp_nm.sort_values("__ord")

            for i, (_, row) in enumerate(sub_dp_nm.iterrows()):
                method = pretty_method_map.get(row["Method_norm"], row["Method_norm"])
                if i == 0:
                    lines.append(
                        rf"\multirow{{{len(sub_dp_nm)}}}{{*}}{{{paper_model}}} & "
                        rf"\multirow{{{len(sub_dp_nm)}}}{{*}}{{DP (no morph)}} "
                        rf"& {method} & {row['Dice']} & {row['HD95']} & {row['MAE']} \\"
                    )
                else:
                    lines.append(
                        rf"& & {method} & {row['Dice']} & {row['HD95']} & {row['MAE']} \\"
                    )

        # ---------- DP + Morph ----------
        sub_dp_m = sub[sub["Setting_norm"] == "DP + Morph"].copy()
        if not sub_dp_m.empty:
            lines.append(r"\cmidrule(lr){2-6}")
            sub_dp_m["__ord"] = sub_dp_m["Method_norm"].map(
                lambda x: method_order_dp.index(x) if x in method_order_dp else 999
            )
            sub_dp_m = sub_dp_m.sort_values("__ord")

            for i, (_, row) in enumerate(sub_dp_m.iterrows()):
                method = pretty_method_map.get(row["Method_norm"], row["Method_norm"])
                if i == 0:
                    lines.append(
                        rf"\multirow{{{len(sub_dp_m)}}}{{*}}{{{paper_model}}} & "
                        rf"\multirow{{{len(sub_dp_m)}}}{{*}}{{DP + Morph}} "
                        rf"& {method} & {row['Dice']} & {row['HD95']} & {row['MAE']} \\"
                    )
                else:
                    lines.append(
                        rf"& & {method} & {row['Dice']} & {row['HD95']} & {row['MAE']} \\"
                    )

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

def winner_batch(dataset_filter="UMN"):
    df = pd.read_csv("test_oct_results_UMN_Duke_lr_max_gradient.csv")

    # clean
    df["Validation_Dice"] = pd.to_numeric(df["Validation_Dice"], errors="coerce")
    df["Batch_Size"] = pd.to_numeric(df["Batch_Size"], errors="coerce")

    # filter UMN
    df = df[df["dataset"] == dataset_filter].copy()

    # keep only BS 8 and 16
    df = df[df["Batch_Size"].isin([8, 16])]

    # group and compare
    summary = (
        df.groupby(["Model_Name", "Batch_Size"])
          .agg(
              Dice_mean=("Validation_Dice", "mean"),
              Dice_std=("Validation_Dice", "std"),
              n=("Validation_Dice", "count"),
          )
          .reset_index()
    )

    print(summary)



dataset_filter="UMN"
table_df = build_best_table(
    file_path="test_oct_results_UMN_Duke_lr_max_gradient.csv",
    dataset_filter=dataset_filter,
    model_filter=("unet", "NestedUNet"),
    k=3,
)
table_df = table_df.applymap(
    lambda x: str(x).replace("±", "$\\pm$") if isinstance(x, str) else x
)
latex_table = build_paper_style_latex(
    table_df,
    caption=f"Best results for UNet and UNet++ on the {dataset_filter} dataset.",
    label="tab:combined_models_best"
)

output_dir = "latex_tables"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, f"table_{dataset_filter}.tex"), "w", encoding="utf-8") as f:
    f.write(latex_table)
print(latex_table)

#best_overall(dataset_filter="UMN",k=3)
#best_per_model(dataset_filter="Duke",k=3)

#winner_batch()
