import os
import pandas as pd
import matplotlib.pyplot as plt

model= "LFUNet"
conditional_point="cond_None"
rlw="rlw_False"
dataset= "Duke"

csv_path = f"/scicore/home/wagner0024/parsar0000/miniconda3/2025_shiva_dp_morph_extension/morph_debug_events/{dataset}/{model}/{conditional_point}/{rlw}/morph_debug_stats.csv"
out_dir = "plots_morph_layerwise"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

last_epoch = df["epoch"].max()
df_last_epoch = df[df["epoch"] == last_epoch].copy()

gain_cols = [c for c in df.columns if c.startswith("dice_gain_c")]
gain_cols = sorted(gain_cols, key=lambda x: int(x.split("c")[-1]))
gain_cols = [c for c in gain_cols if not c.endswith("c0")] # remove background
gain_cols = [c for c in gain_cols if not c.endswith("c8")] # remove fluid for Duke

mean_gain = df_last_epoch[gain_cols].mean()
std_gain = df_last_epoch[gain_cols].std()

layer_ids = [int(c.split("c")[-1]) for c in gain_cols]

plt.figure(figsize=(8, 4.5))
plt.bar(layer_ids, mean_gain.values, yerr=std_gain.values, capsize=3)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Class / Layer index")
plt.ylabel("Mean Dice gain (after - before)")
plt.title(f"Per-layer effect of morphology (final epoch={last_epoch})")
plt.xticks(layer_ids)
plt.tight_layout()

out_path = os.path.join(out_dir, f"per_layer_dice_gain_final_epoch_mean_{dataset}_{model}_{conditional_point}_{rlw}.png")
plt.savefig(out_path, dpi=250, bbox_inches="tight")
plt.close()

print("Saved:", out_path)
print(f"\nFinal epoch: {last_epoch}")
for lid, val in zip(layer_ids, mean_gain.values):
    print(f"class {lid}: {val:.5f}")