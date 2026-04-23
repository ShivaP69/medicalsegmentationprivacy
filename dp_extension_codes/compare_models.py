import torch
import matplotlib.pyplot as plt

from networks import get_model
from data_one_gpu import get_data
from losses import CombinedLoss
from train_one_gpu import eval, segmentation_plots

device = "cuda"
n_classes = 9
img_size = 224
batch_size = 16
dataset = "Duke"
data_path = "./DukeData"  # change if needed

# List of models you want to compare
MODELS = [
    {
        "label": "NestedUNet_NoMorph",
        "ckpt": "./saved_models/models/NestedUNet_Duke_200_16.pt",
        "morphology": False,
        "learnable_radius": False,
        "operation": "open",   # ignored when morphology=False
        "kernel_size": 3,
    },
    {
        "label": "NestedUNet_FixedR3",
        "ckpt": "./saved_models/morphology_models/NestedUNet_Duke_200_16_close_6_FixedR6.pt",
        "morphology": True,
        "learnable_radius": False,
        "operation": "open",
        "kernel_size": 3,
    },
    {
        "label": "NestedUNet_LearnR",
        "ckpt": "./saved_models/morphology_models/NestedUNet_Duke_200_16_close_6_LearnR.pt",
        # or whatever name you used for the learnable checkpoint
        "morphology": True,
        "learnable_radius": True,
        "operation": "open",
        "kernel_size": 3,
    },
]
def load_model(cfg):

    model = get_model(
        model_name="NestedUNet",
        ratio=0.5,
        num_classes=n_classes,
        morphology=cfg["morphology"],
        operation=cfg["operation"],
        kernel_size=cfg["kernel_size"],
        learnable_radius=cfg["learnable_radius"],
    ).to(device)

    checkpoint = torch.load(cfg["ckpt"], map_location=device,weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Strip `_module.` if saved with DataParallel
    new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def main():
    # 1) get test_loader
    train_loader, val_loader, test_loader, _, _, _ = get_data(
        data_path, img_size, batch_size
    )

    criterion_seg = CombinedLoss()

    all_results = []

    for cfg in MODELS:
        print(f"\n=== Testing model: {cfg['label']} ===")
        model = load_model(cfg)

        # 2) compute metrics on test set
        dice, loss, dice_all, mae, per_layer = eval(
            test_loader,
            criterion_seg,
            model,
            n_classes=n_classes,
            dice_s=True,
            device=device,
        )

        print(f"{cfg['label']}  ->  Dice: {dice:.4f}, Loss: {loss:.4f}, MAE: {mae:.4f}")
        all_results.append(
            {
                "label": cfg["label"],
                "dice": float(dice),
                "loss": float(loss),
                "mae": float(mae),
            }
        )

        # 3) qualitative plots on test set
        # You can change num_examples to e.g. 3 or 10
        segmentation_plots(
            test_loader,
            model,
            device,
            model_name=cfg["label"],   # will appear in file name if you save
            DPSGD=False,               # or True if that model is DP
            dataset=dataset,
            num_examples=3,
        )

    # 4) Optionally: bar plot of Dice across models
    labels = [r["label"] for r in all_results]
    dices = [r["dice"] for r in all_results]

    plt.figure()
    plt.bar(labels, dices)
    plt.ylabel("Test Dice")
    plt.xlabel("Model")
    plt.title("Test Dice: NoMorph vs FixedR vs LearnR")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/test_model_compare_dice.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    main()
