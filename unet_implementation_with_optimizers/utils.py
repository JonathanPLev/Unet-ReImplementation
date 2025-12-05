import os

import matplotlib.pyplot as plt

from config import PLOT_DIR, RUN_DETAIL


def dice_score(pred, target, epsilon=1e-6):
    pred = pred.argmax(dim=1)
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=[1, 2])
    union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()


def IoU_score(pred, target, epsilon=1e-6):
    pred = pred.argmax(dim=1)
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=[1, 2])
    pred_sum = pred.sum(dim=[1, 2])
    target_sum = target.sum(dim=[1, 2])
    union = pred_sum + target_sum - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()


def pixel_accuracy(pred, target):
    pred = pred.argmax(dim=1)
    correct = (pred == target).float()
    return correct.mean().item()


def plot_history(history, out_dir=PLOT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    def plot_pair(metric, ylabel):
        plt.figure()
        train_vals = history.get(f"train_{metric}", [])
        val_vals = history.get(f"val_{metric}", [])
        epochs = range(1, len(train_vals) + 1)
        plt.plot(epochs, train_vals, label="train")
        if val_vals and val_vals[0] is not None:
            plt.plot(epochs, val_vals, label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{metric} over epochs")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        out_path = os.path.join(out_dir, f"{metric}_{RUN_DETAIL}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    plot_pair("loss", "Loss")
    plot_pair("dice", "Dice")
    plot_pair("iou", "IoU")
    plot_pair("pixel_acc", "Pixel Accuracy")

    # Combined 2x2 grid of the same metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Training Metrics", fontsize=14)
    metric_info = [
        ("loss", "Loss", axes[0, 0]),
        ("dice", "Dice", axes[0, 1]),
        ("iou", "IoU", axes[1, 0]),
        ("pixel_acc", "Pixel Accuracy", axes[1, 1]),
    ]

    for metric, ylabel, ax in metric_info:
        train_vals = history.get(f"train_{metric}", [])
        val_vals = history.get(f"val_{metric}", [])
        if not train_vals:
            continue  # nothing to plot
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label="train")
        if val_vals and val_vals[0] is not None:
            ax.plot(epochs, val_vals, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} over epochs")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_out_path = os.path.join(out_dir, f"metrics_grid_{RUN_DETAIL}.png")
    fig.savefig(grid_out_path, bbox_inches="tight")
    plt.close(fig)
