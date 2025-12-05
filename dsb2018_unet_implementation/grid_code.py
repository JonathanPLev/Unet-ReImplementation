import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Directory containing your PNGs
PNG_DIR = "/home/sai/Documents/ECS174/Final Project/Resnet-ReImplementation/dsb2018_unet_implementation/plots/2018_train"

files = {
    "Loss": f"{PNG_DIR}/dsb2018_loss_dsb2018_optimized.png",
    "Dice": f"{PNG_DIR}/dsb2018_dice_dsb2018_optimized.png",
    "IoU": f"{PNG_DIR}/dsb2018_iou_dsb2018_optimized.png",
    "Pixel Acc": f"{PNG_DIR}/dsb2018_pixel_acc_dsb2018_optimized.png",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Training Metrics — DSB2018 Optimized U-Net", fontsize=16)

for ax, (title, fp) in zip(axes.flatten(), files.items()):
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing plot file: {fp}")
    img = mpimg.imread(fp)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = f"{PNG_DIR}/metrics_grid_dsb2018_optimized.png"
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print("Saved grid plot to:", out_path)
