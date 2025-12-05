import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DEVICE, IMAGE_ROOT, NUM_WORKERS, PIN_MEMORY
from dataset import SegmentationDataset, transforms as dataset_transforms
from train import train_u_net

# we are implementing the original U-net architecture for the PhC-C2DH-U373 dataset segmentation task.

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("Using GPU: ", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    mask_pattern = os.path.join(str(IMAGE_ROOT), "*_GT", "SEG", "man_seg*.tif")
    mask_paths = sorted(glob.glob(mask_pattern))

    rng = np.random.default_rng(seed=42)
    shuffled = mask_paths.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(0.2 * len(shuffled)))
    val_mask_paths = shuffled[:split_idx]
    train_mask_paths = shuffled[split_idx:]

    train_dataset = SegmentationDataset(
        image_root=str(IMAGE_ROOT),
        mask_paths=train_mask_paths,
        transforms=dataset_transforms,
    )
    val_dataset = SegmentationDataset(
        image_root=str(IMAGE_ROOT),
        mask_paths=val_mask_paths,
        transforms=dataset_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )

    train_u_net(train_loader, val_loader)
