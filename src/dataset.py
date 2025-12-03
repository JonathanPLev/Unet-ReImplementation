import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from elasticdeform import deform_random_grid

from config import (
    CROP_SIZE,
    DEFORM_POINTS,
    DEFORM_SIGMA,
    FLIP_PROBABILITY,
    WEIGHT_MAP_CACHE_DIR,
)
from weight_map import compute_unet_weight_map


def transforms(image, instance_mask, weight_map=None, crop_size=CROP_SIZE):
    image = TF.to_tensor(image)  # (C,H,W)
    instance_mask = torch.as_tensor(instance_mask, dtype=torch.long)
    if weight_map is None:
        weight_map = compute_unet_weight_map(instance_mask.numpy())
    weight_map = torch.as_tensor(weight_map, dtype=torch.float32)

    pad_w = max(0, crop_size - image.shape[2])
    pad_h = max(0, crop_size - image.shape[1])
    padding = (
        pad_w // 2,
        pad_h // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2,
    )
    image = TF.pad(image, padding, padding_mode="reflect")
    instance_mask = TF.pad(
        instance_mask.unsqueeze(0), padding, fill=0, padding_mode="constant"
    ).squeeze(0)
    weight_map = TF.pad(
        weight_map.unsqueeze(0), padding, fill=0, padding_mode="constant"
    ).squeeze(0)

    i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    instance_mask = TF.crop(instance_mask.unsqueeze(0), i, j, h, w).squeeze(0)
    weight_map = TF.crop(weight_map.unsqueeze(0), i, j, h, w).squeeze(0)

    if torch.rand(1) > FLIP_PROBABILITY:
        image = TF.hflip(image)
        instance_mask = TF.hflip(instance_mask.unsqueeze(0)).squeeze(0)
        weight_map = TF.hflip(weight_map.unsqueeze(0)).squeeze(0)

    if torch.rand(1) > FLIP_PROBABILITY:
        image = TF.vflip(image)
        instance_mask = TF.vflip(instance_mask.unsqueeze(0)).squeeze(0)
        weight_map = TF.vflip(weight_map.unsqueeze(0)).squeeze(0)

    if torch.rand(1) < FLIP_PROBABILITY:
        img_np = image.numpy()
        inst_np = instance_mask.numpy()
        weight_np = weight_map.numpy()
        img_def, inst_def, weight_def = deform_random_grid(
            [img_np, inst_np, weight_np],
            sigma=DEFORM_SIGMA,
            points=DEFORM_POINTS,
            order=[
                3,
                0,
                1,
            ],  # bicubic for image, nearest for mask, bilinear for weights
            mode=["reflect", "constant", "reflect"],
            axis=[(1, 2), (0, 1), (0, 1)],
        )
        image = torch.from_numpy(img_def).float()
        instance_mask = torch.from_numpy(inst_def).long()
        weight_map = torch.from_numpy(weight_def).float()

    image = TF.normalize(image, mean=[0.5], std=[0.5])
    mask = (instance_mask > 0).long()

    return image, mask, weight_map


class SegmentationDataset(Dataset):
    def __init__(
        self, image_root, mask_paths, transforms, weight_cache_dir=WEIGHT_MAP_CACHE_DIR
    ):
        self.image_root = image_root
        self.mask_paths = mask_paths
        self.transforms = transforms
        self.weight_cache_dir = weight_cache_dir

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        label_path = self.mask_paths[idx]
        if "01_GT" in label_path:
            image_folder = "01"
        else:
            image_folder = "02"
        instance_mask = np.array(Image.open(label_path))

        label_filename = os.path.basename(label_path)
        base = label_filename.split(".")[0]
        frame_id = base.removeprefix("man_seg")

        image_path = os.path.join(self.image_root, image_folder, f"t{frame_id}.tif")

        image = np.array(Image.open(image_path).convert("L"))

        cache_rel = os.path.relpath(label_path, self.image_root)
        cache_path = os.path.join(
            self.weight_cache_dir, os.path.splitext(cache_rel)[0] + ".npy"
        )
        weight_map = compute_unet_weight_map(instance_mask, cache_path=cache_path)

        if self.transforms:
            image, mask, weight_map = self.transforms(image, instance_mask, weight_map)
        else:
            mask = (instance_mask > 0).astype(np.uint8)

        return image, mask, weight_map


class DataScienceBowlDataset(Dataset):
    """
    Dataset for the Data Science Bowl 2018 nuclei segmentation task.
    Each item corresponds to a single nucleus mask file paired with its image.
    Expects root/<image_id>/images/<image_id>.png and root/<image_id>/masks/*.png
    """

    def __init__(
        self, samples, transforms, weight_cache_dir=WEIGHT_MAP_CACHE_DIR
    ):
        self.samples = samples  # list of (image_path, mask_path)
        self.transforms = transforms
        self.weight_cache_dir = Path(weight_cache_dir) / "dsb2018"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary_mask = (mask > 0).astype(np.uint8)

        cache_rel = Path(mask_path).relative_to(Path(image_path).parents[1])
        cache_path = self.weight_cache_dir / cache_rel.with_suffix(".npy")
        weight_map = compute_unet_weight_map(binary_mask, cache_path=str(cache_path))

        if self.transforms:
            image, mask_out, weight_map = self.transforms(
                image, binary_mask, weight_map
            )
        else:
            mask_out = binary_mask

        return image, mask_out, weight_map
