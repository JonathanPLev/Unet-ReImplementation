"""
Precompute and cache weight maps for the Data Science Bowl 2018 masks.
Run before training to avoid paying the cost in the first epoch.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from config import DSB2018_TRAIN_ROOT, WEIGHT_MAP_CACHE_DIR
from weight_map import compute_unet_weight_map


def iter_mask_paths(train_root: Path):
    for image_dir in train_root.iterdir():
        if not image_dir.is_dir():
            continue
        masks_dir = image_dir / "masks"
        if not masks_dir.exists():
            continue
        for mask_path in sorted(masks_dir.glob("*.png")):
            yield image_dir, mask_path


def main():
    train_root = Path(DSB2018_TRAIN_ROOT)
    cache_root = Path(WEIGHT_MAP_CACHE_DIR) / "dsb2018"
    cache_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for total, (image_dir, mask_path) in enumerate(iter_mask_paths(train_root), start=1):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary_mask = (mask > 0).astype(np.uint8)

        cache_rel = mask_path.relative_to(image_dir).with_suffix(".npy")
        cache_path = cache_root / cache_rel
        compute_unet_weight_map(binary_mask, cache_path=str(cache_path))

        if total % 200 == 0:
            print(f"Cached {total} masks... latest: {cache_rel}")

    print(f"Done. Cached weight maps for {total} masks.")


if __name__ == "__main__":
    main()
