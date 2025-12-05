# test code from chat
from pathlib import Path
import numpy as np
from PIL import Image

from config import DSB2018_TRAIN_ROOT, WEIGHT_MAP_CACHE_DIR
from dataset import build_instance_mask
from weight_map import compute_unet_weight_map


def main():
    train_root = Path(DSB2018_TRAIN_ROOT)
    cache_root = Path(WEIGHT_MAP_CACHE_DIR) / "dsb2018"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Pick the first sample directory
    sample_dir = next(train_root.iterdir())
    print("Testing sample:", sample_dir.name)

    # Paths
    image_path = sample_dir / "images" / (sample_dir.name + ".png")
    masks_dir = sample_dir / "masks"

    print("Image path:", image_path)
    print("Masks dir:", masks_dir)

    # Load image (not needed for cache, but good for debugging)
    img = np.array(Image.open(image_path).convert("L"))
    print("Image shape:", img.shape)

    # Build instance mask from all masks in directory
    instance_mask = build_instance_mask(masks_dir)
    print("Instance mask shape:", instance_mask.shape)
    print("Unique IDs:", np.unique(instance_mask)[:20], "...")

    # Cache file path (ONE FILE PER IMAGE)
    cache_path = cache_root / f"{sample_dir.name}.npy"
    print("Expected cache file:", cache_path)

    # Compute or load from cache
    print("Computing or loading weight map...")
    weight_map = compute_unet_weight_map(instance_mask, cache_path=str(cache_path))

    print("Weight map shape:", weight_map.shape)
    print("Min/max values:", float(weight_map.min()), float(weight_map.max()))
    print("Cache exists now?", cache_path.exists())

    # Load again to confirm instant cache hit
    print("Testing cache reload (should be instant)...")
    weight_map_2 = compute_unet_weight_map(instance_mask, cache_path=str(cache_path))
    print("Reload successful:", weight_map_2.shape == weight_map.shape)


if __name__ == "__main__":
    main()