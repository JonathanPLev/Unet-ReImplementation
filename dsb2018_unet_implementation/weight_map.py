import numpy as np
from scipy import ndimage as ndi
import os
import config


def compute_class_weight_map(mask, h, w):
    weight_class = np.zeros((h, w), dtype=np.float32)
    labels, counts = np.unique(mask, return_counts=True)
    freq = counts / counts.sum()  # frequency of each class
    class_weights = 1.0 / (freq + 1e-8)  # inverse frequency
    class_weights /= class_weights.mean()  # normalize

    for label, weight in zip(labels, class_weights):
        weight_class[mask == label] = weight  # each pixel gets weight of its class

    return weight_class


def compute_unet_weight_map(
    mask, cache_path=None, w0=config.WEIGHT_MAP_W0, sigma=config.WEIGHT_MAP_SIGMA
):
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)
    h, w = mask.shape
    binary_mask = (mask > 0).astype(np.uint8)
    weight_class = compute_class_weight_map(binary_mask, h, w)
    cell_ids = [
        cid for cid in np.unique(mask) if cid != 0
    ]  # collect all non-zero cell ids
    if len(cell_ids) < 2:
        return weight_class.astype(np.float32)

    # distance to each cell i (label)
    distance_maps = []
    for cid in cell_ids:
        cell_mask = mask == cid  # mask
        # distance of each pixel to specific cell, if the pixel is NOT in the cell
        distance = ndi.distance_transform_edt(~cell_mask)
        # array of distances to each cell
        distance_maps.append(distance)

    # each index, coordinate y, coordinate x id stance from a specific pixel to cell (index)
    distance_maps = np.stack(distance_maps, axis=0)

    # get nearest and second nearest cell border for d1 and d2 as defined in u-net formula.
    dist_sorted = np.sort(distance_maps, axis=0)
    d1 = dist_sorted[0]
    d2 = dist_sorted[1]

    # calculate border weight term
    border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * (sigma**2)))

    # weight map with border weight term applied
    w_map = weight_class + border
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, w_map)
    return w_map.astype(np.float32)  # final weight map
