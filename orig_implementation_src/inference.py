"""
Inference script for the original implementation.
This is an example of what the model would output if it did inference on one .tiff image from the PhC dataset.


"""

from pathlib import Path
from math import ceil
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as TF
from PIL import Image

from config import DEVICE
from model import Net

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("Using GPU: ", torch.cuda.get_device_name(0))

PROJECT_ROOT = Path(__file__).parent.parent #Unet-Reimplementation 
EXAMPLE_IMAGE = PROJECT_ROOT / "PhC-C2DH-U373/01/t010.tif" # does not appear in training
MODEL_PATH = PROJECT_ROOT / "orig_impl_checkpoints/best_model_original_implementation.pth"
SAVE_PATH = PROJECT_ROOT / "orig_implementation_src/inference_example/inference_example.png"


def _load_model(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    net = Net()
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    return net


model = _load_model(MODEL_PATH)

def _pad_to_multiple(value, multiple):
    return int(ceil(value / multiple) * multiple)


def preprocess(image, multiple=16):
    image = image.convert("L")
    tensor = TF.ToTensor()(image)

    _, h, w = tensor.shape
    target_w = _pad_to_multiple(w, multiple)
    target_h = _pad_to_multiple(h, multiple)
    pad_w = target_w - w
    pad_h = target_h - h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if pad_w > 0 or pad_h > 0:
        tensor = F.pad(
            tensor.unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="reflect",
        ).squeeze(0)

    tensor = TF.Normalize(mean=[0.5], std=[0.5])(tensor)
    meta = {
        "orig_size": (h, w),
        "padded_size": (target_h, target_w),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
    }
    return tensor, meta


def forward(image, multiple=16):
    tensor, meta = preprocess(image, multiple=multiple)
    batch = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
    logits = logits.squeeze(0).cpu()
    return logits, meta


def logits_to_mask(logits):
    probs = F.softmax(logits, dim=0)
    pred = torch.argmax(probs, dim=0)
    return pred.numpy().astype(np.uint8)


def resize_to_original(mask, meta):
    padded_h, padded_w = meta["padded_size"]
    mask_h, mask_w = mask.shape
    canvas = np.zeros((padded_h, padded_w), dtype=mask.dtype)
    offset_h = max((padded_h - mask_h) // 2, 0)
    offset_w = max((padded_w - mask_w) // 2, 0)
    canvas[offset_h : offset_h + mask_h, offset_w : offset_w + mask_w] = mask

    top, left = meta["padding"][1], meta["padding"][0]
    bottom = padded_h - meta["padding"][3]
    right = padded_w - meta["padding"][2]
    cropped = canvas[top:bottom, left:right]
    orig_h, orig_w = meta["orig_size"]
    return cropped[:orig_h, :orig_w]


def visualize_prediction(image_path, save_path=None):
    image = Image.open(image_path)
    logits, meta = forward(image)
    mask_small = logits_to_mask(logits)
    mask = resize_to_original(mask_small, meta)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask_small, cmap="gray")
    axes[1].set_title(f"Logit Output ({mask_small.shape[1]}x{mask_small.shape[0]})")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, alpha=0.4, cmap="inferno")
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    directory = os.path.dirname(SAVE_PATH)
    os.makedirs(directory, exist_ok=True)
    visualize_prediction(EXAMPLE_IMAGE, SAVE_PATH)
