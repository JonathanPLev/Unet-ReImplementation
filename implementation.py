import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from elasticdeform import deform_random_grid

# we are working on the second segmentation task the U-net architecture tested on.

# i have an nvidia GPU at home. GTX1660Ti. with i5-10600k Intel CPU. and 1 16GB Ram stick.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = (
        True  # optimize convolution algorithms for current GPU
    )
    print("Using GPU: ", torch.cuda.get_device_name(0))

IMAGE_ROOT = "PhC-C2DH-U373"
NUM_WORKERS = 4
BATCH_SIZE = 1  # batch size detailed in the paper
FLIP_PROBABILITY = 0.5
LEARNING_RATE = 0.01
MOMENTUM_TERM = 0.99  # detailed in paper
NUM_OUTPUT_CHANNELS = 2  # detailed in paper
EPOCHS = 100

# TODO: map the image with pixel-wise loss weight so it learns border images. formula 2
# TODO: implement weight initialization as detailed in the paper
# TODO: implement more data augmentation similar to the paper
"""We generate smooth
deformations using random displacement vectors on a coarse 3 by 3 grid. The
displacements are sampled from a Gaussian distribution with 10 pixels standard
deviation. Per-pixel displacements are then computed using bicubic interpolation."""


def compute_class_weight_map(mask, h, w):
    weight_class = np.zeros((h, w), dtype=np.float32)
    labels, counts = np.unique(mask, return_counts=True)
    freq = counts / counts.sum()  # frequency of each class
    class_weights = 1.0 / (freq + 1e-8)  # inverse frequency
    class_weights /= class_weights.mean()  # normalize

    for label, weight in zip(labels, class_weights):
        weight_class[mask == label] = weight  # each pixel gets weight of its class

    return weight_class


def compute_unet_weight_map(mask, cache_path=None, w0=10.0, sigma=5.0):
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)
    h, w = mask.shape

    weight_class = compute_class_weight_map(
        mask, h, w
    )  # inverse frequency weights over entire mask
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


def transforms(image, mask, weight_mask, crop_size=572):
    image = TF.to_tensor(image)  # (C,H,W)
    mask = torch.as_tensor(mask, dtype=torch.long)
    weight_mask = torch.as_tensor(weight_mask, dtype=torch.long)

    pad_w = max(0, crop_size - image.shape[2])
    pad_h = max(0, crop_size - image.shape[1])
    padding = (
        pad_w // 2,
        pad_h // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2,
    )
    image = TF.pad(image, padding, padding_mode="reflect")
    mask = TF.pad(mask.unsqueeze(0), padding, fill=0, padding_mode="constant").squeeze(
        0
    )
    weight_mask = TF.pad(
        weight_mask.unsqueeze(0), padding, fill=0, padding_mode="constant"
    ).squeeze(0)

    i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask.unsqueeze(0), i, j, h, w).squeeze(0)
    weight_mask = TF.crop(weight_mask.unsqueeze(0), i, j, h, w).squeeze(0)

    if torch.rand(1) > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        weight_mask = TF.hflip(weight_mask.unsqueeze(0)).squeeze(0)

    if torch.rand(1) > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        weight_mask = TF.vflip(weight_mask.unsqueeze(0)).squeeze(0)

    if torch.rand(1) < 0.5:
        img_np = image.numpy()
        mask_np = mask.numpy()
        weight_np = weight_mask.numpy()
        img_def, mask_def, weight_def = deform_random_grid(
            [img_np, mask_np, weight_np],
            sigma=10,
            points=3,
            order=[3, 0, 0],  # bicubic for image, nearest for masks
            mode=["reflect", "constant", "constant"],
            axis=[(1, 2), (0, 1), (0, 1)],
        )
        image = torch.from_numpy(img_def).float()
        mask = torch.from_numpy(mask_def).long()
        weight_mask = torch.from_numpy(weight_def).long()

    image = TF.normalize(image, mean=[0.5], std=[0.5])
    weight_map = torch.from_numpy(compute_unet_weight_map(weight_mask.numpy())).float()

    return image, mask, weight_map


class SegmentationDataset(Dataset):
    def __init__(self, image_root, mask_paths, transforms):
        self.image_root = image_root
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        label_path = self.mask_paths[idx]
        if "01_GT" in label_path:
            image_folder = "01"
        else:
            image_folder = "02"
        weight_mask = np.array(Image.open(label_path))
        mask = (weight_mask > 0).astype(np.uint8)  # binary segmentation

        label_filename = os.path.basename(label_path)
        base = label_filename.split(".")[0]
        frame_id = base.removeprefix("man_seg")

        image_path = os.path.join(self.image_root, image_folder, f"t{frame_id}.tif")

        image = np.array(Image.open(image_path).convert("L"))

        if self.transforms:
            image, mask, weight_map = self.transforms(image, mask, weight_mask)

        return image, mask, weight_map


class Net(nn.Module):
    def __init__(self):
        # TODO: double check model architecture implementation (Jon)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0
        )
        self.conv8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0
        )
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0
        )
        self.conv10 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0
        )

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # up convolution
        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, stride=2, kernel_size=2
        )
        self.conv11 = nn.Conv2d(
            in_channels=1024, out_channels=512, stride=1, kernel_size=3, padding=0
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=0
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, stride=2, kernel_size=2
        )
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=256, stride=1, kernel_size=3, padding=0
        )
        self.conv14 = nn.Conv2d(
            in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=0
        )

        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, stride=2, kernel_size=2
        )
        self.conv15 = nn.Conv2d(
            in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=0
        )
        self.conv16 = nn.Conv2d(
            in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=0
        )

        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, stride=2, kernel_size=2
        )
        self.conv17 = nn.Conv2d(
            in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=0
        )
        self.conv18 = nn.Conv2d(
            in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0
        )

        self.out1 = nn.Conv2d(
            in_channels=64,
            out_channels=NUM_OUTPUT_CHANNELS,
            stride=1,
            kernel_size=1,
            padding=0,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # TODO: implement model
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip1 = x
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        skip2 = x
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        skip3 = x
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        skip4 = x
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.dropout(x)
        # end of down sampling

        # up sampling 1
        x = self.up1(x)
        _, _, H, W = x.shape
        skip4_cropped = TF.center_crop(skip4, [H, W])
        x = torch.cat([x, skip4_cropped], dim=1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        # up sampling 2
        x = self.up2(x)
        _, _, H, W = x.shape
        skip3_cropped = TF.center_crop(skip3, [H, W])
        x = torch.cat([x, skip3_cropped], dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        # up sampling 3
        x = self.up3(x)
        _, _, H, W = x.shape
        skip2_cropped = TF.center_crop(skip2, [H, W])
        x = torch.cat([x, skip2_cropped], dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        # up sampling 4
        x = self.up4(x)
        _, _, H, W = x.shape
        skip1_cropped = TF.center_crop(skip1, [H, W])
        x = torch.cat([x, skip1_cropped], dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        # output layer
        x = self.out1(x)
        return x  # logits


def train_u_net(train_loader, val_loader=None):
    net = Net().to(DEVICE)

    criterion = nn.CrossEntropyLoss(reduction="none")
    weights = [p for name, p in net.named_parameters() if "weight" in name]
    biases = [p for name, p in net.named_parameters() if "bias" in name]

    optimizer = torch.optim.SGD(
        [
            {
                "params": weights,
                "weight_decay": 0.0005,
            },  # Apply weight decay to weights
            {"params": biases, "weight_decay": 0},  # No weight decay for biases
        ],
        lr=LEARNING_RATE,
        momentum=MOMENTUM_TERM,
    )

    best_val_dice = -1.0
    best_epoch = -1
    best_model_path = "best_model.pth"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_pixel_acc": [],
        "val_pixel_acc": [],
    }

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        running_pixel_acc = 0.0
        for images, masks, weight_maps in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            weight_maps = weight_maps.to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad()
            outputs = net(images)
            _, _, out_h, out_w = outputs.shape
            if masks.shape[-2:] != (out_h, out_w):
                masks = TF.center_crop(masks.unsqueeze(1), (out_h, out_w)).squeeze(1)
            if weight_maps.shape[-2:] != (out_h, out_w):
                weight_maps = TF.center_crop(
                    weight_maps.unsqueeze(1), (out_h, out_w)
                ).squeeze(1)

            loss_map = criterion(outputs, masks)
            loss = (loss_map * weight_maps).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_score(outputs.detach(), masks)
            running_iou += IoU_score(outputs.detach(), masks)
            running_pixel_acc += pixel_accuracy(outputs.detach(), masks)

        train_loss = running_loss / max(len(train_loader), 1)
        train_dice = running_dice / max(len(train_loader), 1)
        train_iou = running_iou / max(len(train_loader), 1)
        train_pixel_acc = running_pixel_acc / max(len(train_loader), 1)

        val_loss = None
        val_dice = None
        val_iou = None
        val_pixel_acc = None
        if val_loader is not None:
            net.eval()
            v_loss = 0.0
            v_dice = 0.0
            v_iou = 0.0
            v_pixel_acc = 0.0
            with torch.no_grad():
                for images, masks, weight_maps in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    masks = masks.to(DEVICE, non_blocking=True)
                    weight_maps = weight_maps.to(DEVICE, non_blocking=True).float()
                    outputs = net(images)
                    _, _, out_h, out_w = outputs.shape
                    if masks.shape[-2:] != (out_h, out_w):
                        masks = TF.center_crop(
                            masks.unsqueeze(1), (out_h, out_w)
                        ).squeeze(1)
                    if weight_maps.shape[-2:] != (out_h, out_w):
                        weight_maps = TF.center_crop(
                            weight_maps.unsqueeze(1), (out_h, out_w)
                        ).squeeze(1)

                    loss_map = criterion(outputs, masks)
                    loss = (loss_map * weight_maps).mean()
                    v_loss += loss.item()
                    v_dice += dice_score(outputs, masks)
                    v_iou += IoU_score(outputs, masks)
                    v_pixel_acc += pixel_accuracy(outputs, masks)
            val_loss = v_loss / max(len(val_loader), 1)
            val_dice = v_dice / max(len(val_loader), 1)
            val_iou = v_iou / max(len(val_loader), 1)
            val_pixel_acc = v_pixel_acc / max(len(val_loader), 1)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                torch.save(net.state_dict(), best_model_path)

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["train_pixel_acc"].append(train_pixel_acc)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_pixel_acc"].append(val_pixel_acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"train_iou={train_iou:.4f} train_pixacc={train_pixel_acc:.4f}"
            + (
                f" | val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
                f"val_iou={val_iou:.4f} val_pixacc={val_pixel_acc:.4f}"
                if val_loader is not None
                else ""
            )
        )

    if best_epoch >= 0:
        print(f"Best val_dice={best_val_dice:.4f} at epoch {best_epoch+1}")
        print(f"Saved best model to {best_model_path}")
    plot_history(history)


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
    union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()


def pixel_accuracy(pred, target, epsilon=1e-6):
    pred = pred.argmax(dim=1)
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=[1, 2])
    union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2])
    pixel_accuracy = (intersection + epsilon) / (union + epsilon)
    return pixel_accuracy.mean().item()


def plot_history(history, out_dir="plots"):
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
        out_path = os.path.join(out_dir, f"{metric}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    plot_pair("loss", "Loss")
    plot_pair("dice", "Dice")
    plot_pair("iou", "IoU")
    plot_pair("pixel_acc", "Pixel Accuracy")


if __name__ == "__main__":
    mask_pattern = os.path.join(IMAGE_ROOT, "*_GT", "SEG", "man_seg*.tif")
    mask_paths = sorted(glob.glob(mask_pattern))

    rng = np.random.default_rng(seed=42)
    shuffled = mask_paths.copy()
    rng.shuffle(shuffled)
    split_idx = max(1, int(0.2 * len(shuffled)))
    val_mask_paths = shuffled[:split_idx]
    train_mask_paths = shuffled[split_idx:]

    train_dataset = SegmentationDataset(
        image_root=IMAGE_ROOT,
        mask_paths=train_mask_paths,
        transforms=transforms,
    )
    val_dataset = SegmentationDataset(
        image_root=IMAGE_ROOT,
        mask_paths=val_mask_paths,
        transforms=transforms,
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
