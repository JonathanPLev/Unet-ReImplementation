
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from config import *
from model import Net
from utils import dice_score, IoU_score, pixel_accuracy, plot_history
import os

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
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    best_model_path = f"{MODEL_SAVE_PATH}/best_model_{RUN_DETAIL}.pth"

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