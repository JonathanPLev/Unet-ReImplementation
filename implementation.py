import torch
import torch.nn as nn
import torchvision
from torchvision import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import sys
import os
import numpy as np
from PIL import Image

# we are working on the second segmentation task the U-net architecture tested on.

# i have an nvidia GPU at home. GTX1660Ti. with i5-10600k Intel CPU. and 1 16GB Ram stick.
# TODO: add cuda items so we can use the compute available to us for more rapid training.

NUM_WORKERS = 4
BATCH_SIZE = 1  # batch size detailed in the paper
FLIP_PROBABILITY = 0.5
MOMENTUM_TERM = 0.99  # detailed in paper
NUM_OUTPUT_CHANNELS = 2  # detailed in paper

# TODO: map the image with pixel-wise loss weight so it learns border images. formula 2
# TODO: implement weight initialization as detailed in the paper
# TODO: implement more data augmentation similar to the paper
"""We generate smooth
deformations using random displacement vectors on a coarse 3 by 3 grid. The
displacements are sampled from a Gaussian distribution with 10 pixels standard
deviation. Per-pixel displacements are then computed using bicubic interpolation."""


# online, recommended to train U-net using random crops of 256x256 for better generalization
# at inference time can still use arbitrary image size as long as divisble by 16.
def transforms(image, mask, crop_size=256):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.5], std=[0.5])

    image = F.resize(image, size=[512, 512], interpolation=InterpolationMode.BILINEAR)

    mask = (mask > 0).astype(
        np.uint8
    )  # because dataset also has set-up for multi class segmentation.
    mask = torch.tensor(
        mask, dtype=torch.long
    )  # convert to tensor + make correct dtype
    # not sure why this is an error for the interpreter.
    mask = F.resize(
        mask.unsqueeze(0), size=[512, 512], interpolation=InterpolationMode.NEAREST
    ).squeeze(0)  # expects 3 channels, then put back to 2-d dimensions
    i, j, h, w = F.get_params(image, output_size=(crop_size, crop_size))
    image = F.crop(image, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)

    if torch.rand(1) > 0.5:
        image = F.hflip(image)
        mask = F.hflip(mask)

    if torch.rand(1) > 0.5:
        image = F.vflip(image)
        mask = F.vflip(mask)
    return image, mask


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
        mask = np.array(Image.open(label_path))
        mask = (mask > 0).astype(np.uint8)  # binary segmentation

        label_filename = os.path.basename(label_path)
        base = label_filename.split(".")[0]
        frame_id = base.removeprefix("man_seg")

        image_path = os.path.join(self.image_root, image_folder, f"t{frame_id}.tif")

        image = np.array(Image.open(image_path))

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class Net(nn.Module):
    def __init__(self):
        # TODO: double check model architecture implementation (Jon)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0
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
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        # end of down sampling

        # up sampling 1
        x = self.up1(x)
        _, _, H, W = x.shape
        skip4_cropped = torchvision.transforms.CenterCrop([H, W])(skip4)
        x = torch.cat([x, skip4_cropped], dim=1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        # up sampling 2
        x = self.up2(x)
        _, _, H, W = x.shape
        skip3_cropped = torchvision.transforms.CenterCrop([H, W])(skip3)
        x = torch.cat([x, skip3_cropped], dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        # up sampling 3
        x = self.up3(x)
        _, _, H, W = x.shape
        skip2_cropped = torchvision.transforms.CenterCrop([H, W])(skip2)
        x = torch.cat([x, skip2_cropped], dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        # up sampling 4
        x = self.up4(x)
        _, _, H, W = x.shape
        skip1_cropped = torchvision.transforms.CenterCrop([H, W])(skip1)
        x = torch.cat([x, skip1_cropped], dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        # output layer
        x = self.out1(x)
        return x  # logits


# TODO: define train loop
def train_u_net(train_loader):
    net = Net()


# TODO: implement loss function
def loss_func(net):
    nn.CrossEntropyLoss()


if __name__ == "__main__":
    train_dataset = SegmentationDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )  # pin_memory = True, not necessary unless training on Cuda GPU

    train_u_net(train_loader)

    # TODO: create evaluations to see how the model does in training and then test.
    # Testing accuracy: Dice, IoU, Pixel accuracy
