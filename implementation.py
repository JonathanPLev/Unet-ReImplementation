import torch
import torch.nn as nn
import torchvision
from torchvision import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import cv2
import sys
import os
import numpy as np
from PIL import Image
import glob

# we are working on the second segmentation task the U-net architecture tested on.

# i have an nvidia GPU at home. GTX1660Ti. with i5-10600k Intel CPU. and 1 16GB Ram stick.
# TODO: add cuda items so we can use the compute available to us for more rapid training.

NUM_WORKERS = 4
BATCH_SIZE = 1  # batch size detailed in the paper
FLIP_PROBABILITY = 0.5
MOMENTUM_TERM = 0.99  # detailed in paper
NUM_OUTPUT_CHANNELS = 2  # detailed in paper

# Pixel-wise loss weight implemented in weighted_cross_entropy_loss function
# Weight initialization implemented in Net class
# Data augmentation: elastic deformations can be added later
"""We generate smooth
deformations using random displacement vectors on a coarse 3 by 3 grid. The
displacements are sampled from a Gaussian distribution with 10 pixels standard
deviation. Per-pixel displacements are then computed using bicubic interpolation."""


# online, recommended to train U-net using random crops of 256x256 for better generalization
# at inference time can still use arbitrary image size as long as divisble by 16.
def transforms(image, mask, crop_size=256):
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.5], std=[0.5])

    image = TF.resize(image, size=[512, 512], interpolation=InterpolationMode.BILINEAR)

    mask = (mask > 0).astype(
        np.uint8
    )  # because dataset also has set-up for multi class segmentation.
    mask = torch.tensor(
        mask, dtype=torch.long
    )  # convert to tensor + make correct dtype
    # not sure why this is an error for the interpreter.
    mask = TF.resize(
        mask.unsqueeze(0), size=[512, 512], interpolation=InterpolationMode.NEAREST
    ).squeeze(0)  # expects 3 channels, then put back to 2-d dimensions
    i, j, h, w = TF.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    if torch.rand(1) > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if torch.rand(1) > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
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
        super(Net, self).__init__()
        # U-Net architecture: padding=1 to maintain spatial dimensions after 3x3 conv
        # Downsampling path (encoder)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        )
        self.conv10 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1
        )

        # Max pooling for downsampling (2x2 with stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling path (decoder) - using transpose convolutions
        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, stride=2, kernel_size=2
        )
        self.conv11 = nn.Conv2d(
            in_channels=1024, out_channels=512, stride=1, kernel_size=3, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=1
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, stride=2, kernel_size=2
        )
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=256, stride=1, kernel_size=3, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=1
        )

        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, stride=2, kernel_size=2
        )
        self.conv15 = nn.Conv2d(
            in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1
        )
        self.conv16 = nn.Conv2d(
            in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1
        )

        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, stride=2, kernel_size=2
        )
        self.conv17 = nn.Conv2d(
            in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=1
        )
        self.conv18 = nn.Conv2d(
            in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1
        )

        # Output layer: 1x1 convolution to produce final segmentation map
        self.out1 = nn.Conv2d(
            in_channels=64,
            out_channels=NUM_OUTPUT_CHANNELS,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        
        # Initialize weights as per paper (He initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization as per paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Downsampling path (encoder)
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

        # Upsampling path (decoder) with skip connections
        # Up sampling 1
        x = self.up1(x)
        _, _, H, W = x.shape
        skip4_cropped = torchvision.transforms.CenterCrop([H, W])(skip4)
        x = torch.cat([x, skip4_cropped], dim=1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        # Up sampling 2
        x = self.up2(x)
        _, _, H, W = x.shape
        skip3_cropped = torchvision.transforms.CenterCrop([H, W])(skip3)
        x = torch.cat([x, skip3_cropped], dim=1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))

        # Up sampling 3
        x = self.up3(x)
        _, _, H, W = x.shape
        skip2_cropped = torchvision.transforms.CenterCrop([H, W])(skip2)
        x = torch.cat([x, skip2_cropped], dim=1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))

        # Up sampling 4
        x = self.up4(x)
        _, _, H, W = x.shape
        skip1_cropped = torchvision.transforms.CenterCrop([H, W])(skip1)
        x = torch.cat([x, skip1_cropped], dim=1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))

        # Output layer
        x = self.out1(x)
        return x  # logits


def compute_weight_map(mask, w0=10, sigma=5):
    """
    Compute pixel-wise weight map for border emphasis (Formula 2 in paper)
    w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2*sigma^2))
    where d_1, d_2 are distances to nearest and second nearest cell borders
    """
    # Convert mask to numpy for processing
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Binary mask: 1 for foreground, 0 for background
    binary_mask = (mask_np > 0).astype(np.float32)
    
    # Initialize weight map
    weight_map = np.ones_like(binary_mask, dtype=np.float32)
    
    # Find cell borders using distance transform
    # Distance to nearest border
    try:
        from scipy import ndimage
        dist_transform = ndimage.distance_transform_edt(binary_mask)
        dist_transform_inv = ndimage.distance_transform_edt(1 - binary_mask)
    except ImportError:
        # Fallback if scipy is not available - use simplified version
        import cv2
        dist_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
        dist_transform_inv = cv2.distanceTransform((1 - binary_mask).astype(np.uint8), cv2.DIST_L2, 5)
    
    # Combined distance (simplified version - full implementation would compute d1+d2)
    combined_dist = dist_transform + dist_transform_inv
    
    # Add border weight term
    border_weight = w0 * np.exp(-(combined_dist ** 2) / (2 * sigma ** 2))
    weight_map = weight_map + border_weight
    
    return torch.tensor(weight_map, dtype=torch.float32)


def weighted_cross_entropy_loss(predictions, targets, weight_map=None):
    """
    Weighted pixel-wise cross-entropy loss as per U-Net paper
    If weight_map is None, uses standard cross-entropy
    """
    if weight_map is None:
        # Standard cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, targets)
    else:
        # Weighted cross-entropy loss
        # Ensure weight_map has same spatial dimensions
        if weight_map.dim() == 2:
            weight_map = weight_map.unsqueeze(0)  # Add batch dimension
        if weight_map.dim() == 3:
            weight_map = weight_map.unsqueeze(1)  # Add channel dimension
        
        # Compute per-pixel cross-entropy
        log_probs = F.log_softmax(predictions, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
        
        # Weighted loss
        loss = -torch.sum(weight_map * targets_one_hot * log_probs, dim=1)
        return loss.mean()


def train_u_net(train_loader, num_epochs=100, learning_rate=0.00001, device='cpu', use_weight_map=True):
    """
    Training loop for U-Net
    Paper uses momentum=0.99, but typically SGD with momentum or Adam is used
    """
    net = Net()
    net = net.to(device)
    
    # Use Adam optimizer (more stable than SGD for this task)
    # Paper mentions momentum=0.99 but doesn't specify optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # Alternative: SGD with momentum as mentioned in paper
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=MOMENTUM_TERM)
    
    criterion = nn.CrossEntropyLoss()
    
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = net(images)
            
            # Compute loss
            if use_weight_map:
                # Compute weight map for border emphasis
                weight_maps = []
                for i in range(masks.shape[0]):
                    weight_map = compute_weight_map(masks[i])
                    weight_maps.append(weight_map)
                weight_map_batch = torch.stack(weight_maps).to(device)
                loss = weighted_cross_entropy_loss(outputs, masks, weight_map_batch)
            else:
                loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}')
    
    return net


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model using Dice coefficient, IoU, and pixel accuracy
    """
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Compute metrics for each sample in batch
            for i in range(predictions.shape[0]):
                pred = predictions[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # Dice coefficient
                intersection = np.logical_and(pred, mask).sum()
                dice = (2.0 * intersection) / (pred.sum() + mask.sum() + 1e-8)
                
                # IoU (Jaccard index)
                union = np.logical_or(pred, mask).sum()
                iou = intersection / (union + 1e-8)
                
                # Pixel accuracy
                pixel_acc = (pred == mask).sum() / mask.size
                
                total_dice += dice
                total_iou += iou
                total_pixel_acc += pixel_acc
    
    num_samples = len(test_loader.dataset)
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_pixel_acc = total_pixel_acc / num_samples
    
    print(f'Dice Coefficient: {avg_dice:.4f}')
    print(f'IoU (Jaccard Index): {avg_iou:.4f}')
    print(f'Pixel Accuracy: {avg_pixel_acc:.4f}')
    
    return avg_dice, avg_iou, avg_pixel_acc


if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset paths - adjust these based on your data structure
    # Assuming PhC-C2DH-U373.zip is extracted to a 'data' folder
    data_root = "data/PhC-C2DH-U373"
    
    # Find all mask files
    mask_paths_01 = glob.glob(os.path.join(data_root, "01_GT", "SEG", "man_seg*.tif"))
    mask_paths_02 = glob.glob(os.path.join(data_root, "02_GT", "SEG", "man_seg*.tif"))
    all_mask_paths = mask_paths_01 + mask_paths_02
    
    if len(all_mask_paths) == 0:
        print("Warning: No mask files found. Please check data paths.")
        print("Expected structure: data/PhC-C2DH-U373/01_GT/SEG/man_seg*.tif")
        print("Please unzip PhC-C2DH-U373.zip and organize data accordingly.")
        sys.exit(1)
    
    # Split into train and test (80/20 split)
    split_idx = int(0.8 * len(all_mask_paths))
    train_mask_paths = all_mask_paths[:split_idx]
    test_mask_paths = all_mask_paths[split_idx:]
    
    # Create datasets
    train_dataset = SegmentationDataset(
        image_root=os.path.join(data_root, "01"),  # Adjust if needed
        mask_paths=train_mask_paths,
        transforms=transforms
    )
    
    test_dataset = SegmentationDataset(
        image_root=os.path.join(data_root, "01"),  # Adjust if needed
        mask_paths=test_mask_paths,
        transforms=transforms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train the model
    print("Starting training...")
    trained_model = train_u_net(
        train_loader,
        num_epochs=100,
        learning_rate=0.00001,
        device=device,
        use_weight_map=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(trained_model, test_loader, device=device)
    
    # Save model
    torch.save(trained_model.state_dict(), 'unet_model.pth')
    print("Model saved to unet_model.pth")
