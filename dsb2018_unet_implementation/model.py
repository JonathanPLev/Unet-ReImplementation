import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from config import NUM_OUTPUT_CHANNELS


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.conv10 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(1024)
        self.bn10 = nn.BatchNorm2d(1024)

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # up convolution
        self.up1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, stride=2, kernel_size=2, bias=False
        )
        self.conv11 = nn.Conv2d(
            in_channels=1024, out_channels=512, stride=1, kernel_size=3, padding=0, bias=False
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, stride=1, kernel_size=3, padding=0, bias=False
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, stride=2, kernel_size=2, bias=False
        )
        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=256, stride=1, kernel_size=3, padding=0, bias=False
        )
        self.conv14 = nn.Conv2d(
            in_channels=256, out_channels=256, stride=1, kernel_size=3, padding=0, bias=False
        )

        self.up3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, stride=2, kernel_size=2, bias=False
        )
        self.conv15 = nn.Conv2d(
            in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=0, bias=False
        )
        self.conv16 = nn.Conv2d(
            in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=0, bias=False
        )

        self.up4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, stride=2, kernel_size=2, bias=False
        )
        self.conv17 = nn.Conv2d(
            in_channels=128, out_channels=64, stride=1, kernel_size=3, padding=0, bias=False
        )
        self.conv18 = nn.Conv2d(
            in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=0, bias=False
        )

        self.bn11 = nn.BatchNorm2d(512)
        self.bn12 = nn.BatchNorm2d(512)
        self.bn13 = nn.BatchNorm2d(256)
        self.bn14 = nn.BatchNorm2d(256)
        self.bn15 = nn.BatchNorm2d(128)
        self.bn16 = nn.BatchNorm2d(128)
        self.bn17 = nn.BatchNorm2d(64)
        self.bn18 = nn.BatchNorm2d(64)

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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip1 = x
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        skip2 = x
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        skip3 = x
        x = self.pool(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        skip4 = x
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.dropout(x)
        # end of down sampling

        # up sampling 1
        x = self.up1(x)
        _, _, H, W = x.shape
        skip4_cropped = TF.center_crop(skip4, [H, W])
        x = torch.cat([x, skip4_cropped], dim=1)
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))

        # up sampling 2
        x = self.up2(x)
        _, _, H, W = x.shape
        skip3_cropped = TF.center_crop(skip3, [H, W])
        x = torch.cat([x, skip3_cropped], dim=1)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))

        # up sampling 3
        x = self.up3(x)
        _, _, H, W = x.shape
        skip2_cropped = TF.center_crop(skip2, [H, W])
        x = torch.cat([x, skip2_cropped], dim=1)
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))

        # up sampling 4
        x = self.up4(x)
        _, _, H, W = x.shape
        skip1_cropped = TF.center_crop(skip1, [H, W])
        x = torch.cat([x, skip1_cropped], dim=1)
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))

        # output layer
        x = self.out1(x)
        return x  # logits
