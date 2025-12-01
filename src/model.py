
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class Net(nn.Module):
    def __init__(self):
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