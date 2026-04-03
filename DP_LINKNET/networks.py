"""
DP-LinkNet network architectures for document binarization.

Defines LinkNet34, DLinkNet34, and DPLinkNet34 with a ResNet-34 encoder.
Self-contained; no dependency on the DP-LinkNet repository source code.

The architectures follow the beargolden/DP-LinkNet repository structure:
  - LinkNet34:   encoder-decoder with skip connections
  - DLinkNet34:  adds a dilated convolution center block (Dblock)
  - DPLinkNet34: adds both dilated convolutions and spatial pyramid pooling

All three produce a single-channel sigmoid output (per-pixel foreground
probability) suitable for document binarization.

NOTE: The layer names in these classes must match the keys in the trained
checkpoint.  If load_state_dict() reports unexpected or missing keys, the
most likely cause is a naming mismatch in the center block.  Adjust the
attribute names here to match the checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """Decoder block: 1x1 conv -> transposed conv (stride 2) -> 1x1 conv."""

    def __init__(self, in_channels, n_filters):
        super().__init__()
        mid = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.norm1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            mid, mid, 3, stride=2, padding=1, output_padding=1,
        )
        self.norm2 = nn.BatchNorm2d(mid)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        return x


class Dblock(nn.Module):
    """Cascaded dilated convolution block (rates 1, 2, 4) with residual
    summation, used as the center block in D-LinkNet and DP-LinkNet."""

    def __init__(self, channel):
        super().__init__()
        self.dilate1 = nn.Conv2d(channel, channel, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, 3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, 3, dilation=4, padding=4)

    def forward(self, x):
        d1 = F.relu(self.dilate1(x))
        d2 = F.relu(self.dilate2(d1))
        d3 = F.relu(self.dilate3(d2))
        return x + d1 + d2 + d3


class SPPblock(nn.Module):
    """Spatial Pyramid Pooling block.  Pools at three scales, reduces each to
    one channel, upsamples back, and concatenates with the input.

    Output channels = in_channels + 3.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.size()
        layer1 = F.interpolate(
            self.conv(self.pool1(x)), size=(h, w),
            mode="bilinear", align_corners=True,
        )
        layer2 = F.interpolate(
            self.conv(self.pool2(x)), size=(h, w),
            mode="bilinear", align_corners=True,
        )
        layer3 = F.interpolate(
            self.conv(self.pool3(x)), size=(h, w),
            mode="bilinear", align_corners=True,
        )
        return torch.cat([layer1, layer2, layer3, x], dim=1)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=None)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return torch.sigmoid(out)


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=None)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return torch.sigmoid(out)


class DPLinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=None)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(filters[3] + 3, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.spp(self.dblock(e4))

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return torch.sigmoid(out)


MODEL_REGISTRY = {
    "linknet34": LinkNet34,
    "dlinknet34": DLinkNet34,
    "dplinknet34": DPLinkNet34,
}
