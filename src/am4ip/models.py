import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

class Block(nn.Module):
    def __init__(self, ch_in, ch_out, bn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding="same"),
            nn.BatchNorm2d(ch_out) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, bn):
        super().__init__()
        self.block = nn.Sequential(
            Block(ch_in, ch_out, bn),
            Block(ch_out, ch_out, bn))
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.block(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpSample(nn.Module):
    def __init__(self, ch_in, ch_out, bn, bilinear):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(ch_in-ch_out, ch_in-ch_out, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            Block(ch_in, ch_out, bn),
            Block(ch_out, ch_out, bn))

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.block(x)


class UNet(nn.Module):
    def __init__(self,in_channel,  n_classes, batchnorm=False,  bilinear=False):
        super(UNet, self).__init__()

        self.down_conv1 = DownSample(in_channel, 64, batchnorm)
        self.down_conv2 = DownSample(64, 128, batchnorm)
        self.down_conv3 = DownSample(128, 256, batchnorm)
        self.down_conv4 = DownSample(256, 512, batchnorm)

        # Bottleneck
        self.double_conv = nn.Sequential(
                Block(512, 1024, batchnorm),
                Block(1024, 1024, batchnorm))
        
        self.up_conv4 = UpSample(512 + 1024, 512, batchnorm, bilinear)
        self.up_conv3 = UpSample(256 + 512, 256, batchnorm, bilinear)
        self.up_conv2 = UpSample(128 + 256, 128, batchnorm, bilinear)
        self.up_conv1 = UpSample(128 + 64, 64, batchnorm, bilinear)
        
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.out(x)
        return x
    