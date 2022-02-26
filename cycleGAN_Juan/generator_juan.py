import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,padding, stride,use_act = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", kernel_size=kernel_size,padding=padding,stride=stride),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding,output_padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,kernel_size,stride,padding,output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1,stride=1),
            ConvBlock(channels, channels, kernel_size=3, padding=1,stride=1,use_act=False),
        )

    def forward(self, x):
        return x + self.block(x)



class Generator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()

        self.initial = nn.ModuleList(
            [
                ConvBlock(img_channels,out_channels=64,kernel_size=7,stride=1,padding=3)             #C7S1-k 
            ]
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),                              #d128
                ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),                             #d256
            ]
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
            ResidualBlock(256),                                                                      #d256
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),                  #u128
                UpBlock(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),                   #u64
            ]
        )

        self.last = nn.Conv2d(64, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect") #c7s1-3

    def forward(self, x):
        for layer in self.initial:
            x = layer(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


