import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride,instanceNorm2d):
        super().__init__()

        if instanceNorm2d==False:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        layers=[]

        layers.append(Block(in_channels, 64, stride=2, instanceNorm2d=False)) #C64
        layers.append(Block(64, 128, stride=2, instanceNorm2d=True)) #C128
        layers.append(Block(128, 256, stride=2, instanceNorm2d=True)) #C256
        layers.append(Block(256, 512, stride=1, instanceNorm2d=True)) #C512
        layers.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")) #1-D output conv

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return torch.sigmoid(self.model(x))


