import torch
import torch.nn as nn
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)
        if torch.cuda.is_available():
            self.conv = self.conv.cuda()
            self.bn = self.bn.cuda()
            self.activate = self.activate.cuda()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class DarknetBottleneck(nn.Module):
    def __init__(self, in_channels):
        super(DarknetBottleneck, self).__init__()
        self.conv1 = ConvModule(in_channels, in_channels)
        self.conv2 = ConvModule(in_channels, in_channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x

class MyModel(nn.Module):
    def __init__(self, in_channels, num_bottlenecks=3):
        super(MyModel, self).__init__()
        self.bottlenecks = nn.ModuleList()
        for _ in range(num_bottlenecks):
            self.bottlenecks.append(DarknetBottleneck(in_channels))

    def forward(self, x):
        for bottleneck in self.bottlenecks:
            x = bottleneck(x)
        return x

    def __iter__(self):
        return iter(self.bottlenecks)