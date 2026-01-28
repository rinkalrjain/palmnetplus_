import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.conv7 = nn.Conv2d(in_ch, out_ch, 7, padding=3)
        self.bn = nn.BatchNorm2d(out_ch * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return self.relu(self.bn(x))
