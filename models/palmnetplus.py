import torch.nn as nn
from .multiscale_conv import MultiScaleConv
from .swin_block import SwinTransformerBlock
from .detail_refine import DetailRefine

class PalmNetPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = MultiScaleConv(1, 32)
        self.enc2 = MultiScaleConv(96, 64)
        self.enc3 = MultiScaleConv(192, 128)
        self.pool = nn.MaxPool2d(2)
        self.swin = SwinTransformerBlock(384)
        self.up1 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.dec1 = MultiScaleConv(384, 64)
        self.up2 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.dec2 = MultiScaleConv(192, 32)
        self.refine = DetailRefine(96)
        self.out = nn.Conv2d(96, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        s = self.swin(e3)
        d1 = self.dec1(torch.cat([self.up1(s), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], 1))
        return self.out(self.refine(d2)).sigmoid()
