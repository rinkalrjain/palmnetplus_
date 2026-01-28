import torch.nn as nn

class DetailRefine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        r = self.conv(x)
        return x + r * self.se(r)
