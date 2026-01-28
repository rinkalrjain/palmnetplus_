import torch.nn as nn
import torch.nn.functional as F

class SSIM(nn.Module):
    def forward(self, x, y):
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)
        sigma_x = F.avg_pool2d(x**2, 3, 1, 0) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, 0) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 0) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        return (1 - ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sigma_x+sigma_y+C2))).mean()
