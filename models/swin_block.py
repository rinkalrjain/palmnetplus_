import torch.nn as nn
from einops import rearrange

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.attn(self.norm1(x), x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
