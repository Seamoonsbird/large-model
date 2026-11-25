"""课程第 3 讲 CNN 与第 5 讲 ViT 的轻量实现。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, width=1.0, widths=None):
        super().__init__()
        if widths is None:
            widths = (int(32 * width), int(64 * width), int(128 * width))
        c1, c2, c3 = widths
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch=4, dim=64):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                 # B, dim, H/p, W/p
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, num_classes=10, patch=4, dim=64, depth=4, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, dim)
        n_tokens = (28 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model, dtype_bytes=4):
    return count_params(model) * dtype_bytes / 1024 / 1024
