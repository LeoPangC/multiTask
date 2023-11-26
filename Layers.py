import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, q, k, v, mask=None):
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out, attn


class AttentionLayer(nn.Module):
    def __init__(self, dim, dim_head, heads, dropout):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        self.attention = MultiAttention(dim)
        self.to_qury = nn.Linear(dim, inner_dim, bias=False)
        self.to_key = nn.Linear(dim, inner_dim, bias=False)
        self.to_value = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, en_output=None, mask=None):
        b, n, _ = x.shape
        h = self.heads
        if en_output is not None:
            key = en_output
            value = en_output
        else:
            key = self.to_key(x).view(b, h, n, -1)
            value = self.to_value(x).view(b, h, n, -1)
        qury = self.to_qury(x).view(b, h, n, -1)

        out, attn = self.attention(qury, key, value, mask)
        out = self.to_out(out)
        return out, attn


class EnTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # self.layers = nn.ModuleList([])
        self.depth = depth
        self.attention = AttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffd = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            # nn.GELU(),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         Residual(PreNorm(dim, AttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
        #         Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
        #     ]))

    def forward(self, x, en_output=None, mask=None):
        for _ in range(self.depth):
            res = x
            x = self.norm(x)
            x, attn = self.attention(x, en_output, mask)
            x = res + x
            res = x
            x = self.norm(x)
            x = self.ffd(x) + res
        # for attn, ff in self.layers:
        #     x = attn(x, mask=mask)
        #     x = ff(x)
        return x, attn


class DeTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        # self.layers = nn.ModuleList([])
        self.depth = depth
        self.attention = AttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffd = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            # nn.GELU(),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         Residual(PreNorm(dim, AttentionLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
        #         Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
        #     ]))

    def forward(self, x, mask=None):
        for _ in range(self.depth):
            res = x
            x = self.norm(x)
            x, attn = self.attention(x, mask=mask)
            x = res + x
            res = x
            x = self.norm(x)
            x = self.ffd(x) + res
        return x


if __name__ == '__main__':
    print(os.path.exists('./save/transformer'))