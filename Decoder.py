import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from Layers import DeTransformer


class Decoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,emb_dropout=0):
        super().__init__()
        assert image_size % patch_size == 0, '图片的维度必须被patch维度整除'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = DeTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim//2, dim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim//4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        # 图片分块处理
        _, c, h, w = img.shape
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # 降维处理
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # 分类编码
        # cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_token, x), dim=1)
        # 位置编码
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer计算
        x = self.transformer(x, mask)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        # x = self.mlp_head(x)
        # x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=h//p, w=w//p)
        return x