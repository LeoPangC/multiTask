import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Layers import EnTransformer
from Decoder import Decoder


class Encoder(nn.Module):
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

        self.transformer = EnTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
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
        x, attn = self.transformer(x, mask)


        x = self.mlp_head(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=h//p, w=w//p)
        return x


if __name__ == '__main__':
    # encoder = Encoder(image_size=180,
    #                   patch_size=30,
    #                   dim=1024,
    #                   depth=2,
    #                   heads=8,
    #                   mlp_dim=2048,
    #                   channels=2,
    #                   dim_head=64,
    #                   dropout=0.,
    #                   emb_dropout=0)
    decoder = Decoder(image_size=180,
                      patch_size=30,
                      dim=1024,
                      depth=2,
                      heads=8,
                      mlp_dim=2048,
                      channels=2,
                      dim_head=36,
                      dropout=0.,
                      emb_dropout=0)
    u10 = np.load('u10.npy')[:10]
    u10 = np.expand_dims(u10, axis=1)
    u10 = np.concatenate((u10, u10), axis=1)
    u10 = torch.as_tensor(u10, dtype=torch.float32)
    # attn = encoder(u10)
    x = decoder(u10)
    print(x)