
import torch
import torch.nn as nn
from einops import rearrange

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x) 

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x, _ = self.attn(x2, x2, x2)
        x = x + x2
        x2 = self.norm2(x)
        return x + self.ff(x2)

class CCTrans(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_dim=256, depth=4):
        super().__init__()
        self.encoder = EncoderCNN()

        self.patch_proj = nn.Conv2d(256, emb_dim, kernel_size=1)
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_dim))

        self.transformer = nn.Sequential(*[
            TransformerBlock(emb_dim) for _ in range(depth)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)  
        feat = self.patch_proj(feat)  
      
        B, C, H, W = feat.shape
        patches = rearrange(feat, 'b c h w -> b (h w) c')  

        patches = patches + self.pos_embedding[:patches.size(1)]

        patches = self.transformer(patches) 

        feat = rearrange(patches, 'b (h w) c -> b c h w', h=H, w=W)

        out = self.decoder(feat)  
        return out
