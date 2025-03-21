import torch
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(self, img_width=256, img_height=256, patch_size=16, in_chans=3, num_classes=19, 
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_width, img_height, patch_size, in_chans, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.down1 = TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.down2 = TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.down3 = TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.down4 = TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        
        self.up1 = TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.up2 = TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.up3 = TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)
        self.up4 = TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed  # add positional encoding

        x1 = self.down1(x)
        x2 = self.down2(x)
        x3 = self.down3(x)
        x4 = self.down4(x)

        x = self.up1(x)
        x = self.up2(x) + x3
        x = self.up3(x) + x2
        x = self.up4(x) + x1

        x = self.classifier(x)   # per-token classification

        H_p, W_p = H // self.patch_size, W // self.patch_size  # patch grid size
        x = x.transpose(1, 2).reshape(B, -1, H_p, W_p)  # reshape tokens to spatial map
        return torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # residual connection + attention
        return x + self.mlp(self.norm2(x))   # residual connection + MLP
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # A decoder block with self-attention and an MLP refinement
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_width=2048, img_height=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = img_width * img_height // (patch_size ** 2)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x): # (batch_size, in_channels, img_height, img_width)
        x = self.proj(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches_h * num_patches_w)
        return x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # extract batch size, number of features, dimensionality of features
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # split into query, key, and value

        attn = (q @ k.transpose(-2, -1)) * self.scale  # scaled dot-product attention
        attn = attn.softmax(dim=-1)  # softmax to get attention weights
        attn = self.attn_drop(attn)  # dropout to attention scores

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # attention weights to values
        x = self.proj(x)  # final projection
        return self.proj_drop(x) # dropout to the output

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)