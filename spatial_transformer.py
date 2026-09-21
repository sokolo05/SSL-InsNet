import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class LocallyGroupedSelfAttention(nn.Module):
    """
    Locally-Grouped Self-Attention (LSA) targeting micro-scale breakpoint motifs 
    within highly partitioned non-overlapping window segments.
    """
    def __init__(self, dim, num_heads, window_size=(5, 1), attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size  
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size: tuple):
        """
        Args:
            x (torch.Tensor): Flattened spatial sequence [B, N, C].
            size (tuple): Actual spatial dimension topology (H, W) where H * W = N.
        """
        B, N, C = x.shape
        H, W = size
        
        h_group = H // self.window_size[0]
        w_group = W // self.window_size[1]

        # Reshape to 2D block coordinates and cluster into local sub-windows
        x = x.view(B, h_group, self.window_size[0], w_group, self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * h_group * w_group, self.window_size[0] * self.window_size[1], C)

        # Joint compute of Query, Key, and Value tensors
        qkv = self.qkv(x).view(B * h_group * w_group, self.window_size[0] * self.window_size[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Local Attention Matrix Evaluation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Reconstruct back to global sequence orientation
        x = (attn @ v).transpose(1, 2).reshape(B * h_group * w_group, self.window_size[0] * self.window_size[1], C)
        x = x.view(B, h_group, w_group, self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalSubsampledAttention(nn.Module):
    """
    Global Sub-sampled Attention (GSA) modeling regional genomic contexts 
    at a lower spatial resolution to manage computation.
    """
    def __init__(self, dim, num_heads, sr_ratio=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: tuple):
        B, N, C = x.shape
        H, W = size
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Shift sequence representation into a 2D matrix spatial layout for downsampling
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_spatial = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_spatial = self.norm(x_spatial)
            kv = self.kv(x_spatial).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1]

        # Global Structural Attention Matrix Mapping
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MlpBlock(nn.Module):
    """
    Standard multi-layer perceptron module applied identically over elements.
    """
    def __init__(self, dim, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialTransformerBlock(nn.Module):
    """
    Unified Twins-SVT block executing alternated execution profiles 
    of LSA (Local) and GSA (Global) networks.
    """
    def __init__(self, dim, num_heads, window_size=(5, 1), sr_ratio=1, attn_drop=0., proj_drop=0., mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lsa = LocallyGroupedSelfAttention(dim, num_heads, window_size, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = MlpBlock(dim, mlp_ratio, proj_drop)
        
        self.norm3 = nn.LayerNorm(dim)
        self.gsa = GlobalSubsampledAttention(dim, num_heads, sr_ratio, attn_drop, proj_drop)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = MlpBlock(dim, mlp_ratio, proj_drop)

    def forward(self, x, size: tuple):
        # 1. Micro-scale feature processing via Local Window View (LSA)
        x = x + self.lsa(self.norm1(x), size)
        x = x + self.mlp1(self.norm2(x))
        
        # 2. Regional macro-scale consensus construction via Downsampled View (GSA)
        x = x + self.gsa(self.norm3(x), size)
        x = x + self.mlp2(self.norm4(x))
        
        return x
    
class SpatialTransformer(nn.Module):
    """
    Spatial Transformer (S-Trans) Module embedded within the MSP system.
    Resolves complex long-read alignment variations via dual-granularity attention mechanism.
    """
    def __init__(self, input_dim=64, embed_dim=128, num_heads=8, window_size=(5, 1), sr_ratio=2, output_dim=320):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_block = SpatialTransformerBlock(
            dim=embed_dim, num_heads=num_heads, window_size=window_size, sr_ratio=sr_ratio
        )
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x, size: tuple):
        """
        Args:
            x (torch.Tensor): Extracted feature tokens tensor with shape [Batch_Size, Tokens_Num, Input_Dim].
            size (tuple): Downsampled spatial grid structure configurations (Height, Width).
        Returns:
            torch.Tensor: Refined spatial structural encoding representations with shape [Batch_Size, Output_Dim].
        """
        # Linear projection into attention feature dimensions
        x = self.input_embedding(x) 
        
        # Spatial dual-granularity sequence transformations
        x = self.transformer_block(x, size)

        # Global average pool over spatial nodes to extract structural features
        x = x.mean(dim=1) 
        
        # Map to unified framework channel projection sizing
        x = self.output_layer(x) 
        
        return x

