import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) module with adaptive kernel size selection.
    """
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = self._compute_kernel_size()
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, 
                              padding=(self.kernel_size - 1) // 2, 
                              bias=False)
        
    def _compute_kernel_size(self):
        k = int(abs((math.log(self.in_channel, 2) + 1) / 2))
        return k if k % 2 != 0 else k + 1
        
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, but got {x.dim()}D tensor.")
        
        B, C, H, W = x.shape
        if C != self.in_channel:
            raise ValueError(f"Channel mismatch: expected {self.in_channel}, got {C}.")

        y = x.mean(dim=(2, 3)) 
        y = y.unsqueeze(1) 
        y = self.conv(y) 
        y = torch.sigmoid(y)
        y = y.view(B, C, 1, 1)
        
        return x * y

class ChannelAttention(nn.Module):
    """
    Channel Attention Module optimizing structural identity cross-channels.
    """
    def __init__(self, in_channel, ratio=8):
        super().__init__()
        self.filters = max(1, in_channel // ratio)
        self.shared_layer = nn.Sequential(
            nn.Linear(in_channel, self.filters, bias=True),
            nn.ReLU(),
            nn.Linear(self.filters, in_channel, bias=True)
        )
        
    def forward(self, x):
        avg_pool = x.mean(dim=(2, 3)) 
        max_pool = x.amax(dim=(2, 3)) 
        
        avg_out = self.shared_layer(avg_pool)
        max_out = self.shared_layer(max_pool)
        
        y = torch.sigmoid(avg_out + max_out)
        return x * y.unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module focusing on breakpoint motifs and flanking context.
    """
    def __init__(self, kernel_size=(1, 5)):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2), 
                              bias=False)
        
    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True) 
        max_pool = x.amax(dim=1, keepdim=True) 
        concat = torch.cat([avg_pool, max_pool], dim=1) 
        y = torch.sigmoid(self.conv(concat))
        return x * y

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module sequentially combining channel and spatial scopes.
    """
    def __init__(self, in_channel, ratio=8, kernel_size=(1, 5)):
        super().__init__()
        self.channel_att = ChannelAttention(in_channel, ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution to decouple channel and spatial filters for computational efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class InsnetBackbone(nn.Module):
    """
    Insnet Backbone network optimized for feature extraction from long-read alignments.
    Transforms micro-scale motifs into fine-grained alignment representations.
    """
    def __init__(self):
        super().__init__()
        self.backbone_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 5), stride=1, padding=(1, 2)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            SeparableConv2d(128, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            CBAMBlock(64, ratio=7, kernel_size=(1, 5)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            ECALayer(64),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            ECALayer(64),
            
            SeparableConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            
            nn.Flatten()
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Expected shape [Batch_Size * Time_Steps, 1, 200, 5]
                              or [Batch_Size, 1, 200, 5].
        Returns:
            torch.Tensor: Flattened local features tensor with shape [Batch_Size, Dimensions].
        """
        return self.backbone_layers(x)

if __name__ == "__main__":
    # Local validation testing
    # Simulating a unified sequence alignment batch blocks [B * T, Channel, Height, Width]
    sample_tensor = torch.randn([10, 1, 200, 5]) 
    
    model = InsnetBackbone()
    output_tensor = model(sample_tensor)
    
    print("Insnet Backbone forward tracking verified successfully.")
    print("Output feature matrix shape:", output_tensor.shape)  # Should yield [10, 320]
