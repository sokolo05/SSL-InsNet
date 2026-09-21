import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributedTransformer(nn.Module):
    """
    Time-distributed Transformer (T-Trans) Module.
    Captures macro-scale structural patterns across logical time-axis genomic sub-regions.
    """
    def __init__(self, dim_in=320, dim_n=320, num_heads=8, num_layers=1, dim_feedforward=1024, max_t=200, dropout=0.1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_n = dim_n
        
        # 1. Linear Projection: W_p and b_p 
        self.W_p = nn.Linear(dim_in, dim_n)
        
        # Learnable Positional Embedding P 
        self.P = nn.Parameter(torch.zeros(1, max_t, dim_n))
        nn.init.trunc_normal_(self.P, std=0.02)
        
        # 2. Transformer Encoder Layer
        # Note: norm_first=False implements the standard Post-LN architecture governed in the manuscript
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_n,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z):
        """
        Args:
            Z (torch.Tensor): Integrated sub-region sequence from CFI module. Shape: [B, T, dim_in]
        Returns:
            X_ctx (torch.Tensor): Finalized contextual representation. Shape: [B, T, dim_n]
        """
        B, T, C = Z.shape
        if C != self.dim_in:
            raise ValueError(f"Input channel dim {C} does not match expected dim_in {self.dim_in}")
            
        # ----------------------------------------------------
        # 1. Linear Projection and Temporal Embedding
        # ----------------------------------------------------
        # Equation (14): Z_proj = σ(Z * W_p + b_p) + P
        # Using ELU/GELU as activation σ following framework conventions
        Z_linear = F.elu(self.W_p(Z)) 
        Z_proj = Z_linear + self.P[:, :T, :]
        Z_proj = self.dropout(Z_proj)

        # ----------------------------------------------------
        # 2. Permutation and Transformer Encoding
        # ----------------------------------------------------
        # Permutation: (B, T, dim_n) -> (T, B, dim_n) for standard PyTorch sequential efficiency
        Z_perm = Z_proj.permute(1, 0, 2)
        
        #  Internal MHA and FFN loops with Post-LayerNorm residual pipeline
        H_enc = self.transformer_encoder(Z_perm) # Shape: [T, B, dim_n]

        # ----------------------------------------------------
        # 3. Inverse Permutation and Contextual Output
        # ----------------------------------------------------
        # Inverse Permutation: (T, B, dim_n) -> (B, T, dim_n)
        H_enc_batch_first = H_enc.permute(1, 0, 2)
        
        # Equation (17): X_ctx = Permute(H_enc) + Z_proj
        X_ctx = H_enc_batch_first + Z_proj

        return X_ctx

if __name__ == "__main__":
    # Local validation with standard dimensions (dim_in = dim_n = 320)
    batch_size = 4
    time_steps = 50 # Example sequence tracking length of sub-regions
    feature_dim = 320
    
    # Simulate tensor streams coming out of the CFI Module
    simulated_Z = torch.randn([batch_size, time_steps, feature_dim])
    
    t_trans_module = TimeDistributedTransformer(
        dim_in=feature_dim, 
        dim_n=320, 
        num_heads=8, 
        num_layers=2, # Stacked blocks optimization
        dim_feedforward=1024
    )
    
    X_ctx = t_trans_module(simulated_Z)
    
    print("T-Trans Module sequence processing verified successfully.")
    print("Contextual representation tensor output size (X_ctx):", X_ctx.shape) # Verified [4, 50, 320]
