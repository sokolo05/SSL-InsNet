import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings

from backbone_insnet import InsnetBackbone
from spatial_transformer import SpatialTransformer
from confidence_gated_fusion import ConfidenceGatedIntegration
from time_distributed_transformer import TimeDistributedTransformer
from NeighborhoodWeightedFocalLoss import NeighborhoodWeightedFocalLoss
from until import get_data_loader, EarlyStopping, train_fn, valid_fn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SSL_InsNet_model(nn.Module):
    def __init__(self, timesteps=100, d_model=320):
        super(SSL_InsNet_dodel, self).__init__()
        self.timesteps = timesteps
        self.d_model = d_model
        
        self.spatial_encoder = SpatialTransformer(
            input_dim=5,
            embed_dim=128,
            num_heads=8,
            window_size=(5, 1),
            sr_ratio=1,
            output_dim=d_model
        )
        self.conv_encoder = InsnetBackbone()
        
        self.cfi_module = ConfidenceGatedIntegration(d_model=d_model, dropout=0.1)
        self.t_trans = TimeDistributedTransformer(
            dim_in=d_model, 
            dim_n=d_model, 
            num_heads=8, 
            num_layers=4, 
            max_t=200, 
            dropout=0.1
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ELU(), nn.Dropout(0.2),
            nn.Linear(64, 32),      nn.ELU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        total_samples, H, W, C = x.size()
        batch_size = int(total_samples / self.timesteps)
        
        x_spatial = x.squeeze(3) if x.size(3) == 1 else x
        feat_spatial = self.spatial_encoder(x_spatial, size=(H, W))
        
        x_conv = x.permute(0, 3, 1, 2).contiguous()
        feat_conv = self.conv_encoder(x_conv)
        
        if feat_conv.dim() == 4:
            feat_conv = feat_conv.mean(dim=(2, 3))
            
        feat_spatial = feat_spatial.view(batch_size, self.timesteps, self.d_model)
        feat_conv = feat_conv.view(batch_size, self.timesteps, self.d_model)
        
        z_list = []
        for t in range(self.timesteps):
            z_t = self.cfi_module(feat_conv[:, t, :], feat_spatial[:, t, :])
            z_list.append(z_t.unsqueeze(1))
        Z_f = torch.cat(z_list, dim=1)
        
        X_ctx = self.t_trans(Z_f)
        logits = self.mlp(X_ctx)
        
        return logits.view(total_samples, 1)


def train_pipeline(train_file_list, valid_file_list, best_model_path, num_epochs, timesteps, batch_size):
    print(f"Initializing DataLoader for Timesteps: {timesteps}, Batch Size: {batch_size}")
    train_loader = get_data_loader(train_file_list, timesteps, batch_size=batch_size, shuffle=True)
    valid_loader = get_data_loader(valid_file_list, timesteps, batch_size=batch_size, shuffle=False)

    model = SSL_InsNet_dodel(timesteps=timesteps, d_model=320)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Parallel accelerating activated. GPU Count: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
    
    loss_fn = NeighborhoodWeightedFocalLoss(alpha=0.25, gamma=2.0, radius_K=2, reduction="mean")
    print("Neighborhood-Weighted Focal Loss (NWFL) successfully deployed.")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, delta=0.001, path=best_model_path)
    
    train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\n==================== Epoch {epoch + 1}/{num_epochs} =====================")
        
        train_loss, train_metrics, t_auc = train_fn(
            model, DEVICE, timesteps, train_loader, optimizer, loss_fn, train_losses, train_accuracy
        )
        print(f"[Train] Loss: {train_loss:.4f} | AUC: {t_auc:.4f} | F1: {train_metrics['f1_score']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"        TP: {train_metrics['TP']}, FP: {train_metrics['FP']}, TN: {train_metrics['TN']}, FN: {train_metrics['FN']}")
        
        valid_loss, valid_metrics = valid_fn(
            model, DEVICE, timesteps, valid_loader, loss_fn, val_losses, val_accuracy
        )
        print(f"[Valid] Loss: {valid_loss:.4f} | F1: {valid_metrics['f1_score']:.4f} | Acc: {valid_metrics['accuracy']:.4f}")
        print(f"        TP: {valid_metrics['TP']}, FP: {valid_metrics['FP']}, TN: {valid_metrics['TN']}, FN: {valid_metrics['FN']}")
        
        early_stopping.save_checkpoint(valid_metrics['f1_score'], model)
        early_stopping(valid_metrics['f1_score'], model)
        scheduler.step(valid_metrics['f1_score'])
        
        if early_stopping.early_stop:
            print(">> Early stopping triggered. Retaining best model weights. Training workflow finalized.")
            break
