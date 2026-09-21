import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodWeightedFocalLoss(nn.Module):
    """
    Neighborhood-Weighted Focal Loss (NWFL) Module.
    Strictly formulated based on Equations (19) to (21) of the manuscript.
    Incorporates biological spatial continuity via Markov Random Field (MRF) smoothing principles.
    """
    def __init__(self, alpha=0.25, gamma=2.0, radius_K=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.K = radius_K
        self.reduction = reduction

    def _compute_neighborhood_average(self, y):
        """
        Calculates bar{y}_{t,K} based on Equation (21).
        Uses a 1D sliding window of diameter 2K+1, excluding the center element.
        Args:
            y (torch.Tensor): Ground truth labels. Shape: [B, T]
        Returns:
            y_bar (torch.Tensor): Neighborhood averages. Shape: [B, T]
        """
        B, T = y.shape
        # Reshape to [B, 1, T] for 1D convolution/pooling processing
        y_padded = y.unsqueeze(1).float()
        
        # Padding symmetrically or with zeros at boundaries to keep temporal resolution
        # Width of padding on each side is K
        padding_size = self.K
        
        # Define a standard sum pooling kernel via constant 1D Conv
        window_size = 2 * self.K + 1
        kernel = torch.ones(1, 1, window_size, device=y.device)
        
        # Sum up all elements within the window radius K
        window_sum = F.conv1d(y_padded, kernel, padding=padding_size)
        
        # Equation (21): Exclude the center element itself and divide by 2K
        # y_padded contains the center element value
        neighborhood_sum = window_sum - y_padded
        y_bar = neighborhood_sum / (2.0 * self.K)
        
        # Remove the channel dimension -> [B, T]
        return y_bar.squeeze(1)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Predicted raw outputs from model tagging layers [B, T] or [B, T, 1]
            targets (torch.Tensor): Binary ground truth labels [B, T]
        Returns:
            loss (torch.Tensor): Evaluated total loss values L_total
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)
            
        B, T = logits.shape
        
        # Compute tracking probabilities (p_t) for binary cross entropy
        probs = torch.sigmoid(logits)
        # p_t in formula (19) represents probability of the true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        
        # ----------------------------------------------------
        # 1. Neighborhood Average & Weight Derivation
        # ----------------------------------------------------
        # Equation (21): Compute neighborhood averages over window radius K
        y_bar = self._compute_neighborhood_average(targets) # Shape: [B, T]
        
        # Equation (20): Compute temporal consistency weight omega_t
        # |y_t - \bar{y}_{t,K}| quantifies local label turbulence
        label_turbulence = torch.abs(targets.float() - y_bar)
        omega_t = 1.0 + torch.exp(-label_turbulence) # Shape: [B, T]
        
        # ----------------------------------------------------
        # 2. Total Loss Formulation
        # ----------------------------------------------------
        # Basic Focal Loss components
        focal_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        log_p_t = torch.log(torch.clamp(p_t, min=1e-8))
        
        # Equation (19): L_total = - sum ( omega_t * alpha * (1 - p_t)^gamma * log(p_t) )
        loss_matrix = - omega_t * focal_weight * modulating_factor * log_p_t
        
        if self.reduction == 'mean':
            return loss_matrix.mean()
        elif self.reduction == 'sum':
            return loss_matrix.sum()
        else:
            return loss_matrix
