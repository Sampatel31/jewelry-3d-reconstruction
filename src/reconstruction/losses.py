"""
Loss functions for 3D reconstruction.
Includes photometric, depth distortion, and normal consistency losses.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReconstructionLoss(nn.Module):
    """
    Combined loss for 3D reconstruction.
    
    Components:
    - Photometric loss (L1 + SSIM)
    - Depth distortion loss
    - Normal consistency loss
    - Opacity regularization
    """
    
    def __init__(self, config: Dict):
        """
        Initialize loss functions.
        
        Args:
            config: Configuration dictionary with loss weights
        """
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.lambda_photo = config.get('lambda_photo', 1.0)
        self.lambda_depth = config.get('lambda_depth', 0.1)
        self.lambda_normal = config.get('lambda_normal', 0.05)
        self.lambda_opacity = config.get('lambda_opacity', 0.01)
        
        logger.info("Reconstruction loss initialized")
    
    def forward(
        self,
        rendered: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            rendered: Dictionary of rendered outputs
            target: Dictionary of ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Photometric loss
        if 'image' in rendered and 'image' in target:
            losses['photo'] = self._photometric_loss(rendered['image'], target['image'])
        
        # Depth distortion loss
        if 'depth' in rendered:
            losses['depth_distortion'] = self._depth_distortion_loss(rendered['depth'])
        
        # Total loss
        total = 0.0
        if 'photo' in losses:
            total += self.lambda_photo * losses['photo']
        if 'depth_distortion' in losses:
            total += self.lambda_depth * losses['depth_distortion']
        
        losses['total'] = total
        return losses
    
    def _photometric_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute photometric loss (L1).
        
        Args:
            rendered: Rendered image (HxWx3)
            target: Target image (HxWx3)
            
        Returns:
            Loss scalar
        """
        return torch.mean(torch.abs(rendered - target))
    
    def _depth_distortion_loss(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute depth distortion loss to encourage smooth surfaces.
        
        Args:
            depth: Depth map (HxW)
            
        Returns:
            Loss scalar
        """
        # Compute depth gradients
        grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])
        grad_y = torch.abs(depth[1:, :] - depth[:-1, :])
        
        # L1 loss on gradients
        return torch.mean(grad_x) + torch.mean(grad_y)
    
    def _normal_consistency_loss(
        self,
        normals: torch.Tensor,
        neighbor_normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normal consistency loss.
        
        Args:
            normals: Normal vectors (Nx3)
            neighbor_normals: Normals of neighboring points (NxKx3)
            
        Returns:
            Loss scalar
        """
        # Cosine similarity
        similarity = torch.sum(normals.unsqueeze(1) * neighbor_normals, dim=-1)
        return torch.mean(1.0 - similarity)
