"""
2D Gaussian model for surface-aligned reconstruction.
Implements 2D Gaussian surfels with tangent vectors.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GaussianModel(torch.nn.Module):
    """
    2D Gaussian surfel representation for surface reconstruction.
    
    Each Gaussian is defined by:
    - Position (3D center point)
    - 2D covariance in tangent space
    - Opacity
    - Color (SH coefficients)
    - Normal direction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Gaussian model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Gaussian parameters (initialized later)
        self._xyz = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self._features_dc = None
        
        logger.info("Gaussian model initialized")
    
    def initialize_from_pointcloud(self, points: np.ndarray, colors: np.ndarray):
        """
        Initialize Gaussians from point cloud.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-255)
        """
        num_points = len(points)
        logger.info(f"Initializing {num_points} Gaussians from point cloud")
        
        # Convert to tensors
        self._xyz = torch.nn.Parameter(
            torch.from_numpy(points).float().to(self.device),
            requires_grad=True
        )
        
        # Initialize opacities
        self._opacity = torch.nn.Parameter(
            torch.ones((num_points, 1), device=self.device) * 0.1,
            requires_grad=True
        )
        
        # Initialize scales (isotropic)
        scale = self.config.get('gaussian_scale', 0.01)
        self._scaling = torch.nn.Parameter(
            torch.ones((num_points, 2), device=self.device) * scale,
            requires_grad=True
        )
        
        # Initialize rotations (identity)
        self._rotation = torch.nn.Parameter(
            torch.zeros((num_points, 4), device=self.device),
            requires_grad=True
        )
        self._rotation.data[:, 0] = 1.0  # w component
        
        # Initialize colors (DC component only for now)
        colors_normalized = colors.astype(np.float32) / 255.0
        colors_sh = (colors_normalized - 0.5) / 0.28209479177387814  # SH normalization
        self._features_dc = torch.nn.Parameter(
            torch.from_numpy(colors_sh).float().to(self.device),
            requires_grad=True
        )
        
        logger.info("Gaussian initialization complete")
    
    def get_xyz(self) -> torch.Tensor:
        """Get Gaussian positions."""
        return self._xyz
    
    def get_opacity(self) -> torch.Tensor:
        """Get Gaussian opacities."""
        return torch.sigmoid(self._opacity)
    
    def get_scaling(self) -> torch.Tensor:
        """Get Gaussian scales."""
        return torch.exp(self._scaling)
    
    def get_rotation(self) -> torch.Tensor:
        """Get Gaussian rotations (quaternions)."""
        return torch.nn.functional.normalize(self._rotation)
    
    def get_features(self) -> torch.Tensor:
        """Get Gaussian color features."""
        return self._features_dc
    
    def densify(self):
        """
        Densify Gaussians by splitting/cloning.
        
        This method would implement:
        - Clone Gaussians in under-reconstructed regions
        - Split large Gaussians
        - Prune low-opacity Gaussians
        """
        logger.debug("Densifying Gaussians (placeholder)")
        # Placeholder - full implementation would analyze gradients
        pass
    
    def parameters(self):
        """Return list of trainable parameters."""
        params = []
        if self._xyz is not None:
            params.extend([
                self._xyz,
                self._opacity,
                self._scaling,
                self._rotation,
                self._features_dc
            ])
        return params
