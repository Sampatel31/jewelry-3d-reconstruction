"""
Metrics for evaluating reconstruction quality.
Includes LPIPS, Chamfer Distance, and other 3D metrics.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from scipy.spatial import cKDTree

from .logger import get_logger

logger = get_logger(__name__)


def chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    bidirectional: bool = True
) -> float:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        points1: Nx3 array of points
        points2: Mx3 array of points
        bidirectional: If True, compute symmetric distance
        
    Returns:
        Chamfer distance (lower is better)
    """
    # Build KD-trees for efficient nearest neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Compute one-way distances
    distances1, _ = tree2.query(points1, k=1)
    chamfer1 = np.mean(distances1 ** 2)
    
    if bidirectional:
        distances2, _ = tree1.query(points2, k=1)
        chamfer2 = np.mean(distances2 ** 2)
        return (chamfer1 + chamfer2) / 2.0
    else:
        return chamfer1


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (HxWxC, values 0-max_value)
        img2: Second image (same shape as img1)
        max_value: Maximum possible pixel value
        
    Returns:
        PSNR in dB (higher is better)
    """
    mse = np.mean((img1 - img2) ** 2)
    
    if mse < 1e-10:
        return 100.0  # Perfect match
    
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        img1: First image (HxWxC, values 0-1)
        img2: Second image (same shape as img1)
        
    Returns:
        SSIM score (0-1, higher is better)
    """
    try:
        from skimage.metrics import structural_similarity
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3 and img1.shape[2] > 1:
            # Compute SSIM for each channel and average
            ssim_scores = []
            for c in range(img1.shape[2]):
                ssim = structural_similarity(img1[:, :, c], img2[:, :, c], data_range=1.0)
                ssim_scores.append(ssim)
            return np.mean(ssim_scores)
        else:
            return structural_similarity(img1, img2, data_range=1.0)
    except ImportError:
        logger.warning("scikit-image not available, SSIM not computed")
        return 0.0


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity metric.
    Uses pretrained network to compute perceptual distance.
    """
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Initialize LPIPS metric.
        
        Args:
            net: Network backbone ('alex', 'vgg', or 'squeeze')
            device: Computation device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net).to(self.device)
            self.loss_fn.eval()
            logger.info(f"LPIPS metric initialized with {net} backbone")
        except ImportError:
            logger.warning("lpips package not available, install with: pip install lpips")
            self.loss_fn = None
    
    def compute(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: First image (HxWxC, values 0-1)
            img2: Second image (same shape as img1)
            
        Returns:
            LPIPS distance (lower is better, typically 0-1)
        """
        if self.loss_fn is None:
            return 0.0
        
        # Convert to tensors (CxHxW format, range -1 to 1)
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
        
        # Normalize to [-1, 1]
        img1_t = img1_t * 2 - 1
        img2_t = img2_t * 2 - 1
        
        # Move to device
        img1_t = img1_t.to(self.device)
        img2_t = img2_t.to(self.device)
        
        # Compute LPIPS
        with torch.no_grad():
            distance = self.loss_fn(img1_t, img2_t)
        
        return distance.item()


def compute_mesh_metrics(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    pred_faces: Optional[np.ndarray] = None,
    gt_faces: Optional[np.ndarray] = None
) -> dict:
    """
    Compute various metrics between predicted and ground truth meshes.
    
    Args:
        pred_vertices: Predicted mesh vertices (Nx3)
        gt_vertices: Ground truth mesh vertices (Mx3)
        pred_faces: Optional predicted faces
        gt_faces: Optional ground truth faces
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Chamfer Distance
    metrics['chamfer_distance'] = chamfer_distance(pred_vertices, gt_vertices)
    
    # Vertex count comparison
    metrics['vertex_count_pred'] = len(pred_vertices)
    metrics['vertex_count_gt'] = len(gt_vertices)
    
    if pred_faces is not None and gt_faces is not None:
        metrics['face_count_pred'] = len(pred_faces)
        metrics['face_count_gt'] = len(gt_faces)
    
    return metrics
