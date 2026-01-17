"""
DUSt3R integration for pose-free camera estimation from sparse views.
Implements pointmap regression and global alignment optimization.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from ..utils.logger import get_logger
from .global_optimizer import GlobalPoseOptimizer
from .confidence_masking import ConfidenceMasker

logger = get_logger(__name__)


class Camera:
    """Simple camera class to store intrinsics and extrinsics."""
    
    def __init__(self, K: np.ndarray, R: np.ndarray, t: np.ndarray, image: np.ndarray):
        """
        Args:
            K: 3x3 intrinsic matrix
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            image: HxWx3 image array
        """
        self.K = K
        self.R = R
        self.t = t
        self.image = image
        self.height, self.width = image.shape[:2]
    
    @property
    def pose(self) -> np.ndarray:
        """Get 4x4 world-to-camera transform."""
        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[:3, 3] = self.t.flatten()
        return pose


class DUSt3RPoseEstimator:
    """
    Wrapper for DUSt3R model to estimate camera poses from unposed images.
    
    DUSt3R bypasses traditional feature matching by directly regressing
    dense 3D pointmaps from image pairs using a Vision Transformer.
    
    This is critical for jewelry where specular reflections prevent
    reliable feature tracking.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize DUSt3R pose estimator.
        
        Args:
            model_path: Path to pretrained weights (downloads if None)
            device: Computation device
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info("Loading DUSt3R model (MASt3R ViT-Large)...")
        
        # Import DUSt3R modules (gracefully handle if not installed)
        try:
            from dust3r.inference import inference
            from dust3r.model import AsymmetricMASt3R
            from dust3r.utils.device import to_numpy
            
            self.inference_fn = inference
            self.to_numpy = to_numpy
            
            # Load pretrained model
            if model_path is None:
                model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            else:
                model_name = model_path
                
            self.model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            logger.info("DUSt3R model loaded successfully")
            
        except ImportError as e:
            logger.error("Failed to import DUSt3R. Ensure it's installed: pip install git+https://github.com/naver/dust3r.git")
            raise e
        
        # Initialize optimizer
        self.optimizer = GlobalPoseOptimizer()
        self.confidence_masker = ConfidenceMasker()
    
    def estimate_poses(self, images: List[np.ndarray], focal_length_guess: Optional[float] = None) -> Dict:
        """
        Estimate camera poses and generate dense point cloud from unposed images.
        
        Process:
        1. Run pairwise pointmap regression on all image pairs
        2. Extract confidence maps (low conf on specular regions)
        3. Perform global optimization to align pointmaps
        4. Solve for absolute camera poses
        5. Merge pointmaps into unified dense cloud
        
        Args:
            images: List of RGB images as numpy arrays (HxWx3, values 0-255)
            focal_length_guess: Optional prior on focal length (pixels)
            
        Returns:
            Dictionary containing:
                - cameras: List[Camera] with calibrated intrinsics/extrinsics
                - point_cloud: Nx3 numpy array of 3D points
                - colors: Nx3 numpy array of RGB colors (0-255)
                - confidence_maps: List of confidence maps for each image
        """
        num_images = len(images)
        logger.info(f"Estimating poses for {num_images} images")
        
        # Prepare images for DUSt3R
        from dust3r.utils.image import load_images
        
        # Convert numpy arrays to expected format
        img_list = []
        for idx, img in enumerate(images):
            # DUSt3R expects dict with 'img' key
            img_dict = {
                'img': torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                'idx': idx,
                'instance': f'img_{idx}'
            }
            img_list.append(img_dict)
        
        # Run pairwise inference
        logger.info("Running pairwise pointmap regression...")
        with torch.no_grad():
            output = self.inference_fn(
                img_list,
                self.model,
                device=self.device,
                batch_size=1,
                verbose=False
            )
        
        # Extract scene representation
        scene = output['scene']
        
        # Global optimization with confidence weighting
        logger.info("Performing global pose optimization...")
        scene.global_optimization(
            min_conf_thr=3.0,  # Filter low-confidence points
            niter_PnP=100,
            niter_bundle_adjust=50
        )
        
        # Extract calibrated cameras
        cameras = []
        for idx in range(num_images):
            # Get camera parameters
            K = scene.get_intrinsics()[idx].cpu().numpy()
            pose = scene.get_extrinsics()[idx].cpu().numpy()  # 4x4 matrix
            
            R = pose[:3, :3]
            t = pose[:3, 3:]
            
            cam = Camera(K=K, R=R, t=t, image=images[idx])
            cameras.append(cam)
        
        # Extract unified point cloud
        logger.info("Extracting dense point cloud...")
        pts3d, colors, confs = scene.get_dense_pts3d(clean_depth=True)
        
        # Convert to numpy
        point_cloud = self.to_numpy(pts3d).reshape(-1, 3)
        point_colors = self.to_numpy(colors).reshape(-1, 3)
        confidence_maps = [self.to_numpy(c) for c in confs]
        
        # Filter low-confidence points
        valid_mask = confidence_maps[0].flatten() > 3.0
        point_cloud = point_cloud[valid_mask]
        point_colors = point_colors[valid_mask]
        
        logger.info(f"Generated {len(point_cloud)} high-confidence points")
        
        return {
            'cameras': cameras,
            'point_cloud': point_cloud,
            'colors': point_colors,
            'confidence_maps': confidence_maps
        }
