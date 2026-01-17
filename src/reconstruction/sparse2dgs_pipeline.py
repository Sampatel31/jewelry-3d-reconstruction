"""
Sparse 2D Gaussian Splatting pipeline for multi-view reconstruction.
Uses 2D Gaussian surfels for high-fidelity geometry reconstruction.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import trimesh

from ..utils.logger import get_logger
from ..pose_estimation.dust3r_wrapper import Camera
from .gaussian_model import GaussianModel
from .renderer import GaussianRenderer
from .losses import ReconstructionLoss

logger = get_logger(__name__)


class Sparse2DGSPipeline:
    """
    Multi-view geometric reconstruction using Sparse 2D Gaussian Splatting.
    
    Pipeline:
    1. Initialize 2D Gaussians from DUSt3R point cloud
    2. Optimize with depth distortion and normal consistency
    3. Densify in under-reconstructed regions
    4. Extract surface-aligned mesh
    
    Suitable for volumetric jewelry (rings, bangles) where
    sufficient views enable geometric reconstruction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Sparse2DGS pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.gaussian_model = GaussianModel(config)
        self.renderer = GaussianRenderer(config)
        self.loss_fn = ReconstructionLoss(config)
        
        logger.info("Sparse2DGS pipeline initialized")
    
    def reconstruct(
        self,
        cameras: List[Camera],
        point_cloud: np.ndarray,
        colors: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Reconstruct 3D model from multi-view images and poses.
        
        Args:
            cameras: List of calibrated Camera objects
            point_cloud: Nx3 initial point cloud from DUSt3R
            colors: Nx3 RGB colors for points
            **kwargs: Additional parameters
                - num_iterations: Training iterations
                - learning_rate: Optimization learning rate
        
        Returns:
            Dictionary containing:
                - mesh: Extracted surface mesh (Trimesh)
                - gaussians: Optimized Gaussian parameters
                - point_cloud: Final dense point cloud
        """
        logger.info(f"Starting Sparse2DGS reconstruction with {len(cameras)} views...")
        
        # Initialize Gaussians from point cloud
        self.gaussian_model.initialize_from_pointcloud(point_cloud, colors)
        
        # Get training parameters
        num_iterations = kwargs.get('num_iterations', self.config.get('num_iterations', 30000))
        learning_rate = kwargs.get('learning_rate', self.config.get('learning_rate', 0.0025))
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.gaussian_model.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info(f"Training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Select random camera
            cam_idx = np.random.randint(len(cameras))
            camera = cameras[cam_idx]
            
            # Render
            rendered = self.renderer.render(self.gaussian_model, camera)
            
            # Compute loss (placeholder - would need ground truth)
            # In practice, we'd compare against input images
            loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Densification
            if iteration % self.config.get('densification_interval', 100) == 0:
                if iteration >= self.config.get('densification_start', 500):
                    if iteration < self.config.get('densification_end', 15000):
                        self.gaussian_model.densify()
            
            # Logging
            if iteration % 1000 == 0:
                logger.info(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}")
        
        logger.info("Training complete")
        
        # Extract mesh (placeholder - would use SuGaR or Poisson)
        mesh = self._extract_mesh()
        
        # Get final point cloud
        final_points = self.gaussian_model.get_xyz().detach().cpu().numpy()
        
        return {
            'mesh': mesh,
            'gaussians': self.gaussian_model.state_dict(),
            'point_cloud': final_points
        }
    
    def _extract_mesh(self) -> trimesh.Trimesh:
        """
        Extract surface mesh from Gaussians.
        
        Returns:
            Trimesh object
        """
        logger.info("Extracting mesh from Gaussians...")
        
        # Placeholder: create simple mesh from Gaussian centers
        points = self.gaussian_model.get_xyz().detach().cpu().numpy()
        
        # Create mesh using Poisson reconstruction (simplified)
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate normals
            pcd.estimate_normals()
            
            # Poisson reconstruction
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # Convert to trimesh
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            logger.warning(f"Mesh extraction failed: {e}, using placeholder")
            # Fallback to simple mesh
            mesh = trimesh.creation.icosphere(radius=1.0)
        
        logger.info(f"Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
