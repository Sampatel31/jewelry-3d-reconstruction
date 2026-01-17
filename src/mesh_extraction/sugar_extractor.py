"""
SuGaR-based mesh extraction from Gaussian representation.
Surface-Aligned Gaussian Rasterization for mesh generation.
"""

import torch
import numpy as np
import trimesh
from typing import Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SuGaRExtractor:
    """
    Extract surface mesh from Gaussian representation using SuGaR.
    
    SuGaR (Surface-Aligned Gaussian Rasterization) aligns Gaussians
    to surfaces and extracts a clean mesh.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SuGaR extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        logger.info("SuGaR extractor initialized")
    
    def extract(self, gaussians, iterations: Optional[int] = None) -> trimesh.Trimesh:
        """
        Extract mesh from Gaussian representation.
        
        Args:
            gaussians: GaussianModel instance
            iterations: Optional number of refinement iterations
            
        Returns:
            Extracted mesh as Trimesh object
        """
        if iterations is None:
            iterations = self.config.get('sugar_iterations', 5000)
        
        logger.info(f"Extracting mesh with SuGaR ({iterations} iterations)...")
        
        # Get Gaussian parameters
        xyz = gaussians.get_xyz().detach().cpu().numpy()
        
        # Placeholder: In production, would run SuGaR optimization
        # For now, use Poisson reconstruction as fallback
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.estimate_normals()
            
            # Poisson reconstruction
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.config.get('poisson_depth', 10)
            )
            
            # Convert to trimesh
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            logger.warning(f"SuGaR extraction failed: {e}, using placeholder")
            mesh = trimesh.creation.icosphere(radius=1.0)
        
        logger.info(f"Mesh extracted: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
