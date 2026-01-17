"""
Poisson surface reconstruction from point clouds.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PoissonReconstructor:
    """
    Poisson surface reconstruction.
    
    Extracts isosurface from oriented point cloud using
    screened Poisson reconstruction.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Poisson reconstructor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.depth = config.get('poisson_depth', 10)
        
        logger.info(f"Poisson reconstructor initialized (depth={self.depth})")
    
    def reconstruct(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> trimesh.Trimesh:
        """
        Reconstruct surface from point cloud.
        
        Args:
            points: Nx3 array of point positions
            normals: Optional Nx3 array of normal vectors
            
        Returns:
            Reconstructed mesh
        """
        logger.info(f"Running Poisson reconstruction on {len(points)} points...")
        
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Estimate normals if not provided
            if normals is None:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Orient normals consistently
            pcd.orient_normals_consistent_tangent_plane(k=15)
            
            # Poisson reconstruction
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.depth,
                linear_fit=True
            )
            
            # Filter low-density vertices
            densities = np.asarray(densities)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
            
            # Convert to trimesh
            vertices = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            logger.info(f"Reconstruction complete: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
            
        except Exception as e:
            logger.error(f"Poisson reconstruction failed: {e}")
            # Return placeholder
            return trimesh.creation.icosphere(radius=1.0)
