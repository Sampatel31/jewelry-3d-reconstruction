"""
Mesh postprocessing for cleanup and optimization.
"""

import numpy as np
import trimesh
from typing import Optional, Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MeshPostprocessor:
    """
    Postprocessing operations for mesh cleanup and optimization.
    
    Features:
    - Remove degenerate faces
    - Fill holes
    - Smooth surfaces
    - Simplify geometry
    """
    
    def __init__(self, config: Dict):
        """
        Initialize postprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Mesh postprocessor initialized")
    
    def cleanup_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Clean up mesh by removing degenerate elements.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Cleaned mesh
        """
        logger.info("Cleaning mesh...")
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        logger.info(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def fill_holes(self, mesh: trimesh.Trimesh, max_hole_size: Optional[int] = None) -> trimesh.Trimesh:
        """
        Fill small holes in the mesh.
        
        Args:
            mesh: Input mesh
            max_hole_size: Maximum hole size to fill (number of edges)
            
        Returns:
            Mesh with filled holes
        """
        logger.info("Filling holes in mesh...")
        
        # Fill holes using trimesh
        mesh.fill_holes()
        
        logger.info("Holes filled")
        return mesh
    
    def smooth_mesh(self, mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        """
        Apply Laplacian smoothing to mesh.
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mesh
        """
        logger.info(f"Smoothing mesh with {iterations} iterations...")
        
        # Apply Taubin smoothing (better preserves volume)
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
        
        logger.info("Mesh smoothed")
        return mesh
    
    def simplify_mesh(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """
        Simplify mesh to target face count.
        
        Args:
            mesh: Input mesh
            target_faces: Target number of faces
            
        Returns:
            Simplified mesh
        """
        logger.info(f"Simplifying mesh to {target_faces} faces...")
        
        if len(mesh.faces) <= target_faces:
            logger.info("Mesh already has fewer faces than target")
            return mesh
        
        # Use quadric decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        logger.info(f"Simplified mesh: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
        return simplified
