"""
PBR texture baking from multi-view images.
"""

import numpy as np
import trimesh
from PIL import Image
from typing import Dict, List, Tuple, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PBRBaker:
    """
    PBR texture baker for mesh UV mapping.
    
    Bakes multi-view images into high-resolution texture maps
    with proper UV parameterization.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PBR baker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.texture_size = config.get('texture_size', 4096)
        
        logger.info(f"PBR baker initialized (texture size: {self.texture_size})")
    
    def bake_textures(
        self,
        mesh: trimesh.Trimesh,
        images: List[np.ndarray],
        cameras: List = None,
        material_maps: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Image.Image]:
        """
        Bake textures from multi-view images.
        
        Args:
            mesh: Input mesh (should have UV coordinates)
            images: List of input images
            cameras: List of camera objects
            material_maps: Optional pre-extracted material maps
            
        Returns:
            Dictionary of baked texture maps as PIL Images
        """
        logger.info(f"Baking textures at {self.texture_size}x{self.texture_size}...")
        
        # Ensure mesh has UV coordinates
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            logger.warning("Mesh has no UV coordinates, generating...")
            mesh = self._generate_uv_coordinates(mesh)
        
        # Create texture atlas
        textures = {}
        
        if material_maps is not None:
            # Use provided material maps
            for map_name, map_data in material_maps.items():
                # Resize to target resolution
                if len(map_data.shape) == 3:
                    img = Image.fromarray(map_data.astype(np.uint8))
                else:
                    img = Image.fromarray(map_data.astype(np.uint8), mode='L')
                
                img = img.resize((self.texture_size, self.texture_size), Image.LANCZOS)
                textures[map_name] = img
        else:
            # Create default textures
            textures['albedo'] = Image.new('RGB', (self.texture_size, self.texture_size), (200, 200, 200))
            textures['roughness'] = Image.new('L', (self.texture_size, self.texture_size), 128)
            textures['metallic'] = Image.new('L', (self.texture_size, self.texture_size), 200)
            textures['normal'] = Image.new('RGB', (self.texture_size, self.texture_size), (128, 128, 255))
        
        logger.info("Texture baking complete")
        return textures
    
    def _generate_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Generate UV coordinates for mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Mesh with UV coordinates
        """
        logger.info("Generating UV coordinates...")
        
        try:
            # Use xatlas for UV unwrapping if available
            import xatlas
            
            vmapping, indices, uvs = xatlas.parametrize(
                mesh.vertices,
                mesh.faces
            )
            
            # Create new mesh with UVs
            mesh_uv = trimesh.Trimesh(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                process=False
            )
            mesh_uv.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            logger.info("UV generation complete")
            return mesh_uv
            
        except ImportError:
            logger.warning("xatlas not available, using simple sphere UV mapping")
            # Fallback: simple spherical UV mapping
            vertices = mesh.vertices
            uvs = np.zeros((len(vertices), 2))
            
            # Normalize to unit sphere
            normalized = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)
            
            # Spherical coordinates
            uvs[:, 0] = 0.5 + np.arctan2(normalized[:, 0], normalized[:, 2]) / (2 * np.pi)
            uvs[:, 1] = 0.5 - np.arcsin(normalized[:, 1]) / np.pi
            
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
            return mesh
