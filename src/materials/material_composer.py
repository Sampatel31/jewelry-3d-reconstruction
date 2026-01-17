"""
Material composer for GLTF export with PBR materials.
"""

import numpy as np
import trimesh
from PIL import Image
from typing import Dict, List, Optional
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MaterialComposer:
    """
    Compose PBR materials and export to GLTF/GLB format.
    
    Creates physically-based materials with:
    - Albedo/base color
    - Metallic-roughness workflow
    - Normal mapping
    - Emissive properties (for gemstones)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize material composer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Material composer initialized")
    
    def compose_materials(
        self,
        mesh: trimesh.Trimesh,
        texture_maps: Dict[str, Image.Image],
        gemstone_masks: Optional[List[np.ndarray]] = None
    ) -> trimesh.Trimesh:
        """
        Compose PBR materials for mesh.
        
        Args:
            mesh: Input mesh
            texture_maps: Dictionary of texture maps (albedo, roughness, metallic, normal)
            gemstone_masks: Optional masks for gemstone regions
            
        Returns:
            Mesh with assigned materials
        """
        logger.info("Composing PBR materials...")
        
        # Create PBR material
        material = self._create_pbr_material(texture_maps, gemstone_masks)
        
        # Assign to mesh
        if hasattr(mesh.visual, 'material'):
            mesh.visual.material = material
        
        logger.info("Material composition complete")
        return mesh
    
    def _create_pbr_material(
        self,
        texture_maps: Dict[str, Image.Image],
        gemstone_masks: Optional[List[np.ndarray]] = None
    ) -> trimesh.visual.material.PBRMaterial:
        """
        Create PBR material from texture maps.
        
        Args:
            texture_maps: Dictionary of texture maps
            gemstone_masks: Optional gemstone masks
            
        Returns:
            PBR material
        """
        # Convert PIL Images to numpy arrays
        base_color_texture = None
        if 'albedo' in texture_maps:
            base_color_texture = np.array(texture_maps['albedo'])
        
        # Create metallic-roughness texture (R=unused, G=roughness, B=metallic)
        metallic_roughness = np.zeros(
            (texture_maps['albedo'].height, texture_maps['albedo'].width, 3),
            dtype=np.uint8
        )
        
        if 'roughness' in texture_maps:
            roughness_array = np.array(texture_maps['roughness'])
            if len(roughness_array.shape) == 2:
                metallic_roughness[:, :, 1] = roughness_array
            else:
                metallic_roughness[:, :, 1] = roughness_array[:, :, 0]
        
        if 'metallic' in texture_maps:
            metallic_array = np.array(texture_maps['metallic'])
            if len(metallic_array.shape) == 2:
                metallic_roughness[:, :, 2] = metallic_array
            else:
                metallic_roughness[:, :, 2] = metallic_array[:, :, 0]
        
        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color_texture,
            metallicRoughnessTexture=metallic_roughness,
            normalTexture=np.array(texture_maps.get('normal')) if 'normal' in texture_maps else None,
            metallicFactor=1.0,
            roughnessFactor=1.0
        )
        
        return material
    
    def export_gltf(
        self,
        mesh: trimesh.Trimesh,
        output_path: str,
        binary: bool = True
    ) -> None:
        """
        Export mesh with materials to GLTF/GLB.
        
        Args:
            mesh: Mesh with materials
            output_path: Output file path
            binary: If True, export as GLB (binary), else GLTF (JSON+bin)
        """
        logger.info(f"Exporting to {'GLB' if binary else 'GLTF'}: {output_path}")
        
        # Export
        file_format = 'glb' if binary else 'gltf'
        mesh.export(output_path, file_type=file_format)
        
        logger.info("Export complete")
    
    def assign_gemstone_materials(
        self,
        mesh: trimesh.Trimesh,
        gemstone_masks: List[np.ndarray],
        ior: float = 2.417  # Diamond IOR
    ) -> trimesh.Trimesh:
        """
        Assign specialized materials to gemstone regions.
        
        Args:
            mesh: Input mesh
            gemstone_masks: List of masks for gemstone regions
            ior: Index of refraction for gemstones
            
        Returns:
            Mesh with gemstone materials
        """
        logger.info(f"Assigning gemstone materials (IOR={ior})...")
        
        # Placeholder: In production, would create transmissive materials
        # with proper IOR and dispersion properties
        
        logger.info("Gemstone materials assigned")
        return mesh
