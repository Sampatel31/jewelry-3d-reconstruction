"""
Tests for material extraction and composition.
"""

import pytest
import numpy as np
from PIL import Image
import trimesh

from config.pipeline_config import PipelineConfig
from src.materials.pbr_baker import PBRBaker
from src.materials.gemstone_segmenter import GemstoneSegmenter
from src.materials.material_composer import MaterialComposer


class TestMaterialExtraction:
    """Test cases for material extraction and processing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PipelineConfig()
    
    def test_pbr_baker_initialization(self):
        """Test PBR baker initializes correctly."""
        baker = PBRBaker(self.config.to_dict())
        assert baker is not None
        assert baker.texture_size == self.config.texture_size
    
    def test_texture_baking(self):
        """Test texture baking from images."""
        baker = PBRBaker(self.config.to_dict())
        
        # Create dummy mesh with UVs
        mesh = trimesh.creation.icosphere(radius=1.0)
        
        # Create dummy images
        images = [
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Bake textures
        textures = baker.bake_textures(mesh, images)
        
        # Verify results
        assert textures is not None
        assert 'albedo' in textures
        assert textures['albedo'] is not None
    
    def test_gemstone_segmenter_initialization(self):
        """Test gemstone segmenter initializes correctly."""
        segmenter = GemstoneSegmenter(self.config.to_dict())
        assert segmenter is not None
    
    def test_material_composer_initialization(self):
        """Test material composer initializes correctly."""
        composer = MaterialComposer(self.config.to_dict())
        assert composer is not None
    
    def test_material_composition(self):
        """Test composing materials for mesh."""
        composer = MaterialComposer(self.config.to_dict())
        
        # Create dummy mesh
        mesh = trimesh.creation.icosphere(radius=1.0)
        
        # Create dummy texture maps
        texture_maps = {
            'albedo': Image.new('RGB', (1024, 1024), color='white'),
            'roughness': Image.new('L', (1024, 1024), 128),
            'metallic': Image.new('L', (1024, 1024), 200),
            'normal': Image.new('RGB', (1024, 1024), (128, 128, 255))
        }
        
        # Compose materials
        result_mesh = composer.compose_materials(mesh, texture_maps)
        
        # Verify result
        assert result_mesh is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
