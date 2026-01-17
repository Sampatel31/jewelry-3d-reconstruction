"""
Tests for single-view TRELLIS pipeline (Pipeline A).
"""

import pytest
import numpy as np
from PIL import Image

from config.pipeline_config import PipelineConfig
from src.reconstruction.trellis_pipeline import TRELLISPipeline


class TestTRELLISPipeline:
    """Test cases for TRELLIS single-view reconstruction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PipelineConfig()
        self.pipeline = TRELLISPipeline(self.config.to_dict())
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        assert self.pipeline is not None
        assert self.pipeline.config is not None
    
    def test_reconstruct_from_single_image(self):
        """Test reconstruction from a single image."""
        # Create dummy image
        image = Image.new('RGB', (512, 512), color='white')
        
        # Run reconstruction
        result = self.pipeline.reconstruct(image)
        
        # Verify result
        assert result is not None
        assert 'mesh' in result
        assert result['mesh'] is not None
        
        # Check mesh properties
        mesh = result['mesh']
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    
    def test_reconstruct_with_alpha_channel(self):
        """Test reconstruction with RGBA image."""
        # Create dummy RGBA image
        image = Image.new('RGBA', (512, 512), color=(255, 255, 255, 255))
        
        # Run reconstruction
        result = self.pipeline.reconstruct(image)
        
        # Verify result
        assert result is not None
        assert 'mesh' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
