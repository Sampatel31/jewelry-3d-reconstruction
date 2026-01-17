"""
Tests for multi-view Sparse2DGS pipeline (Pipeline B).
"""

import pytest
import numpy as np
from PIL import Image

from config.pipeline_config import PipelineConfig
from src.reconstruction.sparse2dgs_pipeline import Sparse2DGSPipeline
from src.pose_estimation.dust3r_wrapper import Camera


class TestSparse2DGSPipeline:
    """Test cases for Sparse2DGS multi-view reconstruction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PipelineConfig()
        self.pipeline = Sparse2DGSPipeline(self.config.to_dict())
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        assert self.pipeline is not None
        assert self.pipeline.config is not None
        assert self.pipeline.gaussian_model is not None
        assert self.pipeline.renderer is not None
    
    def test_reconstruct_from_multiple_views(self):
        """Test reconstruction from multiple views."""
        # Create dummy cameras
        cameras = []
        for i in range(3):
            K = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]])
            R = np.eye(3)
            t = np.array([[0], [0], [i * 2]])
            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            cam = Camera(K=K, R=R, t=t, image=image)
            cameras.append(cam)
        
        # Create dummy point cloud
        point_cloud = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)
        
        # Run reconstruction (with very few iterations for testing)
        result = self.pipeline.reconstruct(
            cameras=cameras,
            point_cloud=point_cloud,
            colors=colors,
            num_iterations=10  # Minimal iterations for testing
        )
        
        # Verify result
        assert result is not None
        assert 'mesh' in result
        assert result['mesh'] is not None
        
        # Check mesh properties
        mesh = result['mesh']
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
