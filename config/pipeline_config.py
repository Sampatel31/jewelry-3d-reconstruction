"""
Pipeline configuration for jewelry 3D reconstruction.
Contains hyperparameters for all processing stages.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class PipelineConfig:
    """
    Configuration for the reconstruction pipeline.
    
    Attributes:
        # Input processing
        background_removal: Enable automatic background removal
        single_view_threshold: Maximum images for generative mode
        multi_view_range: (min, max) images for geometric mode
        image_size: Target image dimensions (height, width)
        
        # Pose estimation (Pipeline B)
        dust3r_model: DUSt3R model variant
        min_confidence: Minimum confidence threshold for points
        focal_length_guess: Initial focal length estimate (pixels)
        
        # Reconstruction
        num_iterations: Training iterations for reconstruction
        learning_rate: Initial learning rate
        batch_size: Batch size for training
        
        # Gaussian parameters
        gaussian_scale: Initial scale for Gaussians
        opacity_threshold: Opacity pruning threshold
        densification_interval: Steps between densification
        
        # Mesh extraction
        mesh_resolution: Voxel resolution for mesh extraction
        sugar_iterations: SuGaR refinement iterations
        poisson_depth: Poisson reconstruction depth
        
        # Material extraction
        pbr_resolution: PBR texture resolution
        enable_gemstone_detection: Enable automatic gem segmentation
        metallic_threshold: Threshold for metal detection
        
        # Output
        output_format: Output file format ('glb', 'gltf', 'obj')
        texture_size: Output texture resolution
        
        # Device
        device: Computation device ('cuda' or 'cpu')
        num_workers: Number of data loading workers
        
        # Debug
        verbose: Enable verbose logging
        save_intermediate: Save intermediate results
    """
    
    # Input processing
    background_removal: bool = True
    single_view_threshold: int = 2
    multi_view_range: Tuple[int, int] = (3, 5)
    image_size: Tuple[int, int] = (1024, 1024)
    
    # Pose estimation
    dust3r_model: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    min_confidence: float = 3.0
    focal_length_guess: Optional[float] = None
    
    # Reconstruction
    num_iterations: int = 30000
    learning_rate: float = 0.0025
    batch_size: int = 1
    
    # Gaussian parameters
    gaussian_scale: float = 0.01
    opacity_threshold: float = 0.005
    densification_interval: int = 100
    densification_start: int = 500
    densification_end: int = 15000
    
    # Mesh extraction
    mesh_resolution: int = 512
    sugar_iterations: int = 5000
    poisson_depth: int = 10
    
    # Material extraction
    pbr_resolution: int = 4096
    enable_gemstone_detection: bool = True
    metallic_threshold: float = 0.7
    
    # Output
    output_format: str = "glb"
    texture_size: int = 4096
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Debug
    verbose: bool = False
    save_intermediate: bool = False
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'background_removal': self.background_removal,
            'single_view_threshold': self.single_view_threshold,
            'multi_view_range': self.multi_view_range,
            'image_size': self.image_size,
            'dust3r_model': self.dust3r_model,
            'min_confidence': self.min_confidence,
            'focal_length_guess': self.focal_length_guess,
            'num_iterations': self.num_iterations,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gaussian_scale': self.gaussian_scale,
            'opacity_threshold': self.opacity_threshold,
            'densification_interval': self.densification_interval,
            'densification_start': self.densification_start,
            'densification_end': self.densification_end,
            'mesh_resolution': self.mesh_resolution,
            'sugar_iterations': self.sugar_iterations,
            'poisson_depth': self.poisson_depth,
            'pbr_resolution': self.pbr_resolution,
            'enable_gemstone_detection': self.enable_gemstone_detection,
            'metallic_threshold': self.metallic_threshold,
            'output_format': self.output_format,
            'texture_size': self.texture_size,
            'device': self.device,
            'num_workers': self.num_workers,
            'verbose': self.verbose,
            'save_intermediate': self.save_intermediate,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
