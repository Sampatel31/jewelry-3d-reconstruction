"""
Model configuration for pretrained weights and paths.
Handles automatic model downloading and caching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """
    Configuration for model weights and paths.
    
    Attributes:
        cache_dir: Directory to cache downloaded models
        dust3r_model: DUSt3R model identifier or path
        trellis_model: TRELLIS model identifier or path
        rembg_model: Background removal model name
        sam_model: Segment Anything model checkpoint
        mate_model: MatE material extraction model
    """
    
    # Cache directory
    cache_dir: Path = Path.home() / ".cache" / "jewelry-3d-reconstruction"
    
    # Model identifiers
    dust3r_model: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    trellis_model: str = "JeffreyXiang/TRELLIS-image-large"
    rembg_model: str = "u2net"
    sam_model: str = "facebook/sam-vit-huge"
    mate_model: Optional[str] = None  # To be added if available
    
    def __post_init__(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model."""
        return self.cache_dir / model_name
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'cache_dir': str(self.cache_dir),
            'dust3r_model': self.dust3r_model,
            'trellis_model': self.trellis_model,
            'rembg_model': self.rembg_model,
            'sam_model': self.sam_model,
            'mate_model': self.mate_model,
        }


# Default model configuration
DEFAULT_MODEL_CONFIG = ModelConfig()
