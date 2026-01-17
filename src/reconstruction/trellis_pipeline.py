"""
Single-view 3D reconstruction pipeline using TripoSR.
TripoSR is a fast, feed-forward 3D reconstruction model from Stability AI.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
import trimesh

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TRELLISPipeline:
    """
    Single-view generative reconstruction using TripoSR.
    
    TripoSR generates 3D geometry from a single image using:
    1. Feed-forward transformer architecture
    2. Triplane representation
    3. Fast inference (no diffusion steps needed)
    
    Suitable for planar jewelry (necklaces, earrings) where
    backside geometry must be hallucinated.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize TripoSR pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        logger.info("Initializing TripoSR pipeline...")
        
        # Try to load TripoSR model via HuggingFace Hub
        try:
            import sys
            import subprocess
            
            # Check if TripoSR is available, if not, try to install it
            try:
                from tsr.system import TSR
            except ImportError:
                logger.info("TripoSR not found, attempting to install from GitHub...")
                try:
                    # Try installing directly from GitHub with proper subdirectory
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/VAST-AI-Research/TripoSR.git"
                    ], stderr=subprocess.DEVNULL)
                    from tsr.system import TSR
                    logger.info("TripoSR installed successfully")
                except Exception as install_error:
                    logger.warning(f"Could not install TripoSR: {install_error}")
                    raise ImportError("TripoSR not available")
            
            # Load pretrained model from HuggingFace
            model_name = config.get('triposr_model', 'stabilityai/TripoSR')
            logger.info(f"Loading TripoSR model from {model_name}...")
            
            self.model = TSR.from_pretrained(
                model_name,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TripoSR model loaded successfully")
            self.model_available = True
            
        except Exception as e:
            logger.warning(f"TripoSR not available: {e}. Using fallback placeholder.")
            logger.info("This is expected - TripoSR requires manual installation.")
            logger.info("The system will continue with placeholder meshes for demonstration.")
            self.model_available = False
            self.model = None
    
    def reconstruct(self, image: Image.Image, **kwargs) -> Dict:
        """
        Reconstruct 3D model from single image using TripoSR.
        
        Args:
            image: Input RGB or RGBA image
            **kwargs: Additional parameters
                - mc_resolution: Marching cubes resolution (default 256)
                - remove_background: Whether to remove background (default True)
        
        Returns:
            Dictionary containing:
                - mesh: Trimesh object
                - render: Optional render preview
        """
        logger.info("Starting TripoSR reconstruction...")
        
        if not self.model_available:
            logger.warning("TripoSR model not available, returning placeholder mesh")
            return self._create_placeholder_result()
        
        # Preprocess image
        if image.mode == 'RGBA':
            # Keep alpha channel for background removal
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image.convert('RGB')
        
        # Set parameters
        mc_resolution = kwargs.get('mc_resolution', 256)
        remove_background = kwargs.get('remove_background', False)  # Already done in preprocessing
        
        # Prepare image for TripoSR
        # TripoSR expects images preprocessed with rembg
        from rembg import remove as rembg_remove
        
        if remove_background and image.mode != 'RGBA':
            # Remove background if not already done
            image_no_bg = rembg_remove(image_rgb)
        else:
            image_no_bg = image
        
        # Run TripoSR inference
        logger.info("Running TripoSR inference...")
        with torch.no_grad():
            # TripoSR expects [1, 3, H, W] tensor
            scene_codes = self.model([image_no_bg], device=self.device)
        
        # Extract mesh using marching cubes
        logger.info(f"Extracting mesh with resolution {mc_resolution}...")
        meshes = self.model.extract_mesh(scene_codes, resolution=mc_resolution)
        
        # Get the first mesh
        mesh_data = meshes[0]
        
        # Convert to trimesh
        vertices = mesh_data.vertices.cpu().numpy()
        faces = mesh_data.faces.cpu().numpy()
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # Clean up mesh
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        logger.info(f"TripoSR reconstruction complete: {len(vertices)} vertices, {len(faces)} faces")
        
        return {
            'mesh': mesh,
            'scene_codes': scene_codes,
            'render': None
        }
    
    def _create_placeholder_result(self) -> Dict:
        """
        Create placeholder result when model is not available.
        
        Returns:
            Dictionary with placeholder mesh
        """
        # Create a simple ring mesh as placeholder
        mesh = trimesh.creation.torus(major_radius=1.0, minor_radius=0.1)
        
        return {
            'mesh': mesh,
            'slat': None,
            'gaussians': None
        }
