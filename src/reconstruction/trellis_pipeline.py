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
        logger.info("⚠️ IMPORTANT: TripoSR GitHub repo doesn't have proper setup.py")
        logger.info("📌 Solution: Using direct model implementation from HuggingFace")
        
        # Install TripoSR package directly using alternative method
        try:
            import sys
            import subprocess
            
            # First try importing
            try:
                from tsr.system import TSR
                logger.info("✅ TripoSR already installed")
            except ImportError:
                logger.info("📦 Installing TripoSR package dependencies...")
                
                # Install required packages for TripoSR
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q",
                    "omegaconf", "einops", "pytorch-lightning", "nvdiffrast"
                ], stderr=subprocess.STDOUT)
                
                # Clone and install TripoSR manually
                logger.info("📥 Downloading TripoSR source code...")
                import tempfile
                import os
                import shutil
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Clone the repository
                    subprocess.check_call([
                        "git", "clone", "--depth", "1",
                        "https://github.com/VAST-AI-Research/TripoSR.git",
                        tmpdir
                    ], stderr=subprocess.STDOUT)
                    
                    # Add to Python path
                    tsr_path = os.path.join(tmpdir, "tsr")
                    if tsr_path not in sys.path:
                        sys.path.insert(0, tmpdir)
                    
                    # Now import should work
                    from tsr.system import TSR
                    logger.info("✅ TripoSR loaded from source")
            
            # Load pretrained model from HuggingFace
            model_name = config.get('triposr_model', 'stabilityai/TripoSR')
            logger.info(f"📥 Loading TripoSR model from {model_name}...")
            logger.info("   This will download ~2GB of model weights (one-time only)")
            
            self.model = TSR.from_pretrained(
                model_name,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ TripoSR model loaded and ready for 3D reconstruction!")
            self.model_available = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load TripoSR: {e}")
            logger.error("📋 Error details:")
            import traceback
            logger.error(traceback.format_exc())
            logger.error("")
            logger.error("🔧 TROUBLESHOOTING:")
            logger.error("   1. Ensure you have git installed: !apt-get install -y git")
            logger.error("   2. Ensure GPU is available (TripoSR works best with GPU)")
            logger.error("   3. Try restarting the runtime and running again")
            logger.error("")
            raise RuntimeError(f"TripoSR initialization failed. Cannot continue without 3D reconstruction model. Error: {e}")
    
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
        
        # NO FALLBACKS - Fail if TripoSR is not available
        if not hasattr(self, 'model_available') or not self.model_available:
            raise RuntimeError(
                "❌ TripoSR model is not available!\n"
                "Cannot perform 3D reconstruction without the model.\n"
                "Please check the error messages during initialization."
            )
        
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
