"""
Image preprocessing module for background removal and normalization.
Integrates rembg and SAM for high-quality segmentation.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict
import cv2

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Handles all image preprocessing operations.
    
    Features:
    - Background removal using rembg (U-2-Net)
    - Alpha matting for clean edges
    - Image normalization and resizing
    - Mask generation from edge detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration with preprocessing parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Lazy load rembg to save memory
        self._rembg_session = None
        
        logger.info(f"Preprocessor initialized on {self.device}")
    
    def remove_background(self, image: Image.Image, alpha_matting: bool = True) -> Image.Image:
        """
        Remove background from image using deep learning segmentation.
        
        Uses rembg with U-2-Net model for precise foreground extraction.
        Applies alpha matting for smooth edges on reflective surfaces.
        
        Args:
            image: Input PIL Image (RGB)
            alpha_matting: Whether to apply alpha matting refinement
            
        Returns:
            PIL Image in RGBA format with transparent background
        """
        from rembg import remove, new_session
        
        # Initialize session lazily
        if self._rembg_session is None:
            logger.info("Loading background removal model (U-2-Net)...")
            self._rembg_session = new_session("u2net")
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Remove background
        output = remove(
            img_array,
            session=self._rembg_session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        # Convert back to PIL
        result = Image.fromarray(output).convert("RGBA")
        
        logger.debug("Background removed successfully")
        return result
    
    def create_mask_from_edges(self, image: Image.Image, threshold: int = 30) -> np.ndarray:
        """
        Create binary mask from edge detection (fallback if no alpha channel).
        
        Args:
            image: Input PIL Image
            threshold: Edge detection threshold
            
        Returns:
            Binary mask as numpy array (bool)
        """
        # Convert to grayscale
        gray = np.array(image.convert("L"))
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold, threshold * 2)
        
        # Dilate edges to create mask
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Fill interior
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask > 127
    
    def normalize_image(self, image: Image.Image, target_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """
        Normalize image size and format for neural network input.
        
        Args:
            image: Input PIL Image
            target_size: Target dimensions (height, width)
            
        Returns:
            Resized PIL Image
        """
        # Maintain aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Pad to target size
        padded = Image.new("RGBA", target_size, (0, 0, 0, 0))
        offset = ((target_size[0] - image.size[0]) // 2, 
                  (target_size[1] - image.size[1]) // 2)
        padded.paste(image, offset)
        
        return padded
