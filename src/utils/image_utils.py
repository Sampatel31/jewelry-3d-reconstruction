"""
Image processing utilities.
Handles resizing, normalization, and color space conversions.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import cv2

from .logger import get_logger

logger = get_logger(__name__)


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        maintain_aspect: If True, maintain aspect ratio and pad
        resample: Resampling method
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        # Resize maintaining aspect ratio
        image.thumbnail(target_size, resample)
        
        # Pad to target size if needed
        if image.size != target_size:
            padded = Image.new(image.mode, target_size, (0, 0, 0, 0) if image.mode == 'RGBA' else (0, 0, 0))
            offset = ((target_size[0] - image.size[0]) // 2, 
                     (target_size[1] - image.size[1]) // 2)
            padded.paste(image, offset)
            return padded
        return image
    else:
        return image.resize(target_size, resample)


def normalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None
) -> np.ndarray:
    """
    Normalize image array.
    
    Args:
        image: Input image array (HxWxC, values 0-255)
        mean: Mean values for each channel
        std: Standard deviation for each channel
        
    Returns:
        Normalized image array (float32, 0-1 or standardized)
    """
    # Convert to float
    img = image.astype(np.float32) / 255.0
    
    # Apply normalization if provided
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = (img - mean) / std
    
    return img


def denormalize_image(
    image: np.ndarray,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None
) -> np.ndarray:
    """
    Denormalize image array back to 0-255 range.
    
    Args:
        image: Normalized image array
        mean: Mean values used in normalization
        std: Standard deviation used in normalization
        
    Returns:
        Image array with values 0-255 (uint8)
    """
    img = image.copy()
    
    # Reverse standardization if applied
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = (img * std) + mean
    
    # Scale to 0-255
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def create_alpha_mask(image: Image.Image) -> np.ndarray:
    """
    Extract alpha mask from RGBA image.
    
    Args:
        image: PIL Image (should have alpha channel)
        
    Returns:
        Binary mask as numpy array (bool)
    """
    if image.mode != 'RGBA':
        logger.warning(f"Image mode is {image.mode}, expected RGBA")
        # Convert to RGBA
        image = image.convert('RGBA')
    
    alpha = np.array(image.split()[-1])
    mask = alpha > 127
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray, background: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Apply binary mask to image with specified background.
    
    Args:
        image: Input image array (HxWx3)
        mask: Binary mask (HxW)
        background: Background color (R, G, B)
        
    Returns:
        Masked image array
    """
    result = image.copy()
    result[~mask] = background
    return result


def compute_image_gradient(image: np.ndarray) -> np.ndarray:
    """
    Compute image gradient magnitude using Sobel operator.
    
    Args:
        image: Input image (grayscale or RGB)
        
    Returns:
        Gradient magnitude array
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude
