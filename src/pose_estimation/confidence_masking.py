"""
Confidence masking for filtering unreliable points.
"""

import numpy as np
from typing import List, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceMasker:
    """
    Confidence-based filtering for 3D points and observations.
    
    Filters out low-confidence points that are likely to be:
    - Specular reflections
    - Occlusion boundaries
    - Texture-less regions
    """
    
    def __init__(self, min_confidence: float = 3.0):
        """
        Initialize confidence masker.
        
        Args:
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        logger.info(f"Confidence masker initialized (threshold={min_confidence})")
    
    def filter_points(
        self,
        points_3d: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter 3D points based on confidence scores.
        
        Args:
            points_3d: Nx3 array of 3D points
            confidences: N-length array of confidence scores
            
        Returns:
            Tuple of (filtered_points, valid_mask)
        """
        valid_mask = confidences > self.min_confidence
        filtered_points = points_3d[valid_mask]
        
        logger.info(f"Filtered points: {len(filtered_points)}/{len(points_3d)} kept")
        return filtered_points, valid_mask
    
    def filter_by_percentile(
        self,
        points_3d: np.ndarray,
        confidences: np.ndarray,
        percentile: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points by confidence percentile.
        
        Args:
            points_3d: Nx3 array of 3D points
            confidences: N-length array of confidence scores
            percentile: Percentile threshold (0-100)
            
        Returns:
            Tuple of (filtered_points, valid_mask)
        """
        threshold = np.percentile(confidences, percentile)
        valid_mask = confidences >= threshold
        filtered_points = points_3d[valid_mask]
        
        logger.info(f"Filtered by {percentile}th percentile: {len(filtered_points)}/{len(points_3d)} kept")
        return filtered_points, valid_mask
