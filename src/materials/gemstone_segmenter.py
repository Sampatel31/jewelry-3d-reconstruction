"""
Gemstone segmentation using SAM (Segment Anything Model).
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GemstoneSegmenter:
    """
    Automatic gemstone detection and segmentation.
    
    Uses SAM to identify and segment gemstones in jewelry,
    enabling specialized material assignment (refraction, dispersion).
    """
    
    def __init__(self, config: Dict):
        """
        Initialize gemstone segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        logger.info("Initializing gemstone segmenter...")
        
        # Try to load SAM model
        try:
            from transformers import SamModel, SamProcessor
            
            model_name = config.get('sam_model', 'facebook/sam-vit-huge')
            self.processor = SamProcessor.from_pretrained(model_name)
            self.model = SamModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            logger.info("SAM model loaded successfully")
            self.model_available = True
            
        except ImportError:
            logger.warning("SAM not available, gemstone segmentation disabled")
            self.model_available = False
    
    def segment_gemstones(
        self,
        image: Image.Image,
        prompt_points: List[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment gemstones from image.
        
        Args:
            image: Input PIL Image
            prompt_points: Optional list of (x, y) points indicating gemstones
            
        Returns:
            Dictionary containing:
                - masks: List of binary masks for each gemstone
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: Confidence scores for each detection
        """
        logger.info("Segmenting gemstones...")
        
        if not self.model_available:
            logger.warning("SAM not available, returning empty segmentation")
            return {
                'masks': [],
                'boxes': [],
                'scores': []
            }
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare inputs
        if prompt_points is None:
            # Automatic gemstone detection based on reflectivity
            prompt_points = self._detect_reflective_regions(image)
        
        if len(prompt_points) == 0:
            logger.info("No gemstones detected")
            return {
                'masks': [],
                'boxes': [],
                'scores': []
            }
        
        # Run SAM segmentation
        with torch.no_grad():
            inputs = self.processor(
                image,
                input_points=[prompt_points],
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            masks = self.processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )[0]
        
        # Convert to numpy
        masks_np = [m.cpu().numpy() for m in masks]
        
        # Compute bounding boxes
        boxes = [self._mask_to_bbox(m) for m in masks_np]
        
        # Get confidence scores
        scores = outputs.iou_scores.cpu().numpy().flatten().tolist()
        
        logger.info(f"Segmented {len(masks_np)} gemstones")
        
        return {
            'masks': masks_np,
            'boxes': boxes,
            'scores': scores
        }
    
    def _detect_reflective_regions(self, image: Image.Image) -> List[Tuple[int, int]]:
        """
        Detect highly reflective regions (likely gemstones).
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of (x, y) point coordinates
        """
        import cv2
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # High value (brightness) indicates reflections
        _, bright_mask = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get centroids
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))
        
        return points
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """
        Convert binary mask to bounding box.
        
        Args:
            mask: Binary mask (HxW)
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return [int(x1), int(y1), int(x2), int(y2)]
