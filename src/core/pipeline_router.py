"""
Pipeline routing system that analyzes input images and selects optimal reconstruction strategy.
Implements automatic detection of jewelry type and input configuration.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from ..utils.logger import get_logger
from .preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class PipelineRouter:
    """
    Intelligent routing system for jewelry reconstruction pipelines.
    
    Analyzes input images and determines whether to use:
    - Pipeline A: Single-view generative (TRELLIS) for planar items
    - Pipeline B: Multi-view geometric (Sparse2DGS) for volumetric items
    
    Attributes:
        config: Configuration dictionary with routing parameters
        preprocessor: Image preprocessing module
    """
    
    def __init__(self, config: Dict):
        """
        Initialize router with configuration.
        
        Args:
            config: Dictionary containing:
                - single_view_threshold: Max images for generative mode
                - multi_view_range: (min, max) images for geometric mode
                - background_removal: Whether to auto-remove backgrounds
        """
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        logger.info("Pipeline router initialized")
    
    def analyze_input(self, image_paths: List[str]) -> Dict:
        """
        Analyze input images and determine reconstruction strategy.
        
        Performs:
        1. Image loading and validation
        2. Background removal (if enabled)
        3. Mask extraction
        4. Topology analysis
        5. Pipeline selection
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Dictionary containing:
                - mode: 'GENERATIVE' or 'GEOMETRIC'
                - cleaned_images: List of RGBA PIL Images
                - masks: List of binary masks (numpy arrays)
                - metadata: Analysis results (image count, estimated topology)
                
        Raises:
            ValueError: If image count is invalid or images cannot be processed
        """
        num_images = len(image_paths)
        logger.info(f"Analyzing {num_images} input images")
        
        # Validate input count
        if num_images == 0:
            raise ValueError("No input images provided")
        if num_images > 5:
            raise ValueError(f"Too many images ({num_images}). Maximum is 5.")
        
        # Load and preprocess images
        cleaned_images = []
        masks = []
        
        for idx, path in enumerate(image_paths):
            logger.debug(f"Processing image {idx + 1}/{num_images}: {path}")
            
            # Load image
            img = Image.open(path).convert("RGB")
            
            # Remove background (returns RGBA)
            if self.config.get('background_removal', True):
                cleaned = self.preprocessor.remove_background(img)
            else:
                cleaned = img
                
            # Extract mask
            if cleaned.mode == 'RGBA':
                mask = np.array(cleaned.split()[-1]) > 127
            else:
                # Create mask from edge detection if no alpha
                mask = self.preprocessor.create_mask_from_edges(cleaned)
            
            cleaned_images.append(cleaned)
            masks.append(mask)
        
        # Determine pipeline mode
        mode = self._determine_mode(num_images, masks)
        
        # Analyze topology
        metadata = self._analyze_topology(cleaned_images, masks)
        metadata['image_count'] = num_images
        
        result = {
            'mode': mode,
            'cleaned_images': cleaned_images,
            'masks': masks,
            'metadata': metadata
        }
        
        logger.info(f"Pipeline mode selected: {mode}")
        logger.info(f"Estimated topology: {metadata.get('topology_type', 'unknown')}")
        
        return result
    
    def _determine_mode(self, num_images: int, masks: List[np.ndarray]) -> str:
        """
        Determine which pipeline to use based on input characteristics.
        
        Logic:
        - 1 image -> GENERATIVE (must hallucinate backside)
        - 2 images -> GENERATIVE (insufficient for triangulation)
        - 3+ images -> GEOMETRIC (can triangulate geometry)
        
        Args:
            num_images: Number of input images
            masks: List of binary masks
            
        Returns:
            'GENERATIVE' or 'GEOMETRIC'
        """
        if num_images <= 2:
            return 'GENERATIVE'
        else:
            return 'GEOMETRIC'
    
    def _analyze_topology(self, images: List[Image.Image], masks: List[np.ndarray]) -> Dict:
        """
        Analyze object topology from masks.
        
        Estimates:
        - Topology type (planar vs volumetric)
        - Genus (number of holes/loops)
        - Bounding box dimensions
        - Aspect ratio
        
        Args:
            images: List of preprocessed images
            masks: List of binary masks
            
        Returns:
            Dictionary with topology analysis results
        """
        from scipy import ndimage
        from skimage import measure
        
        # Analyze first mask for topology
        mask = masks[0]
        
        # Compute connected components to estimate genus
        labeled, num_components = ndimage.label(~mask)
        num_holes = num_components - 1  # Background is 1 component
        
        # Compute bounding box
        coords = np.column_stack(np.where(mask))
        bbox_height = coords[:, 0].max() - coords[:, 0].min()
        bbox_width = coords[:, 1].max() - coords[:, 1].min()
        aspect_ratio = bbox_width / max(bbox_height, 1)
        
        # Estimate topology type
        if num_holes > 3 or aspect_ratio > 2.0:
            topology_type = 'planar'  # Likely necklace/chain
        else:
            topology_type = 'volumetric'  # Likely ring/bangle
        
        return {
            'topology_type': topology_type,
            'estimated_genus': num_holes,
            'aspect_ratio': aspect_ratio,
            'bbox_dimensions': (bbox_height, bbox_width)
        }
    
    def route(self, analysis_result: Dict):
        """
        Return appropriate pipeline instance based on analysis.
        
        Args:
            analysis_result: Output from analyze_input()
            
        Returns:
            Pipeline instance (TRELLISPipeline or Sparse2DGSPipeline)
        """
        mode = analysis_result['mode']
        
        if mode == 'GENERATIVE':
            from ..reconstruction.trellis_pipeline import TRELLISPipeline
            return TRELLISPipeline(self.config)
        else:
            from ..reconstruction.sparse2dgs_pipeline import Sparse2DGSPipeline
            return Sparse2DGSPipeline(self.config)
