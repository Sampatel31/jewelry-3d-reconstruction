"""
I/O utilities for loading and saving various file formats.
Handles images, meshes, and 3D data.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, Any
import trimesh

from .logger import get_logger

logger = get_logger(__name__)


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        img = Image.open(path)
        logger.debug(f"Loaded image: {path} ({img.size[0]}x{img.size[1]}, {img.mode})")
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def save_image(image: Image.Image, path: Union[str, Path], quality: int = 95) -> None:
    """
    Save an image to disk.
    
    Args:
        image: PIL Image object
        path: Output path
        quality: JPEG quality (1-100) if saving as JPEG
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(path, quality=quality)
        else:
            image.save(path)
        logger.debug(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image {path}: {e}")
        raise


def save_mesh(
    mesh: trimesh.Trimesh,
    path: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a mesh to disk.
    
    Args:
        mesh: Trimesh object
        path: Output path
        file_format: File format override (if None, inferred from extension)
        **kwargs: Additional arguments passed to mesh export
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format is None:
        file_format = path.suffix[1:]  # Remove leading dot
    
    try:
        mesh.export(str(path), file_type=file_format, **kwargs)
        logger.info(f"Saved mesh: {path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
    except Exception as e:
        logger.error(f"Failed to save mesh {path}: {e}")
        raise


def load_mesh(path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load a mesh from disk.
    
    Args:
        path: Path to mesh file
        
    Returns:
        Trimesh object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")
    
    try:
        mesh = trimesh.load(str(path), process=False)
        logger.debug(f"Loaded mesh: {path} ({len(mesh.vertices)} vertices)")
        return mesh
    except Exception as e:
        raise ValueError(f"Failed to load mesh {path}: {e}")


def save_point_cloud(
    points: np.ndarray,
    path: Union[str, Path],
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None
) -> None:
    """
    Save point cloud to PLY file.
    
    Args:
        points: Nx3 array of point positions
        path: Output path
        colors: Optional Nx3 array of RGB colors (0-255)
        normals: Optional Nx3 array of normal vectors
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create point cloud
    cloud = trimesh.points.PointCloud(vertices=points)
    
    if colors is not None:
        cloud.colors = colors
    
    # Save as PLY
    try:
        cloud.export(str(path))
        logger.info(f"Saved point cloud: {path} ({len(points)} points)")
    except Exception as e:
        logger.error(f"Failed to save point cloud {path}: {e}")
        raise
