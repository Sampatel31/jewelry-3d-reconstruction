"""
Geometry utilities for 3D operations.
Handles transformations, bounding boxes, and geometric computations.
"""

import numpy as np
from typing import Tuple, Optional
import trimesh

from .logger import get_logger

logger = get_logger(__name__)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply homogeneous transformation to points.
    
    Args:
        points: Nx3 array of points
        transform: 4x4 transformation matrix
        
    Returns:
        Transformed Nx3 array of points
    """
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply transformation
    transformed_h = points_h @ transform.T
    
    # Convert back to 3D
    transformed = transformed_h[:, :3] / transformed_h[:, 3:]
    return transformed


def compute_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        points: Nx3 array of points
        
    Returns:
        Tuple of (min_corner, max_corner) each as 3D arrays
    """
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    return min_corner, max_corner


def normalize_points(points: np.ndarray, scale: Optional[float] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Normalize points to unit sphere.
    
    Args:
        points: Nx3 array of points
        scale: Optional fixed scale (if None, computed from data)
        
    Returns:
        Tuple of (normalized_points, scale_factor, centroid)
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center points
    centered = points - centroid
    
    # Compute scale
    if scale is None:
        scale = np.max(np.linalg.norm(centered, axis=1))
    
    # Normalize
    normalized = centered / scale
    
    return normalized, scale, centroid


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute vertex normals from mesh faces.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        
    Returns:
        Nx3 array of vertex normals
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.vertex_normals


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates vec1 to vec2.
    
    Args:
        vec1: Source vector (3D)
        vec2: Target vector (3D)
        
    Returns:
        3x3 rotation matrix
    """
    # Normalize vectors
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    
    # Compute rotation axis and angle
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    # Handle parallel vectors
    if np.allclose(v, 0):
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            v = np.cross(a, perp)
            v = v / np.linalg.norm(v)
            return 2 * np.outer(v, v) - np.eye(3)
    
    # Rodrigues' rotation formula
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    
    return rotation_matrix


def camera_to_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert camera extrinsics (R, t) to world-to-camera transform.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def world_to_camera(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation and translation from world-to-camera transform.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Tuple of (R, t) where R is 3x3 rotation and t is 3x1 translation
    """
    R = T[:3, :3]
    t = T[:3, 3:4]
    return R, t
