"""
3D model viewer component for Gradio.
"""

import gradio as gr
from typing import Optional


def create_3d_viewer(model_path: Optional[str] = None) -> gr.Model3D:
    """
    Create a 3D model viewer component.
    
    Args:
        model_path: Optional path to 3D model file
        
    Returns:
        Gradio Model3D component
    """
    viewer = gr.Model3D(
        value=model_path,
        label="3D Preview",
        clear_color=[0.0, 0.0, 0.0, 0.0],
        camera_position=(1.5, 1.5, 1.5)
    )
    
    return viewer
