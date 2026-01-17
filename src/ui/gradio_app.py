"""
Gradio interface for jewelry 3D reconstruction.
Provides user-friendly UI for uploading images and downloading results.
"""

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
from typing import List, Optional, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.pipeline_config import PipelineConfig
from src.core.pipeline_router import PipelineRouter
from src.utils.logger import setup_logging, get_logger
from src.utils.io_utils import save_mesh

logger = get_logger(__name__)


class JewelryReconstructionApp:
    """
    Main application class for jewelry 3D reconstruction.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize application.
        
        Args:
            config: Optional pipeline configuration
        """
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        self.router = PipelineRouter(config.to_dict())
        
        logger.info("Jewelry reconstruction app initialized")
    
    def reconstruct(
        self,
        images: List,
        output_format: str = "glb",
        enable_materials: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Reconstruct 3D model from input images.
        
        Args:
            images: List of uploaded images
            output_format: Output file format ('glb', 'gltf', or 'obj')
            enable_materials: Whether to extract PBR materials
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (output_file_path, status_message)
        """
        try:
            logger.info(f"Starting reconstruction with {len(images)} images...")
            progress(0.1, desc="Analyzing input images...")
            
            # Save uploaded images temporarily
            temp_dir = Path(tempfile.mkdtemp())
            image_paths = []
            
            for idx, img in enumerate(images):
                if isinstance(img, (str, Path)):
                    # Handle file path (string or Path object)
                    image_paths.append(str(img))
                elif isinstance(img, dict):
                    # Handle file upload dict
                    img_path = img.get('name', img.get('path'))
                    image_paths.append(img_path)
                else:
                    # Handle PIL Image or numpy array
                    img_path = temp_dir / f"input_{idx}.png"
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    img.save(img_path)
                    image_paths.append(str(img_path))
            
            # Analyze input and route to appropriate pipeline
            progress(0.2, desc="Routing to reconstruction pipeline...")
            analysis = self.router.analyze_input([str(p) for p in image_paths])
            
            pipeline = self.router.route(analysis)
            
            # Run reconstruction
            if analysis['mode'] == 'GENERATIVE':
                progress(0.3, desc="Running single-view reconstruction (TRELLIS)...")
                result = pipeline.reconstruct(analysis['cleaned_images'][0])
            else:
                progress(0.3, desc="Estimating camera poses (DUSt3R)...")
                
                # Convert images to numpy arrays
                images_np = [np.array(img.convert('RGB')) for img in analysis['cleaned_images']]
                
                # Estimate poses
                from src.pose_estimation.dust3r_wrapper import DUSt3RPoseEstimator
                pose_estimator = DUSt3RPoseEstimator(device=self.config.device)
                
                pose_result = pose_estimator.estimate_poses(images_np)
                
                progress(0.5, desc="Running multi-view reconstruction (Sparse2DGS)...")
                result = pipeline.reconstruct(
                    cameras=pose_result['cameras'],
                    point_cloud=pose_result['point_cloud'],
                    colors=pose_result['colors']
                )
            
            # Extract mesh
            progress(0.7, desc="Extracting surface mesh...")
            mesh = result['mesh']
            
            # Apply materials if enabled
            if enable_materials:
                progress(0.8, desc="Extracting PBR materials...")
                
                from src.materials.pbr_baker import PBRBaker
                from src.materials.material_composer import MaterialComposer
                
                baker = PBRBaker(self.config.to_dict())
                composer = MaterialComposer(self.config.to_dict())
                
                # Bake textures
                texture_maps = baker.bake_textures(
                    mesh,
                    [np.array(img) for img in analysis['cleaned_images']]
                )
                
                # Compose materials
                mesh = composer.compose_materials(mesh, texture_maps)
            
            # Save output
            progress(0.9, desc="Saving output file...")
            output_path = temp_dir / f"reconstructed.{output_format}"
            save_mesh(mesh, output_path, file_format=output_format)
            
            progress(1.0, desc="Complete!")
            
            status = f"""
            ✅ Reconstruction successful!
            
            Pipeline: {analysis['mode']}
            Topology: {analysis['metadata']['topology_type']}
            Vertices: {len(mesh.vertices)}
            Faces: {len(mesh.faces)}
            Format: {output_format.upper()}
            """
            
            logger.info("Reconstruction complete")
            return str(output_path), status
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}", exc_info=True)
            error_msg = f"❌ Reconstruction failed: {str(e)}"
            return None, error_msg


def create_gradio_interface() -> gr.Blocks:
    """
    Create Gradio interface for the application.
    
    Returns:
        Gradio Blocks interface
    """
    # Initialize app
    app = JewelryReconstructionApp()
    
    # Create interface
    with gr.Blocks(title="Jewelry 3D Reconstruction", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔷 Jewelry 3D Reconstruction System
        
        Upload 1-5 images of jewelry to create a high-fidelity 3D model with PBR materials.
        
        **Pipeline Selection:**
        - **1-2 images**: Single-view generative reconstruction (TRELLIS)
        - **3+ images**: Multi-view geometric reconstruction (Sparse2DGS + DUSt3R)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                image_input = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload Images (1-5)",
                    type="filepath"
                )
                
                with gr.Row():
                    format_dropdown = gr.Dropdown(
                        choices=["glb", "gltf", "obj"],
                        value="glb",
                        label="Output Format"
                    )
                    
                    materials_checkbox = gr.Checkbox(
                        value=True,
                        label="Extract PBR Materials"
                    )
                
                reconstruct_btn = gr.Button("🚀 Reconstruct 3D Model", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                
                output_file = gr.File(label="Download 3D Model")
                status_text = gr.Textbox(label="Status", lines=8)
        
        gr.Markdown("""
        ### Tips
        - For best results, use 3-5 well-lit images from different angles
        - Ensure jewelry is clearly visible against a clean background
        - Supported items: rings, necklaces, earrings, bangles
        - Output includes high-resolution textures (4K)
        """)
        
        # Connect button to function
        reconstruct_btn.click(
            fn=app.reconstruct,
            inputs=[image_input, format_dropdown, materials_checkbox],
            outputs=[output_file, status_text]
        )
    
    return interface


def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting Jewelry 3D Reconstruction System...")
    
    # Detect if running in Colab
    try:
        import google.colab
        in_colab = True
        logger.info("Running in Google Colab - enabling public link sharing")
    except ImportError:
        in_colab = False
    
    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=in_colab  # Auto-enable sharing in Colab
    )


if __name__ == "__main__":
    main()
