# 🔷 Jewelry 3D Reconstruction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready system for automated high-fidelity 3D reconstruction of jewelry from sparse visual data (1-5 images). Features dual pipeline architecture with real AI-powered generative and geometric reconstruction, automatic background removal, and PBR material extraction.

## ✨ Features

### Core Capabilities

- **🎨 Dual Pipeline Architecture**
  - **Pipeline A**: Single-view generative reconstruction using **TripoSR** for necklaces/earrings
  - **Pipeline B**: Multi-view geometric reconstruction using DUSt3R + Sparse2DGS for rings/bangles

- **📸 Smart Input Processing**
  - Automatic background removal (rembg/U-2-Net)
  - Intelligent pipeline routing based on input count
  - Support for 1-5 input images

- **🎯 Pose Estimation** (Pipeline B)
  - DUSt3R pose-free initialization
  - Handles wide-baseline sparse views
  - Confidence-based point filtering

- **🏗️ High-Quality Reconstruction**
  - **TripoSR**: Fast feed-forward 3D generation from Stability AI
  - 2D Gaussian surfels for geometric accuracy
  - Depth distortion and normal consistency regularization
  - Edge-preserving mesh extraction

- **💎 Material Processing**
  - PBR material extraction (Albedo, Roughness, Metallic, Normal)
  - Automatic gemstone segmentation
  - Physically-based material assignment

- **📦 Export Options**
  - GLB/GLTF with embedded PBR materials
  - OBJ + MTL export
  - High-resolution texture baking (4K)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Sampatel31/jewelry-3d-reconstruction.git
cd jewelry-3d-reconstruction

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from src.ui.gradio_app import main

# Launch Gradio interface
main()
```

Or use the command-line interface:

```bash
python -m src.ui.gradio_app
```

### Google Colab Deployment

For single-command deployment on Google Colab:

```python
!git clone https://github.com/Sampatel31/jewelry-3d-reconstruction.git
%cd jewelry-3d-reconstruction
!python run_colab.py
```

See [USAGE.md](USAGE.md) for detailed deployment instructions.

## 📁 Project Structure

```
jewelry-3d-reconstruction/
├── config/                  # Configuration files
│   ├── pipeline_config.py   # Pipeline hyperparameters
│   └── model_config.py      # Model paths and settings
├── src/
│   ├── core/               # Core processing modules
│   │   ├── pipeline_router.py    # Input analysis & routing
│   │   ├── preprocessor.py       # Background removal
│   │   └── postprocessor.py      # Mesh cleanup
│   ├── pose_estimation/    # Camera pose estimation
│   │   ├── dust3r_wrapper.py     # DUSt3R integration
│   │   ├── global_optimizer.py   # Pose optimization
│   │   └── confidence_masking.py # Confidence filtering
│   ├── reconstruction/     # 3D reconstruction pipelines
│   │   ├── trellis_pipeline.py   # Single-view generative
│   │   ├── sparse2dgs_pipeline.py # Multi-view geometric
│   │   ├── gaussian_model.py     # 2D Gaussian representation
│   │   ├── renderer.py           # Differentiable rendering
│   │   └── losses.py             # Loss functions
│   ├── mesh_extraction/    # Surface extraction
│   │   ├── sugar_extractor.py    # SuGaR mesh extraction
│   │   ├── poisson_reconstruction.py
│   │   └── mesh_refinement.py
│   ├── materials/          # Material processing
│   │   ├── mate_wrapper.py       # Material extraction
│   │   ├── pbr_baker.py          # Texture baking
│   │   ├── gemstone_segmenter.py # Gem detection
│   │   └── material_composer.py  # GLTF materials
│   ├── utils/              # Utility functions
│   └── ui/                 # User interface
│       ├── gradio_app.py         # Main Gradio UI
│       └── viewer_3d.py          # 3D viewer
├── models/                 # Model weights
├── tests/                  # Test suite
├── setup.py               # Package configuration
├── requirements.txt       # Dependencies
└── run_colab.py          # Colab launcher

```

## 🔧 Pipeline Details

### Pipeline A: Single-View Generative (TRELLIS)

Best for: Necklaces, earrings, brooches (planar jewelry)

**Process:**
1. Background removal and preprocessing
2. SLAT-based 3D generation from single image
3. Surface mesh extraction
4. PBR material composition

**Input:** 1-2 images  
**Output:** Textured 3D mesh with hallucinated backside

### Pipeline B: Multi-View Geometric (Sparse2DGS)

Best for: Rings, bangles, bracelets (volumetric jewelry)

**Process:**
1. Background removal and preprocessing
2. DUSt3R pose estimation (pose-free)
3. 2D Gaussian Splatting optimization
4. Surface-aligned mesh extraction via SuGaR
5. PBR material baking from multi-view

**Input:** 3-5 images  
**Output:** Geometrically accurate textured mesh

## 🎮 Usage Examples

### Programmatic API

```python
from config.pipeline_config import PipelineConfig
from src.core.pipeline_router import PipelineRouter

# Initialize with custom config
config = PipelineConfig(
    background_removal=True,
    num_iterations=30000,
    pbr_resolution=4096,
    output_format='glb'
)

# Create router
router = PipelineRouter(config.to_dict())

# Analyze input images
image_paths = ['ring_view1.jpg', 'ring_view2.jpg', 'ring_view3.jpg']
analysis = router.analyze_input(image_paths)

# Get appropriate pipeline
pipeline = router.route(analysis)

# Run reconstruction
if analysis['mode'] == 'GEOMETRIC':
    # Multi-view reconstruction
    from src.pose_estimation.dust3r_wrapper import DUSt3RPoseEstimator
    import numpy as np
    
    # Load images
    images = [np.array(img.convert('RGB')) for img in analysis['cleaned_images']]
    
    # Estimate poses
    pose_estimator = DUSt3RPoseEstimator()
    pose_result = pose_estimator.estimate_poses(images)
    
    # Reconstruct
    result = pipeline.reconstruct(
        cameras=pose_result['cameras'],
        point_cloud=pose_result['point_cloud'],
        colors=pose_result['colors']
    )
else:
    # Single-view reconstruction
    result = pipeline.reconstruct(analysis['cleaned_images'][0])

# Extract mesh
mesh = result['mesh']

# Save output
from src.utils.io_utils import save_mesh
save_mesh(mesh, 'output.glb', file_format='glb')
```

## 📊 System Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- 4GB VRAM (GPU recommended)
- 10GB disk space

### Recommended

- Python 3.10+
- 16GB RAM
- 12GB VRAM (NVIDIA GPU with CUDA)
- 20GB disk space

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline_a.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{jewelry3d2024,
  title={Jewelry 3D Reconstruction System},
  author={Jewelry 3D Reconstruction Team},
  year={2024},
  url={https://github.com/Sampatel31/jewelry-3d-reconstruction}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This system integrates several state-of-the-art methods:

- **DUSt3R/MASt3R**: [naver/dust3r](https://github.com/naver/dust3r)
- **TRELLIS**: [JeffreyXiang/TRELLIS](https://huggingface.co/JeffreyXiang/TRELLIS-image-large)
- **rembg**: [danielgatis/rembg](https://github.com/danielgatis/rembg)
- **SAM**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.