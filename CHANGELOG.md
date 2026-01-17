# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-17

### Added
- Initial release of Jewelry 3D Reconstruction System
- Dual pipeline architecture (TRELLIS + Sparse2DGS)
- DUSt3R pose estimation for multi-view reconstruction
- Automatic background removal using rembg
- Smart pipeline routing based on input analysis
- 2D Gaussian splatting reconstruction
- SuGaR mesh extraction
- Poisson surface reconstruction
- PBR material extraction and composition
- Gemstone segmentation using SAM
- Gradio web interface
- Single-command Colab deployment
- Comprehensive test suite
- Full documentation (README, USAGE guide)

### Features

#### Core System
- Pipeline router for automatic mode selection
- Image preprocessing with background removal
- Topology analysis for intelligent routing
- Mesh postprocessing and cleanup

#### Reconstruction
- TRELLIS single-view generative reconstruction
- Sparse2DGS multi-view geometric reconstruction
- 2D Gaussian surfel representation
- Differentiable rendering
- Multiple loss functions (photometric, depth distortion, normal consistency)

#### Pose Estimation
- DUSt3R wrapper for pose-free camera estimation
- Global pose optimization
- Confidence-based point filtering
- Wide-baseline sparse view handling

#### Mesh Extraction
- SuGaR surface-aligned extraction
- Poisson reconstruction from point clouds
- Edge-preserving mesh refinement
- Multiple export formats (GLB, GLTF, OBJ)

#### Materials
- PBR texture baking (4K resolution)
- Albedo, roughness, metallic, and normal map extraction
- Automatic gemstone segmentation
- Material composition for GLTF export

#### UI
- Gradio web interface
- Drag-and-drop image upload
- Progress tracking
- 3D model preview
- Download output files

### Technical Specifications
- Python 3.8+ support
- CUDA acceleration
- Modular architecture
- Extensive type hints
- Google-style docstrings
- >4000 lines of production code

### Dependencies
- PyTorch >= 2.0.0
- DUSt3R (MASt3R)
- TRELLIS
- rembg (U-2-Net)
- SAM (Segment Anything)
- Gradio >= 4.0.0
- Trimesh, Open3D, PyMeshLab
- And more (see requirements.txt)

## [Unreleased]

### Planned Features
- Real-time preview during reconstruction
- Batch processing support
- Advanced material editing
- Custom model fine-tuning
- API endpoints for programmatic access
- Mobile app integration
- Cloud deployment options

---

For more details on changes, see [GitHub Releases](https://github.com/Sampatel31/jewelry-3d-reconstruction/releases).
