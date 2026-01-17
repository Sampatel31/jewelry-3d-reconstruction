# 📖 Usage Guide: Google Colab Deployment

This guide provides step-by-step instructions for deploying the Jewelry 3D Reconstruction System on Google Colab.

## 🚀 Single-Command Deployment

### Option 1: Using run_colab.py (Recommended)

Open a new Google Colab notebook and run:

```python
# Clone repository
!git clone https://github.com/Sampatel31/jewelry-3d-reconstruction.git
%cd jewelry-3d-reconstruction

# Run single-command launcher
!python run_colab.py
```

This will:
1. Install all dependencies
2. Download required models
3. Launch the Gradio interface
4. Provide a public URL for access

### Option 2: Manual Installation

If you prefer manual control:

```python
# 1. Clone repository
!git clone https://github.com/Sampatel31/jewelry-3d-reconstruction.git
%cd jewelry-3d-reconstruction

# 2. Install core dependencies
!pip install -q torch torchvision numpy pillow opencv-python scipy scikit-image
!pip install -q trimesh open3d pymeshlab rembg transformers diffusers accelerate
!pip install -q gradio pyyaml tqdm requests einops imageio imageio-ffmpeg

# 3. Install DUSt3R (optional, for multi-view)
!pip install -q git+https://github.com/naver/dust3r.git

# 4. Install package
!pip install -e .

# 5. Launch application
from src.ui.gradio_app import main
main()
```

## 🎯 Using the Interface

### Step 1: Upload Images

1. Click the "Upload Images" area
2. Select 1-5 images of your jewelry piece
3. Supported formats: JPG, PNG, WebP
4. Recommended: Use 3-5 well-lit images from different angles

### Step 2: Configure Settings

**Output Format:**
- **GLB** (recommended): Single binary file with embedded textures
- **GLTF**: JSON + separate texture files
- **OBJ**: Classic format with MTL material file

**Extract PBR Materials:**
- ✅ Enabled: Full PBR material extraction (albedo, roughness, metallic, normal)
- ❌ Disabled: Basic texture only (faster)

### Step 3: Reconstruct

1. Click "🚀 Reconstruct 3D Model"
2. Wait for processing (2-10 minutes depending on:
   - Number of images
   - Selected pipeline
   - Material extraction settings
3. Download the result when complete

## 📸 Photography Tips

### Lighting
- Use diffused natural light or softbox lighting
- Avoid harsh shadows and specular highlights
- Maintain consistent lighting across all views

### Background
- Plain white or neutral background works best
- Background removal works automatically
- Ensure jewelry is clearly separated from background

### Camera Angles (Multi-View)

For optimal results with 3+ images:

```
View 1: Front view (0°)
View 2: Side view (90°)
View 3: Top view (from above)
View 4: Opposite side (180°) [optional]
View 5: Bottom-angled view [optional]
```

### Single-View Tips

For 1-2 images (generative mode):
- Capture the most important/visible side
- Ensure all key features are visible
- Use high resolution (1024x1024 or higher)

## 🎨 Pipeline Selection

The system automatically selects the appropriate pipeline:

### Generative Pipeline (1-2 images)
- **Best for**: Necklaces, earrings, brooches
- **Advantages**: Works with single image
- **Limitations**: Backside is hallucinated
- **Processing time**: 2-5 minutes

### Geometric Pipeline (3+ images)
- **Best for**: Rings, bangles, bracelets
- **Advantages**: Geometrically accurate
- **Limitations**: Requires multiple views
- **Processing time**: 5-10 minutes

## 🔧 Advanced Configuration

### Custom Pipeline Config

```python
from config.pipeline_config import PipelineConfig

# Create custom configuration
config = PipelineConfig(
    # Input processing
    background_removal=True,
    image_size=(1024, 1024),
    
    # Reconstruction
    num_iterations=30000,
    learning_rate=0.0025,
    
    # Materials
    pbr_resolution=4096,
    enable_gemstone_detection=True,
    metallic_threshold=0.7,
    
    # Output
    output_format='glb',
    texture_size=4096,
    
    # Device
    device='cuda',
    verbose=True
)

# Use with application
from src.ui.gradio_app import JewelryReconstructionApp
app = JewelryReconstructionApp(config)
```

### Model Downloads

Models are automatically downloaded on first use. To pre-download:

```python
%cd jewelry-3d-reconstruction
!python models/download_models.py
```

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```python
# Reduce resolution
config = PipelineConfig(
    image_size=(512, 512),
    pbr_resolution=2048,
    texture_size=2048
)
```

### Issue: DUSt3R Installation Fails

**Solution:**
```python
# Use single-view mode only
# Upload 1-2 images instead of 3+
# System will automatically use TRELLIS pipeline
```

### Issue: Slow Processing

**Solution:**
- Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU)
- Reduce number of iterations: `num_iterations=15000`
- Disable material extraction for faster results

### Issue: Model Download Timeout

**Solution:**
```python
# Set longer timeout
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

# Or use cached models if available
from config.model_config import ModelConfig
model_config = ModelConfig()
```

## 📊 Performance Benchmarks

On Google Colab with T4 GPU:

| Configuration | Time | Quality |
|--------------|------|---------|
| 1 image, no materials | 2-3 min | Good |
| 1 image, with materials | 3-5 min | Excellent |
| 3 images, no materials | 5-7 min | Excellent |
| 3 images, with materials | 7-10 min | Outstanding |
| 5 images, full quality | 10-15 min | Best |

## 💡 Best Practices

### Do's
✅ Use high-resolution images (1024x1024+)  
✅ Ensure good lighting  
✅ Capture multiple angles for volumetric items  
✅ Clean background or use auto-removal  
✅ Keep jewelry in focus  

### Don'ts
❌ Don't use blurry images  
❌ Don't mix different lighting conditions  
❌ Don't include multiple jewelry pieces  
❌ Don't use extreme angles  
❌ Don't use images with heavy shadows  

## 🎓 Example Workflows

### Workflow 1: Quick Ring Reconstruction

```python
# Upload 3 images: front, side, top
# Output format: GLB
# Materials: Enabled
# Click Reconstruct
# Download GLB file
# Import into Blender/Unity/Unreal
```

### Workflow 2: High-Quality Necklace

```python
# Upload 1 high-res image
# Output format: GLTF
# Materials: Enabled with gemstone detection
# Click Reconstruct
# Download GLTF + textures
# Fine-tune materials in 3D software
```

### Workflow 3: Batch Processing

```python
from src.ui.gradio_app import JewelryReconstructionApp
import glob

app = JewelryReconstructionApp()

# Process multiple items
for item_dir in glob.glob('items/*'):
    images = glob.glob(f'{item_dir}/*.jpg')
    output_file, status = app.reconstruct(images)
    print(f"Processed {item_dir}: {status}")
```

## 📱 Sharing Your Results

After reconstruction:

1. **View in Browser**: Open GLB files directly in browser
2. **Share Link**: Use Sketchfab, Poly Haven, or similar
3. **AR Preview**: Use model viewers for AR preview
4. **3D Printing**: Export to STL for 3D printing

## 🔗 Additional Resources

- [DUSt3R Documentation](https://github.com/naver/dust3r)
- [TRELLIS Documentation](https://huggingface.co/JeffreyXiang/TRELLIS-image-large)
- [Gradio Documentation](https://gradio.app/docs/)
- [Trimesh Documentation](https://trimsh.org/)

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/Sampatel31/jewelry-3d-reconstruction/issues)
- Check the [Troubleshooting](#-troubleshooting) section
- Review the [FAQ](https://github.com/Sampatel31/jewelry-3d-reconstruction/wiki/FAQ)
