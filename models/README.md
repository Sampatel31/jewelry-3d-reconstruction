# Model Weights

This directory contains scripts for downloading pretrained model weights.

## Models Used

### 1. DUSt3R (MASt3R)
- **Source**: [naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)
- **Purpose**: Pose-free camera estimation for multi-view reconstruction
- **License**: Apache 2.0

### 2. TRELLIS
- **Source**: [JeffreyXiang/TRELLIS-image-large](https://huggingface.co/JeffreyXiang/TRELLIS-image-large)
- **Purpose**: Single-view generative 3D reconstruction
- **License**: Apache 2.0

### 3. SAM (Segment Anything)
- **Source**: [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge)
- **Purpose**: Gemstone segmentation
- **License**: Apache 2.0

### 4. rembg (U-2-Net)
- **Source**: u2net model (downloaded automatically)
- **Purpose**: Background removal
- **License**: Apache 2.0

## Download

Models are automatically downloaded on first use via Hugging Face transformers.

To pre-download all models:

```bash
python download_models.py
```

## Cache Location

Models are cached in:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`
