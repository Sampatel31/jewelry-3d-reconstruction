# Example Images

This directory contains example images for testing the jewelry 3D reconstruction system.

## Usage

Place your jewelry images here for testing:

```
examples/
├── necklace_example.jpg       # Single-view necklace
├── ring_view1.jpg             # Ring from front
├── ring_view2.jpg             # Ring from side
└── ring_view3.jpg             # Ring from top
```

## Image Requirements

### Single-View (Generative Pipeline)
- Resolution: 1024x1024 or higher
- Format: JPG, PNG, or WebP
- Background: Clean, preferably white
- Lighting: Even, diffused lighting
- Focus: Sharp, clear details

### Multi-View (Geometric Pipeline)
- Number of views: 3-5 images
- Angles: Cover different sides (front, side, top)
- Consistency: Same lighting across all views
- Overlap: Sufficient overlap between views
- Background: Clean background (auto-removed)

## Example Test Command

```python
from src.core.pipeline_router import PipelineRouter
from config.pipeline_config import PipelineConfig

config = PipelineConfig()
router = PipelineRouter(config.to_dict())

# Test with example images
image_paths = [
    'examples/ring_view1.jpg',
    'examples/ring_view2.jpg',
    'examples/ring_view3.jpg'
]

analysis = router.analyze_input(image_paths)
print(f"Pipeline mode: {analysis['mode']}")
print(f"Topology: {analysis['metadata']['topology_type']}")
```

## Sample Images

To add sample images, either:
1. Add your own jewelry photos
2. Download from public datasets
3. Use the provided test script to generate synthetic examples

Note: This directory is empty by default. Add your own images for testing.
