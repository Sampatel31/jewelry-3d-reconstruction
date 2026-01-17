"""
Setup configuration for jewelry-3d-reconstruction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="jewelry-3d-reconstruction",
    version="0.1.0",
    author="Jewelry 3D Reconstruction Team",
    author_email="",
    description="High-fidelity 3D reconstruction system for jewelry from sparse visual data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sampatel31/jewelry-3d-reconstruction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "trimesh>=3.23.0",
        "open3d>=0.17.0",
        "pymeshlab>=2022.2",
        "rembg>=2.0.50",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "accelerate>=0.20.0",
        "gradio>=4.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "einops>=0.7.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jewelry-3d-reconstruct=src.ui.gradio_app:main",
        ],
    },
)
