#!/usr/bin/env python3
"""
Single-command launcher for Google Colab deployment.
This script installs all dependencies and launches the Gradio interface.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install all required dependencies."""
    print("📦 Installing dependencies...")
    
    # Install core dependencies
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "numpy", "pillow", "opencv-python",
        "scipy", "scikit-image", "trimesh", "open3d", "pymeshlab",
        "rembg", "transformers", "diffusers", "accelerate",
        "gradio", "pyyaml", "tqdm", "requests", "einops",
        "imageio", "imageio-ffmpeg"
    ])
    
    # Install DUSt3R from GitHub
    print("📦 Installing DUSt3R...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/naver/dust3r.git"
        ])
    except:
        print("⚠️ DUSt3R installation failed (optional)")
    
    print("✅ Dependencies installed")


def setup_environment():
    """Setup environment and download models."""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    print("✅ Environment ready")


def launch_app():
    """Launch the Gradio application."""
    print("🚀 Launching Jewelry 3D Reconstruction System...")
    
    from src.ui.gradio_app import main
    main()


def main():
    """Main entry point for Colab deployment."""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   Jewelry 3D Reconstruction System                   ║
    ║   Single-Command Colab Deployment                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        print("✅ Running in Google Colab")
    except ImportError:
        in_colab = False
        print("ℹ️ Not running in Colab (local mode)")
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Launch app
    launch_app()


if __name__ == "__main__":
    main()
