# Contributing to Jewelry 3D Reconstruction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/jewelry-3d-reconstruction.git
cd jewelry-3d-reconstruction
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## 📝 Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints for function signatures
- Write Google-style docstrings
- Maximum line length: 100 characters

Example:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### Code Formatting

Use black and flake8:
```bash
black src/ tests/
flake8 src/ tests/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline_a.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Write tests for all new features
- Maintain >80% code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:
```python
def test_feature_name():
    # Arrange
    config = PipelineConfig()
    
    # Act
    result = function_under_test(config)
    
    # Assert
    assert result is not None
    assert result.property == expected_value
```

## 📦 Adding New Features

### Feature Branch Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Add tests
4. Update documentation
5. Commit with descriptive message:
```bash
git commit -m "Add feature: brief description"
```

6. Push to your fork:
```bash
git push origin feature/your-feature-name
```

7. Create a Pull Request

### Pull Request Guidelines

- Describe what the PR does
- Link related issues
- Include screenshots for UI changes
- Ensure all tests pass
- Update CHANGELOG.md

## 🏗️ Architecture

### Module Structure

```
src/
├── core/           # Core processing (routing, preprocessing)
├── pose_estimation/ # Camera pose estimation
├── reconstruction/ # 3D reconstruction pipelines
├── mesh_extraction/ # Surface extraction
├── materials/      # Material processing
├── utils/          # Utility functions
└── ui/            # User interface
```

### Adding a New Pipeline

1. Create pipeline file in `src/reconstruction/`
2. Inherit from base pipeline class
3. Implement `reconstruct()` method
4. Add to router in `pipeline_router.py`
5. Write tests in `tests/`
6. Update documentation

## 🐛 Bug Reports

### Before Reporting

- Search existing issues
- Try to reproduce with minimal example
- Check if it's already fixed in main branch

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 3090]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

Use the Feature Request template:

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How would you implement this?

**Alternatives**
Other approaches considered

**Additional Context**
Mockups, references, etc.
```

## 📖 Documentation

### Updating Documentation

- Update README.md for major changes
- Update USAGE.md for user-facing changes
- Add docstrings to all public functions
- Include code examples where helpful

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep it up-to-date

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make this project better for everyone!
