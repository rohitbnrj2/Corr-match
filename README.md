# Corr Match

A Python package for finding correspondences between Single-Photon Lidar frames.

## Description

This project focuses on developing algorithms and tools for establishing correspondences between frames captured by Single-Photon Lidar systems. This is useful for applications in computer vision, 3D reconstruction, and autonomous systems.

## Installation

Place the images within the corr_match/dataset/fans. You have to create dataset & fans folder. If you want to test other data points name them appropriately.

### For Development

Optional, but if you want a mamba/conda environment please just create a conda environment with python=3.12

1. Clone the repository:
```bash
git clone https://github.com/rohitbnrj2/Corr-match.git
cd corr_match
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

4. Install development dependencies: [Not used currently]
```bash
poetry install --with dev,docs
```
