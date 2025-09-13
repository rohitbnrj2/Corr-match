# SPAD Investigations 

A Python package for finding SPAD research gaps, essentially optimized for investigations.

## Description

This package focuses on simulating SPAD images, testing Correspondence with SIFT, Inpainting SPAD binary frames with diffusion.

## Installation

Place the images within the corr_match/dataset/fans. You have to create dataset & fans folder. If you want to test other data points name them appropriately.

### For Development

Optional, but if you want a mamba/conda environment please just create a conda environment with python=3.12

1. Clone the repository:
```bash
git clone https://github.com/rohitbnrj2/spad_investigations.git
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
