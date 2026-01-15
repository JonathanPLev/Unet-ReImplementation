# U-Net Reimplementation for Cell Segmentation

This project provides a comprehensive reimplementation of the U-Net architecture for biomedical image segmentation, specifically focusing on cell segmentation tasks. The implementation includes both the original U-Net architecture and improved versions with modern optimizations.

## Project Overview

U-Net is a convolutional neural network architecture designed for biomedical image segmentation. This project reimplements the original paper's architecture and extends it with modern training techniques and optimizations. The implementation achieves competitive performance on cell segmentation datasets including the PhC-C2DH-U373 dataset and the Data Science Bowl 2018 challenge.

### Key Features

- Original U-Net architecture implementation
- Improved U-Net with modern optimizers and techniques
- Support for multiple datasets (PhC-C2DH-U373, Data Science Bowl 2018)
- Comprehensive training and inference pipelines
- Performance evaluation and visualization tools

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with at least 6GB VRAM (recommended)
- CUDA 12.1+ (for GPU acceleration)

### Using UV (Recommended)

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Resnet-ReImplementation
   ```

3. **Create and sync the environment**:
   ```bash
   uv sync
   ```

4. **Pull large files** (for original model checkpoints):
   ```bash
   git lfs install
   git lfs pull
   ```

### Running the Code

To run any Python file in the appropriate environment:
```bash
cd <directory>
uv run python <script_name>.py
```

## Quick Start

### 1. Dataset Setup

#### PhC-C2DH-U373 Dataset
1. Download the PhC-C2DH-U373.zip file
2. Extract it to `data/PhC-C2DH-U373/`
3. The dataset contains:
   - `.tif` files as input images
   - `man_seg000.tif` files as ground truth labels

#### Data Science Bowl 2018 Dataset
1. Download `stage1_train.zip` from [Kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
2. Extract to `data-science-bowl-2018/stage1_train/`
3. Ensure structure: `data-science-bowl-2018/stage1_train/<image_id>/{images,masks}`

### 2. Configuration

In `src/config.py`, set your preferred dataset:
```python
DATASET_CHOICE = "phc-u373"  # or "data-science-bowl-2018"
```

### 3. Training

#### Train on PhC-C2DH-U373
```bash
uv run python -m src.main
```

#### Train on Data Science Bowl 2018
```bash
uv run python -m src.main
# or alternatively:
uv run python -m src.train_2018
```

### 4. Testing and Inference

#### Original U-Net Inference
```bash
cd orig_implementation_src
uv run python inference.py
```
An example inference result can be found in `orig_implementation_src/inference_example/inference_example.png`.

#### Testing Trained Models
Run the appropriate test scripts in the `src/` directory to evaluate model performance on test datasets.

## Results

### Performance Metrics

The original U-Net implementation achieves:
- **Dice Score**: 92%
- **IoU Score**: 82%

*Note: The final scores may differ from the original paper due to the absence of cross-validation for optimal train/test splits.*

### Visualization

Result plots and graphs are available in `orig_implementation_src/orig_impl_plots/`. Use `create_plots.py` to generate additional visualizations of training progress and model performance.

## Report

For detailed methodology, experimental results, and analysis, refer to our project report:
- **Project Report**: `ECS 174 Project_ U-Net Presentation.pdf`

## Project Structure

```
Resnet-ReImplementation/
├── src/                          # Main implementation
├── orig_implementation_src/      # Original U-Net implementation
├── orig_impl_checkpoints/        # Pre-trained model checkpoints
├── data/                         # Dataset storage
├── dsb2018_unet_implementation/  # Data Science Bowl specific implementation
├── unet_implementation_with_optimizers/  # Optimized versions
└── create_plots.py              # Visualization utilities
```

## System Requirements

- **Python**: 3.11+
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
- **CUDA**: 12.1+
- **PyTorch**: 2.9.1+

## Dependencies

Key dependencies include:
- PyTorch & torchvision
- NumPy
- Matplotlib
- Elastic deform (for data augmentation)
- Kaggle API (for dataset download)

All dependencies are automatically managed by UV through `pyproject.toml`.
