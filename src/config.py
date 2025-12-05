from pathlib import Path

import torch

# paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PHC_IMAGE_ROOT = PROJECT_ROOT / "PhC-C2DH-U373"
DSB2018_TRAIN_ROOT = PROJECT_ROOT / "data-science-bowl-2018" / "stage1_train"
MODEL_SAVE_PATH = PROJECT_ROOT / "checkpoints"
WEIGHT_MAP_CACHE_DIR = PROJECT_ROOT / "weight_cache"
PLOT_DIR_BASE = PROJECT_ROOT / "plots"

# dataset selection
# options: "phc-u373" (default), "data-science-bowl-2018"
DATASET_CHOICE = "data-science-bowl-2018"
PLOT_DIR = (
    PLOT_DIR_BASE / "phc_train"
    if DATASET_CHOICE == "phc-u373"
    else PLOT_DIR_BASE / "2018_train"
)
PLOT_PREFIX = "phc-u373" if DATASET_CHOICE == "phc-u373" else "dsb2018"
# Data Science Bowl sampling
# None = use all masks; set an int to cap masks per image to reduce epoch size
DSB_MAX_MASKS_PER_IMAGE = None

# run save tag
RUN_DETAIL = "original_implementation"

# data loading
NUM_WORKERS = 12
BATCH_SIZE = 8  # adjust based on GPU memory; 8 works on H100 and 4070Ti with 572 crops
CROP_SIZE = 572

# augmentation
FLIP_PROBABILITY = 0.5
DEFORM_SIGMA = 10.0
DEFORM_POINTS = 3

# weight map
WEIGHT_MAP_W0 = 10.0
WEIGHT_MAP_SIGMA = 5.0
WEIGHT_MAP_CACHE_EXTENSION = ".npy"

# model/training (detailed in paper)
NUM_OUTPUT_CHANNELS = 2
LEARNING_RATE = 1e-4
EPOCHS = 50

# cuda configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
