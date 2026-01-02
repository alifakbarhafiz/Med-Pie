"""
Configuration constants for Med-Pie application.
"""
import os
from pathlib import Path

# Model Configuration
# Model weights are stored in models/weights/ directory (relative to project root)
# This makes the project portable - works anywhere
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, 
    "models", 
    "weights", 
    "tb_detection_model.pth"
)

# If the local file doesn't exist, try alternative locations
if not os.path.exists(MODEL_WEIGHTS_PATH):
    # Try Colab path (for Google Colab environments)
    colab_path = "/content/drive/MyDrive/Perbinusan/TB_Project/tb_detector.pth"
    if os.path.exists(colab_path):
        MODEL_WEIGHTS_PATH = colab_path
    else:
        # Keep the relative path - user will need to place model there
        pass

# Image Processing
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model Architecture
TB_TYPE_VALUES = ['none', 'latent_tb', 'active_tb']
NUM_DET_CLASSES = 3  # 3 classes: none, latent_tb, active_tb (matches checkpoint)
CLS_NUM_OUTPUTS = 3  # none, latent_tb, active_tb

# Color Scheme (Apple Design System)
COLORS = {
    'background_soft': '#F5F5F7',
    'background_white': '#FFFFFF',
    'accent_blue': '#007AFF',
    'success_green': '#34C759',
    'warning_orange': '#FF9500',
    'critical_red': '#FF3B30',
    'text_primary': '#1D1D1F',
    'text_secondary': '#86868B',
}

# TB Classification Colors
TB_COLORS = {
    0: COLORS['success_green'],   # none
    1: COLORS['accent_blue'],     # latent_tb
    2: COLORS['critical_red']     # active_tb
}

# UI Constants
BORDER_RADIUS = 12  # 12px rounded corners
SHADOW_BLUR = 20
ANIMATION_DURATION = 0.3

# Default Settings
DEFAULT_NMS_THRESHOLD = 0.5
DEFAULT_SCORE_THRESHOLD = 0.5

