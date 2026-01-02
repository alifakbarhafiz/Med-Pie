# Med-Pie - Professional Medical Imaging Demonstration Platform

Med-Pie is a professional-grade medical imaging demonstration platform that utilizes a Multi-Task Deep Learning model (ConvNeXt-Tiny + FPN + Faster R-CNN) to perform simultaneous TB Detection, Multi-class Classification, and Severity Estimation on Chest X-rays.

## Features

- **Multi-Task Analysis**: Simultaneous detection, classification, and severity estimation
- **Apple-Inspired Design**: Minimalist, clean, and clinical UI with glassmorphism effects
- **Grad-CAM Explainability**: Visual heatmaps showing model attention
- **Clinical Reports**: Automated PDF report generation
- **Real-time Analysis**: Fast inference with GPU acceleration support

## Project Structure

```
B_demo_v2/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore patterns
├── DOCKER_README.md       # Docker deployment guide
├── setup.bat / setup.sh   # Quick setup wrappers
├── run.bat / run.sh       # Quick run wrappers
├── models/
│   ├── __init__.py
│   ├── lumina_inference.py  # Model loading and inference
│   └── weights/           # Model weights directory
│       └── tb_detection_model.pth  # Place your model here
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing utilities
│   ├── visualization.py     # Bounding box visualization
│   └── gradcam.py           # Grad-CAM implementation
├── components/
│   ├── __init__.py
│   ├── sidebar.py           # Sidebar UI component
│   └── cards.py             # Result card components
├── styles/
│   └── custom.css           # Apple design system CSS
├── reports/
│   ├── __init__.py
│   └── pdf_generator.py     # PDF report generation
├── scripts/                # Setup and utility scripts
│   ├── setup_portable.bat/sh  # Portable setup scripts
│   ├── run.bat/sh/ps1        # Run scripts
│   ├── validate_setup.py     # Validation utilities
│   └── ...
├── docs/                   # Documentation
│   ├── QUICKSTART.md
│   ├── GIT_BASH_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── ...
├── reference/              # Reference implementation files
│   ├── tb_inference.py     # Original inference script
│   └── tb_train.py         # Original training script
└── tests/                  # Test suite
    ├── unit/
    └── integration/
```

## Installation

### Quick Setup (Portable - Recommended)

**Windows:**
```cmd
setup.bat
run.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

This creates a virtual environment and installs everything automatically.

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Verify installation:**
   ```bash
   python scripts/verify_installation.py
   ```

5. **Place your model weights**:
   - Place your trained model checkpoint as `tb_detection_model.pth` in `models/weights/` directory

See **[docs/README_PORTABLE.md](docs/README_PORTABLE.md)** for detailed portable setup instructions.

## Quick Start

### Docker Deployment (Fastest)

If you have Docker installed, you can run the entire application with one command:

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or using Docker directly
docker build -t med-pie:latest .
docker run -d -p 8501:8501 --name med-pie-app med-pie:latest
```

The app will be available at `http://localhost:8501`

See **[DOCKER_README.md](DOCKER_README.md)** for detailed Docker instructions.

### Local Setup (3 Steps)

**Windows (Command Prompt):**
```cmd
setup.bat
run.bat
```

**Windows (Git Bash):**
```bash
chmod +x setup.sh scripts/run_gitbash.sh
./setup.sh
./scripts/run_gitbash.sh
```

**Linux/Mac:**
```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

The app will automatically open in your browser at `http://localhost:8501`

### Git Bash Users

See **[docs/GIT_BASH_GUIDE.md](docs/GIT_BASH_GUIDE.md)** for detailed Git Bash instructions.

### Detailed Guide

For detailed setup instructions, troubleshooting, and advanced usage, see **[docs/QUICKSTART.md](docs/QUICKSTART.md)**

## Usage

### Local Usage

1. **Start the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload a chest X-ray image**:
   - Supported formats: PNG, JPG, JPEG
   - The app will automatically analyze the image

3. **View Results**:
   - **At-a-Glance**: X-ray with detected lesions, classification, and severity
   - **Deep Analysis**: Structural localization and Grad-CAM heatmap
   - **Clinical Report**: Automated diagnostic summary with PDF download

### Web Deployment

Want to deploy this to the web? See **[docs/WEB_DEPLOYMENT.md](docs/WEB_DEPLOYMENT.md)** for comprehensive deployment guides covering:
- Streamlit Cloud (easiest, free tier)
- Docker deployment
- Heroku
- VPS/EC2 deployment
- AWS with auto-scaling

## Configuration

Edit `config.py` to customize:
- Model weights path
- Image processing parameters
- UI colors and styling
- Default thresholds

## Model Architecture

The application uses a Multi-Task Deep Learning model:

- **Backbone**: ConvNeXt-Tiny (timm)
- **Detection**: Faster R-CNN with Feature Pyramid Network (FPN)
- **Classification Head**: 2-layer MLP (Hidden: 256) → 3 classes (none, latent_tb, active_tb)
- **Severity Head**: 3-layer MLP (Hidden: 256, 128) → scalar [0, 1]

## Design System

The UI follows Apple's design principles:
- **Colors**: Soft Gray (#F5F5F7), Pure White (#FFFFFF), San Francisco Blue (#007AFF)
- **Typography**: System sans-serif (Inter font)
- **Components**: 12px rounded corners (squircle style), soft shadows, smooth animations
- **Glassmorphism**: Translucent cards with backdrop blur

## Technical Details

- **Preprocessing**: 512×512 resize, ImageNet normalization
- **Inference**: Automatic CUDA/CPU detection
- **Visualization**: Color-coded bounding boxes (Green=Healthy, Blue=Latent, Red=Active)
- **Explainability**: Grad-CAM on ConvNeXt backbone final layer

## Requirements

- Python 3.8+
- PyTorch 2.2.2+
- Streamlit 1.28+
- CUDA-capable GPU (optional, for faster inference)

## License

This is a demonstration platform for educational and research purposes.

## Testing

The project includes a comprehensive test suite:

### Quick Validation

```bash
# Validate setup
python scripts/validate_setup.py
```

### Run Tests

```bash
# Install test dependencies (pytest is included in requirements.txt)
pip install -r requirements.txt

# Run all tests
pytest

# Run unit tests only (fast, no model required)
pytest tests/unit/ -v

# Run integration tests (requires model weights)
pytest tests/integration/ -v -m integration

# Run with coverage
pytest --cov=. --cov-report=html
```

See `tests/README.md` for detailed testing documentation.

## Notes

- This application is for demonstration purposes only
- All findings should be reviewed by qualified medical professionals
- The model should be validated on your specific dataset before clinical use

