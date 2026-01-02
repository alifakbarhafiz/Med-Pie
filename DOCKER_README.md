# Med-Pie Docker Deployment Guide

This guide explains how to build and run Med-Pie using Docker.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, but recommended)
- Model weights file: `models/weights/tb_detection_model.pth`

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The application will be available at: `http://localhost:8501`

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t med-pie:latest .

# Run the container
docker run -d \
  --name med-pie-app \
  -p 8501:8501 \
  med-pie:latest

# View logs
docker logs -f med-pie-app

# Stop the container
docker stop med-pie-app
docker rm med-pie-app
```

## Building the Image

### Standard Build

```bash
docker build -t med-pie:latest .
```

### Build with Custom Tag

```bash
docker build -t med-pie:v1.0.0 .
```

### Build with No Cache (for clean rebuild)

```bash
docker build --no-cache -t med-pie:latest .
```

## Running the Container

### Basic Run

```bash
docker run -d -p 8501:8501 --name med-pie-app med-pie:latest
```

### Run with Custom Port

```bash
docker run -d -p 8080:8501 --name med-pie-app med-pie:latest
```

### Run with Volume Mount (for external model weights)

If your model weights are stored outside the container:

```bash
docker run -d \
  -p 8501:8501 \
  -v /path/to/models/weights:/app/models/weights:ro \
  --name med-pie-app \
  med-pie:latest
```

### Run with Environment Variables

```bash
docker run -d \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll \
  -e PYTHONUNBUFFERED=1 \
  --name med-pie-app \
  med-pie:latest
```

## Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f med-pie

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart services
docker-compose restart

# View running services
docker-compose ps
```

## Model Weights

### Option 1: Include in Docker Image (Default)

The model weights should be in `models/weights/tb_detection_model.pth` before building:

```bash
# Ensure model weights are in place
ls models/weights/tb_detection_model.pth

# Build image (includes model weights)
docker build -t med-pie:latest .
```

### Option 2: Mount from Host (For Large Files)

If model weights are too large or stored elsewhere:

1. **Update `.dockerignore`**: Uncomment the line excluding model weights:
   ```
   models/weights/*.pth
   ```

2. **Mount volume when running**:
   ```bash
   docker run -d \
     -p 8501:8501 \
     -v /path/to/model/weights:/app/models/weights:ro \
     --name med-pie-app \
     med-pie:latest
   ```

3. **Or use docker-compose.yml**: Uncomment the volume mount line

### Option 3: Download on First Run

Modify `config.py` to download model weights if not found (see WEB_DEPLOYMENT.md for details).

## Health Check

The container includes a health check that verifies the Streamlit server is running:

```bash
# Check container health
docker ps  # Shows health status

# Inspect health check
docker inspect --format='{{json .State.Health}}' med-pie-app | jq
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs med-pie-app

# Check if port is already in use
netstat -an | grep 8501  # Linux/Mac
netstat -an | findstr 8501  # Windows
```

### Model Not Found

```bash
# Verify model weights are in the container
docker exec med-pie-app ls -lh /app/models/weights/

# Check if model path is correct
docker exec med-pie-app python -c "from config import MODEL_WEIGHTS_PATH; print(MODEL_WEIGHTS_PATH)"
```

### Out of Memory

If the container runs out of memory:

1. **Increase Docker memory limit** (Docker Desktop → Settings → Resources)
2. **Reduce model size** or use CPU-only PyTorch
3. **Adjust resource limits in docker-compose.yml**

### Slow Performance

- **Use GPU**: If you have NVIDIA GPU, use `nvidia-docker` (see GPU section below)
- **Increase resources**: Adjust CPU/memory limits in docker-compose.yml
- **Optimize model**: Consider model quantization or smaller model variant

## GPU Support (NVIDIA)

If you have an NVIDIA GPU and want to use it:

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Run with GPU

```bash
docker run -d \
  --gpus all \
  -p 8501:8501 \
  --name med-pie-app \
  med-pie:latest
```

### Docker Compose with GPU

Update `docker-compose.yml`:

```yaml
services:
  med-pie:
    # ... existing configuration ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Production Deployment

### Build for Production

```bash
# Build with production tag
docker build -t med-pie:production .

# Tag for registry
docker tag med-pie:production your-registry/med-pie:v1.0.0
```

### Push to Registry

```bash
# Docker Hub
docker tag med-pie:latest yourusername/med-pie:latest
docker push yourusername/med-pie:latest

# AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag med-pie:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/med-pie:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/med-pie:latest

# Google Container Registry
docker tag med-pie:latest gcr.io/PROJECT-ID/med-pie:latest
docker push gcr.io/PROJECT-ID/med-pie:latest
```

### Run in Production

```bash
# Use docker-compose with production settings
docker-compose -f docker-compose.prod.yml up -d
```

## Multi-Stage Build Benefits

The Dockerfile uses multi-stage builds to:
- ✅ Reduce final image size
- ✅ Improve build caching
- ✅ Separate build dependencies from runtime
- ✅ Optimize layer caching

## Image Size Optimization

Current optimizations:
- Uses `python:3.9-slim` (smaller base image)
- Removes apt cache after installation
- Uses `--no-cache-dir` for pip
- Multi-stage build to exclude build tools

## Security Best Practices

1. **Run as non-root user** (optional, add to Dockerfile):
   ```dockerfile
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser
   ```

2. **Use specific image tags** instead of `latest` in production

3. **Scan images for vulnerabilities**:
   ```bash
   docker scan med-pie:latest
   ```

4. **Keep base images updated** regularly

## Environment Variables

Available environment variables:

- `STREAMLIT_SERVER_FILE_WATCHER_TYPE`: Set to `poll` (default in Docker)
- `PYTHONUNBUFFERED`: Set to `1` for real-time logs

## Networking

### Expose on Different Port

```bash
docker run -d -p 8080:8501 med-pie:latest
```

### Use Custom Network

```bash
docker network create med-pie-network
docker run -d --network med-pie-network -p 8501:8501 med-pie:latest
```

## Backup and Restore

### Backup Model Weights

```bash
docker cp med-pie-app:/app/models/weights/tb_detection_model.pth ./backup/
```

### Restore Model Weights

```bash
docker cp ./backup/tb_detection_model.pth med-pie-app:/app/models/weights/
docker restart med-pie-app
```

## Cleanup

```bash
# Remove container
docker rm med-pie-app

# Remove image
docker rmi med-pie:latest

# Remove all unused containers, networks, images
docker system prune -a

# Remove volumes
docker volume prune
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Deployment](https://docs.streamlit.io/deploy)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

---

**Need help?** Check the main [README.md](../README.md) or [docs/WEB_DEPLOYMENT.md](docs/WEB_DEPLOYMENT.md) for more deployment options.

