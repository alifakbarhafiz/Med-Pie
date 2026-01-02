# Med-Pie Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with Python
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models/weights && \
    mkdir -p /app/.streamlit

# Create Streamlit config directory and file
RUN echo "\
[server]\n\
port = 8501\n\
address = '0.0.0.0'\n\
enableCORS = false\n\
enableXsrfProtection = true\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
" > /app/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

