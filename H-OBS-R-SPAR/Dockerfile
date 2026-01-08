FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace/H-OBS-R-SPAR

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project files
COPY . /workspace/H-OBS-R-SPAR

# Set environment variables
ENV PYTHONPATH=/workspace/H-OBS-R-SPAR:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Download pretrained models
RUN python scripts/download_pretrained.py || echo "Pretrained download script not ready yet"

# Set default command
CMD ["bash"]
