# Base image
FROM runpod/base:0.4.0-cuda11.8.0

# Install PyTorch, TorchVision, and Torchaudio with CUDA 11.8 support
RUN python3.11 -m pip install --ignore-installed --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --ignore-installed --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Create a symlink for python (if needed)
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Add source files
ADD src .

# Run the main script
CMD python3.11 -u /handler.py
