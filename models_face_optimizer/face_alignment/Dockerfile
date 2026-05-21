FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    libcudnn8 \
    libcudnn8-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Triton Server
RUN wget https://github.com/triton-inference-server/server/releases/download/v2.30.0/tritonserver2.30.0.linux-gpu.tar.gz && \
    tar -xzf tritonserver2.30.0.linux-gpu.tar.gz -C /opt && \
    rm tritonserver2.30.0.linux-gpu.tar.gz

# Set environment
ENV PATH /opt/tritonserver/bin:$PATH
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /app
COPY models_face_optimizer/face_alignment /app/face_alignment

# Build backend
WORKDIR /app/face_alignment
RUN mkdir build && \
    cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
    -DTRITON_INCLUDE_PATHS=/opt/tritonserver/include && \
    make -j$(nproc)

# Copy built library to models directory
RUN mkdir -p /models/face_alignment/1 && \
    cp build/lib/libface_alignment_backend.so /models/face_alignment/1/ && \
    cp config.pbtxt /models/face_alignment/

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models", "--strict-model-config=false"]
