FROM nvcr.io/nvidia/tritonserver:25.03-py3

# Install build dependencies for CUDA 12.8 on Ubuntu 24.04
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    build-essential \
    libcudnn8 \
    libcudnn8-dev \
    wget \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment for CUDA 12.8
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/tritonserver/lib:$LD_LIBRARY_PATH

# Copy source code
WORKDIR /app
COPY models_face_optimizer/face_alignment /app/face_alignment

# Build backend
WORKDIR /app/face_alignment
RUN mkdir build && \
    cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
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
