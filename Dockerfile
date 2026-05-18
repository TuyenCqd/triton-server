FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \ 
    ca-certificates \ 
    cmake \ 
    build-essential \ 
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*

COPY . .
# RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir cupy-cuda12x

RUN pip install --no-cache-dir -r requirements.txt
