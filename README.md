# Triton Person Detection Server

This is a complete setup for running NVIDIA Triton Inference Server with a person detection model using TensorRT optimization.

## Project Structure

```
triton_server/
├── Dockerfile                    # Docker image for Triton server
├── docker-compose.yml            # Docker Compose orchestration
├── models/
│   ├── person_detection/         # Main detection model (RT-DETR)
│   │   ├── 1/
│   │   │   ├── model.onnx
│   │   │   ├── model.plan        # Generated TensorRT model
│   │   │   └── convert.sh        # Conversion script
│   │   └── config.pbtxt          # Model configuration
│   ├── person_detection_ensemble/
│   │   └── config.pbtxt          # Ensemble config (preprocess → model → postprocess)
│   ├── person_detection_preprocess/
│   │   ├── 1/
│   │   │   ├── preprocess.py
│   │   │   └── model.plan
│   │   └── config.pbtxt
│   └── person_detection_postprocess/
│       ├── 1/
│       │   ├── postprocess.py
│       │   └── model.py
│       └── config.pbtxt
├── infer/
│   ├── infer_person.py           # CLI inference script
└── README.md                      # This file
```

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker & Docker Compose with NVIDIA Container Toolkit
- NVIDIA Triton server 25.03 image from nvcr.io

## Quick Start

### 1. Convert ONNX to TensorRT Plan

The ONNX model needs to be converted to TensorRT plan format for optimized inference:

```bash
bash convert_model.sh
```

This will:
- Build a temporary Docker image with TensorRT tools
- Convert `model_onnx` to `model.plan`
- Place the `.plan` file in `models/person_detection/1/`

**Alternatively**, convert manually:

```bash
cd models/person_detection/1/
bash convert.sh
```

### 2. Start Triton Server

Using Docker Compose (recommended):

```bash
docker-compose up -d
```

Or with Docker directly:

```bash
docker build -t triton-person-detection:latest .
docker run --gpus all --rm -v $(pwd)/models:/models -v $(pwd)/infer:/infer \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  triton-person-detection:latest
```

### 3. Verify Server

Check if Triton server is running:

```bash
curl http://localhost:8000/v2/health/ready
```

List available models:

```bash
curl http://localhost:8000/v2/models
```

## Usage

### CLI Inference

Run detection on a single image:

```bash
python infer/infer_person.py \
  --image /path/to/image.jpg \
  --host localhost:8001 \
  --model rtdetr_ensemble \
  --threshold 0.5 \
  --output result.jpg
```

Options:
- `--image`: Path to input image (required)
- `--host`: Triton server address (default: localhost:8001)
- `--model`: Model name to use (default: rtdetr_ensemble)
- `--threshold`: Confidence threshold (default: 0.5)
- `--output`: Output image path (default: result_person_detection.jpg)
- `--batch-size`: Batch size (default: 1)


## Model Configuration

### Input/Output Specs

**Ensemble Model** (`rtdetr_ensemble`):
- Input: `IMAGE_IN` - UINT8 tensor [3, H, W] - RGB image
- Outputs:
  - `BOXES`: FP32 tensor [-1, 4] - Bounding boxes [x1, y1, x2, y2]
  - `SCORES`: FP32 tensor [-1] - Confidence scores
  - `CLASSES`: INT32 tensor [-1] - Class IDs
  - `NUM_DETECTIONS`: INT32 tensor [1] - Number of valid detections

**Main Model** (`person_detection`):
- Input: `input` - FP32 tensor [3, 560, 560]
- Outputs:
  - `dets`: FP32 tensor [-1, -1] - Raw detections
  - `labels`: FP32 tensor [-1, 2] - Raw labels

### Dynamic Batching

Configured with:
- Preferred batch sizes: [4, 8, 16, 32]
- Max queue delay: 2000 microseconds
- Max batch size: 32

## Performance Tuning

### TensorRT Optimization

The model is optimized with:
- FP16 precision for faster inference
- Dynamic shapes:
  - Min: 1x3x560x560
  - Optimal: 4x3x560x560
  - Max: 32x3x560x560

### Memory Management

- Shared memory disabled for stability
- GPU memory allows full batch processing
- Adjust `shm_size` in docker-compose.yml if needed

## Troubleshooting

### Model not loading

Check logs:
```bash
docker logs triton_person_detection
```

Verify model structure:
```bash
ls -la models/person_detection/1/
```

Ensure `model.plan` exists. If not, run conversion script.

### Conversion fails

Ensure GPU is available:
```bash
nvidia-smi
```

Check CUDA version compatibility with base image.

### API connection issues

Verify Triton is running:
```bash
curl http://localhost:8000/v2/health/ready
```

Check environment variable:
```bash
docker logs triton_inference_api
# Should show "Connected to Triton server..."
```

### Slow inference

- Check GPU utilization: `nvidia-smi`
- Verify batch size configuration
- Monitor Triton metrics: `http://localhost:8002/metrics`

## Advanced Configuration

### Custom Preprocessing

Edit `models/person_detection_preprocess/1/preprocess.py` and regenerate plan file.

### Custom Postprocessing

Edit `models/person_detection_postprocess/1/postprocess.py` and restart server.

### Enable Shared Memory

In `docker-compose.yml`, modify:
```yaml
environment:
  - TRITON_ENABLE_METRICS=true
```

And in inference client:
```python
PersonDetectionClient(
    triton_host="localhost:8001",
    shared_memory=True,
    shared_cuda_memory=True
)
```

## Production Deployment

### Recommended Setup

1. Use Kubernetes for orchestration
2. Enable Triton metrics for monitoring
3. Use model versioning for A/B testing
4. Set up proper logging and alerting

### Security

- Run Triton in isolated network
- Use TLS for gRPC connections
- Implement authentication for API
- Validate input images before processing

## References

- [NVIDIA Triton Inference Server Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md)

## Support

For issues or questions:
1. Check Triton server logs
2. Verify GPU availability
3. Ensure model files exist in correct locations
4. Check inference client compatibility

