/usr/src/tensorrt/bin/trtexec --onnx=preprocess.onnx \
        --saveEngine=model.plan \
        --minShapes=raw_input:1x3x100x100 \
        --optShapes=raw_input:4x3x1080x1920 \
        --maxShapes=raw_input:8x3x2160x3840 \
        --fp16