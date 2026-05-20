LD_PRELOAD=/app/BatchedNmsExtendLandmark/build/libmy_plugin.so /usr/src/tensorrt/bin/trtexec \
    --onnx=arcface.onnx \
    --saveEngine=model.plan \
    --minShapes=data:1x3x112x112 \
    --optShapes=data:16x3x112x112 \
    --maxShapes=data:32x3x112x112
