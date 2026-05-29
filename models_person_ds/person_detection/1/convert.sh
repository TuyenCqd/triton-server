/usr/src/tensorrt/bin/trtexec \
    --onnx=inference_model.onnx \
    --saveEngine=model.plan \
    --minShapes=input:1x3x560x560 \
    --optShapes=input:16x3x560x560 \
    --maxShapes=input:32x3x560x560 \
    --fp16 \
    --stronglyTyped
