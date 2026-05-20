LD_PRELOAD=/home/tuyenmb/projects/cctv-face/triton-server/BatchedNmsExtendLandmark/build/libmy_plugin.so /usr/src/tensorrt/bin/trtexec \
    --onnx=scrfd-post-320-320.onnx.nms.onnx \
    --saveEngine=model.plan \
    --minShapes=input.1:1x3x320x320 \
    --optShapes=input.1:16x3x320x320 \
    --maxShapes=input.1:32x3x320x320 \
    --fp16
