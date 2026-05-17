trtexec --onnx=postprocess.onnx \
        --saveEngine=model.plan \
        --minShapes=dets:1x300x4,labels:1x300x2,orig_shape:1x2 \
        --optShapes=dets:4x300x4,labels:4x300x2,orig_shape:4x2 \
        --maxShapes=dets:8x300x4,labels:8x300x2,orig_shape:8x2 \
        --fp16