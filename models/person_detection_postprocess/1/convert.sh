trtexec --onnx=postprocess.onnx \
        --saveEngine=model.plan \
        --minShapes=dets:1x300x4,labels:1x300x2,orig_shape:1x2 \
        --optShapes=dets:16x300x4,labels:16x300x2,orig_shape:16x2 \
        --maxShapes=dets:32x300x4,labels:32x300x2,orig_shape:32x2 \
        --fp16