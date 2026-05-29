import numpy as np
from common.triton_base import TritonBaseClient

class FaceRegClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            ("INPUT_IMAGE", "UINT8")
        ]
        
        self.meta_outputs = [
            ("norm_embeddings", "FP32", [-1, 512]),
        ]

    def preprocess(self, frame):
        # img_input = np.array(frame, dtype=np.float32)
        img_input = np.array(frame, dtype=np.uint8)
        # Thêm chiều batch để thành [1, H, W, 3]
        # img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def predict(self, frame, verbose= False):
        img_blob = self.preprocess(frame)
        
        # if verbose:
            # tik = time.time()

        batch_result = self.run(
            [img_blob], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose = verbose
        )

        return batch_result