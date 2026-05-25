import numpy as np
from common.triton_base import TritonBaseClient

class FaceExtPreClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            # ("person_image", "FP32"),
            ("person_image", "UINT8"),
            ("landmarks", "FP32"),
            ("bboxes", "FP32")
        ]
        
        self.meta_outputs = [
            ("face_aligned_112", "FP32", [-1, 3, 112, 112]),
            ("face_aligned_224", "FP32", [-1, 3, 224, 224]),
            ("face_aligned_nhwc", "FP32", [-1, 112, 112, 3]),
        ]

    def preprocess(self, frame, landmarks, bboxes):
        # img_input = np.array(frame, dtype=np.float32)
        img_input = np.array(frame, dtype=np.uint8)
        landmarks_input = np.expand_dims(landmarks, axis=0)
        bboxes_input = np.expand_dims(bboxes, axis=0)
        # Thêm chiều batch để thành [1, H, W, 3]
        # img_input = np.expand_dims(img_input, axis=0)
        return img_input, landmarks_input, bboxes_input

    def predict(self, frame, landmarks, bboxes, verbose= False):
        img_blob, landmark_input, bboxes_input = self.preprocess(frame, landmarks, bboxes)
        
        # if verbose:
            # tik = time.time()

        batch_result = self.run(
            [img_blob, landmark_input, bboxes_input], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose = verbose
        )

        return batch_result