import numpy as np
import cv2
from common.triton_base import TritonBaseClient

class FaceEnsembleClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            # ("INPUT_IMAGE", "FP32")
            ("INPUT_IMAGE", "UINT8")
        ]
        
        # self.meta_outputs = [
        #     ("tmp_detections", "INT32", [-1]),
        #     ("final_boxes", "FP32", [-1, 4]),
        #     ("final_scores", "FP32", [ -1, 1]),
        #     ("tmp_classes", "FP32", [-1, 1]),
        #     ("final_landmarks", "FP32", [-1, 5, 2]),
        # ]
        
        self.meta_outputs = [
            ("tmp_detections", "INT32", [-1, 1]),
            ("tmp_boxes", "FP32", [-1, 1, 4]),
            ("tmp_scores", "FP32", [ -1, 1]),
            ("tmp_classes", "FP32", [-1, 1]),
            ("tmp_landmarks", "FP32", [-1, 1, 10]),
            ("ratio_value", "FP32", [-1, 1]),
        ]
        self.TARGET_SIZE = (320, 320) 

    def preprocess(self, frame):
        # Chọn một kích thước tạm thời để đồng bộ shape cho toàn bộ Batch
        # Kích thước này không ảnh hưởng đến tỷ lệ ảnh vì DALI sẽ tính toán lại dựa trên ảnh này
        TEMP_SIZE = (320, 320) 
        
        if isinstance(frame, list) or (isinstance(frame, np.ndarray) and frame.ndim == 1):
            resized_imgs = []
            for img in frame:
                # Ép về cùng kích thước hình học để đóng gói được vào mảng NumPy 4D
                img_resized = cv2.resize(img, TEMP_SIZE)
                resized_imgs.append(img_resized)
            
            # Tạo mảng NumPy dạng [Batch_Size, H, W, C] kiểu UINT8
            img_input = np.array(resized_imgs, dtype=np.uint8)
        else:
            # Nếu chỉ là 1 ảnh đơn lẻ
            img_resized = cv2.resize(frame, TEMP_SIZE)
            img_input = np.array(img_resized, dtype=np.uint8)
            img_input = np.expand_dims(img_input, axis=0) # Thành [1, H, W, C]
            
        return img_input



    # def preprocess(self, frame):
    #     # img_input = np.array(frame, dtype=np.float32)
    #     img_input = np.array(frame, dtype=np.uint8)
    #     # Thêm chiều batch để thành [1, H, W, 3]
    #     # img_input = np.expand_dims(img_input, axis=0)
    #     return img_input

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
    