import numpy as np
import json
import cv2
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.input_size = (112, 112)

    def transform_img(self, img):
        height, width = img.shape[:2]

        if height == 0 or width == 0:
            return None
        
        resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(
            resized_img, self.input_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized_img

    def execute(self, requests):
        responses = []
        for request in requests:
            # Lấy input (giả sử input tên là "raw_image", shape [H, W, C])
            in_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
            img = in_tensor.as_numpy()
            
            # Xử lý trường hợp ảnh bị mất chiều (ví dụ từ [H,W,C] thành [C,H,W] hoặc bị bóp batch)
            if len(img.shape) == 4: # Nếu bị dư chiều batch [1, H, W, 3]
                img = img[0]

            resized_img = self.transform_img(img)
            
            if resized_img is None:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonModelException("Invalid input image size (0x0)"))
                )
                continue

            # Logic Preprocess
            batch_data = np.ones((1, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
            h_r, w_r = resized_img.shape[:2]
            batch_data[0, :h_r, :w_r, :] = resized_img
            batch_data = ((batch_data / 255) - 0.5) / 0.5
            
            # Chuyển về NCHW
            batch_data = batch_data.transpose((0, 3, 1, 2))

            # Output tensor cho model tiếp theo
            out_tensor = pb_utils.Tensor("preprocessed_input", batch_data)

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[out_tensor]
            ))
        return responses
