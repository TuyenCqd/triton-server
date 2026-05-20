import cv2
import json
import numpy as np
import triton_python_backend_utils as pb_utils

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

# def norm_crop(img, landmark, image_size=112):
#     # Tính ratio dựa trên chuẩn 112 của ArcFace
#     ratio = float(image_size) / 112.0
#     dst = ARCFACE_DST * ratio
#     M, _ = cv2.estimateAffinePartial2D(landmark, dst, method=cv2.LMEDS)
#     if M is not None:
#         return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
#     return cv2.resize(img, (image_size, image_size))

def norm_crop(
    img, landmark, bbox, image_size=112, margin=.3
):
    x1, y1, x2, y2 = bbox.astype(np.int32)
    w = x2 - x1
    h = y2 - y1

    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(img.shape[1], x2 + mx)
    y2 = min(img.shape[0], y2 + my)
    # crop ROI
    roi = img[y1:y2, x1:x2]

    # landmark local coordinate
    landmark_local = landmark.copy()
    landmark_local[:, 0] -= x1
    landmark_local[:, 1] -= y1

    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    M, _ = cv2.estimateAffinePartial2D(landmark_local, dst, method=cv2.LMEDS)

    if M is None:
        return None
    aligned = cv2.warpAffine(roi, M, (image_size, image_size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    return aligned

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        
        # Cấu trúc: { name: (size, mean, std, is_nchw) }
        self.output_specs = {
            "face_aligned_112": (112, 127.5, 128.0, True),
            "face_aligned_224": (224, 127.5, 127.5, True),
            "face_aligned_nhwc": (112, 0.0, 255.0, False) 
        }
        
        self.specs = {}
        for name, (size, mean, std, is_nchw) in self.output_specs.items():
            cfg = pb_utils.get_output_config_by_name(self.model_config, name)
            dtype = pb_utils.triton_string_to_numpy(cfg["data_type"])
            self.specs[name] = {
                "size": size, 
                "mean": mean, 
                "std": std, 
                "dtype": dtype, 
                "is_nchw": is_nchw
            }

    def execute(self, requests):
        responses = []
        for request in requests:
            img_in = pb_utils.get_input_tensor_by_name(request, "person_image").as_numpy()
            lmk_in = pb_utils.get_input_tensor_by_name(request, "landmarks").as_numpy().reshape(-1, 5, 2)
            bbox_in = pb_utils.get_input_tensor_by_name(request, "bboxes").as_numpy().reshape(-1, 4)
            
            if len(img_in.shape) == 4: img_in = np.squeeze(img_in, axis=0)

            aligned_results = {name: [] for name in self.output_specs}

            for lmk, bbox in zip(lmk_in, bbox_in):
                if lmk.shape == (5, 2):
                    for name, spec in self.specs.items():
                        face_img = norm_crop(img_in, lmk, bbox, image_size=spec["size"])
                        aligned_results[name].append(face_img)

            output_tensors = []
            for name, spec in self.specs.items():
                if aligned_results[name]:
                    if spec["is_nchw"]:
                        # NCHW dùng cho các model TensorRT thông thường
                        data = cv2.dnn.blobFromImages(
                            aligned_results[name], 1.0/spec["std"], (spec["size"],)*2, (spec["mean"],)*3, swapRB=True
                        ).astype(spec["dtype"])
                    else:
                        # print("aligned_results[name]: ", aligned_results[name])
                        # NHWC (-1, 112, 112, 3)
                        # Chuyển list thành array (N, H, W, C), RGB, và chuẩn hóa
                        data = np.stack(aligned_results[name], axis=0).astype(np.float32)
                        data = (data[:, :, :, ::-1] - spec["mean"]) / spec["std"] # BGR to RGB và norm
                        data = data.astype(spec["dtype"])
                else:
                    # Trả về tensor rỗng đúng định dạng nếu không có mặt
                    shape = (0, 3, spec["size"], spec["size"]) if spec["is_nchw"] else (0, spec["size"], spec["size"], 3)
                    data = np.empty(shape, dtype=spec["dtype"])
                
                output_tensors.append(pb_utils.Tensor(name, data))

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
            
        return responses
