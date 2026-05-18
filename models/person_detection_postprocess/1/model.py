import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        """Khởi tạo nhẹ nhàng, không cần device GPU hay NMS threshold"""
        pass

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # 1. Trích xuất Tensor thô từ RT-DETR
            in_dets = pb_utils.get_input_tensor_by_name(request, "dets")
            in_labels = pb_utils.get_input_tensor_by_name(request, "labels")
            in_orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape")

            # Chuyển sang mảng Numpy thuần túy
            dets = in_dets.as_numpy()
            labels = in_labels.as_numpy()
            orig_shape = in_orig_shape.as_numpy()

            batch_size = dets.shape[0]

            # 2. Tính Sigmoid cho Scores để ép về khoảng [0.0 -> 1.0]
            scores = 1.0 / (1.0 + np.exp(-labels[:, :, 0]))
            classes = np.zeros_like(scores, dtype=np.int32) # Class luôn là 0 (Person)

            # 3. Khôi phục tọa độ về kích thước ảnh gốc
            cx = dets[:, :, 0]
            cy = dets[:, :, 1]
            w_norm = dets[:, :, 2]
            h_norm = dets[:, :, 3]

            orig_h = orig_shape[:, 0].reshape(batch_size, 1)
            orig_w = orig_shape[:, 1].reshape(batch_size, 1)

            w = w_norm * orig_w
            h = h_norm * orig_h
            
            # Đổi sang hệ tọa độ [x_min, y_min, width, height] cho hàm NMS của OpenCV
            x = (cx * orig_w) - (w / 2.0)
            y = (cy * orig_h) - (h / 2.0)

            # Giới hạn ranh giới (Clip)
            x = np.clip(x, 0, orig_w)
            y = np.clip(y, 0, orig_h)

            # Gộp lại thành ma trận chuẩn [batch_size, 300, 4]
            boxes = np.stack([x, y, w, h], axis=-1).astype(np.float32)

            # 4. Trả TOÀN BỘ 300 kết quả về Client (Không có Padding, Không NMS)
            out_tensor_boxes = pb_utils.Tensor("FINAL_BOXES", boxes)
            out_tensor_scores = pb_utils.Tensor("FINAL_SCORES", scores.astype(np.float32))
            out_tensor_classes = pb_utils.Tensor("FINAL_CLASSES", classes)

            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_boxes, out_tensor_scores, out_tensor_classes]
            )
            responses.append(response)

        return responses

    def finalize(self):
        pass