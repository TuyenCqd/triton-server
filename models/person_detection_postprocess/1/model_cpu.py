import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        """Khởi tạo khi Triton nạp mô hình."""
        self.conf_threshold = 0.5  # Ngưỡng tự tin
        self.nms_threshold = 0.45  # Ngưỡng đè nhau của NMS
        
    def nms_numpy(self, boxes, scores, iou_threshold):
        """Hàm Non-Maximum Suppression thuần Numpy."""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def execute(self, requests):
        """Hàm này được gọi mỗi khi có request gửi tới."""
        responses = []
        
        for request in requests:
            # 1. Trích xuất Tensor từ Triton và chuyển về Numpy (Chạy trên CPU)
            in_dets = pb_utils.get_input_tensor_by_name(request, "dets").as_numpy()
            in_labels = pb_utils.get_input_tensor_by_name(request, "labels").as_numpy()
            in_orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape").as_numpy()

            batch_size = in_dets.shape[0]
            
            out_boxes = []
            out_scores = []
            out_classes = []
            out_num_dets = []

            # 2. Xử lý từng ảnh trong Batch
            for b in range(batch_size):
                # Tính Sigmoid (sử dụng clip để tránh lỗi tràn số np.exp)
                raw_logits = np.clip(in_labels[b, :, 0], -500, 500)
                scores = 1.0 / (1.0 + np.exp(-raw_logits))
                
                # Tạo mặt nạ lọc Threshold
                mask = scores >= self.conf_threshold

                valid_scores = scores[mask]
                valid_dets = in_dets[b, mask, :]
                
                orig_h, orig_w = in_orig_shape[b, 0], in_orig_shape[b, 1]

                if len(valid_scores) > 0:
                    # Khôi phục tọa độ về kích thước gốc
                    cx = valid_dets[:, 0] * orig_w
                    cy = valid_dets[:, 1] * orig_h
                    w  = valid_dets[:, 2] * orig_w
                    h  = valid_dets[:, 3] * orig_h

                    # Chuyển đổi sang hệ tọa độ [x1, y1, x2, y2]
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    # Giới hạn ranh giới (Clamping)
                    x1 = np.clip(x1, 0, orig_w)
                    y1 = np.clip(y1, 0, orig_h)
                    x2 = np.clip(x2, 0, orig_w)
                    y2 = np.clip(y2, 0, orig_h)

                    boxes = np.stack([x1, y1, x2, y2], axis=1)

                    # ÁP DỤNG NMS (Numpy)
                    keep_idx = self.nms_numpy(boxes, valid_scores, self.nms_threshold)
                    
                    boxes = boxes[keep_idx]
                    valid_scores = valid_scores[keep_idx]
                    classes = np.zeros_like(valid_scores, dtype=np.int32)
                else:
                    # Nếu không có đối tượng nào vượt qua threshold
                    boxes = np.empty((0, 4), dtype=np.float32)
                    valid_scores = np.empty((0,), dtype=np.float32)
                    classes = np.empty((0,), dtype=np.int32)

                out_boxes.append(boxes)
                out_scores.append(valid_scores)
                out_classes.append(classes)
                out_num_dets.append(len(boxes))

            # 3. Đệm (Padding) mảng về cùng kích thước cho cả Batch
            max_dets = max(out_num_dets) if max(out_num_dets) > 0 else 1
            
            batch_boxes = np.zeros((batch_size, max_dets, 4), dtype=np.float32)
            batch_scores = np.zeros((batch_size, max_dets), dtype=np.float32)
            batch_classes = np.zeros((batch_size, max_dets), dtype=np.int32)
            batch_num_dets = np.array(out_num_dets, dtype=np.int32).reshape(batch_size, 1)

            # Gán dữ liệu hợp lệ vào mảng đã đệm số 0
            for b in range(batch_size):
                n = out_num_dets[b]
                if n > 0:
                    batch_boxes[b, :n] = out_boxes[b]
                    batch_scores[b, :n] = out_scores[b]
                    batch_classes[b, :n] = out_classes[b]

            # 4. Trả kết quả về Triton (Numpy -> Triton Tensor)
            out_tensor_boxes = pb_utils.Tensor("FINAL_BOXES", batch_boxes)
            out_tensor_scores = pb_utils.Tensor("FINAL_SCORES", batch_scores)
            out_tensor_classes = pb_utils.Tensor("FINAL_CLASSES", batch_classes)
            out_tensor_num = pb_utils.Tensor("NUM_DETECTIONS", batch_num_dets)

            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_boxes, out_tensor_scores, out_tensor_classes, out_tensor_num]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Dọn dẹp bộ nhớ khi model bị unload."""
        pass