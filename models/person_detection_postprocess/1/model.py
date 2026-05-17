import triton_python_backend_utils as pb_utils
import torch
import torchvision

class TritonPythonModel:
    def initialize(self, args):
        """Khởi tạo khi Triton nạp mô hình."""
        self.conf_threshold = 0.15  # Ngưỡng tự tin
        self.nms_threshold = 0.45  # Ngưỡng đè nhau của NMS
        
        # Bật thiết bị GPU mặc định
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        """Hàm này được gọi mỗi khi có request gửi tới."""
        responses = []
        
        for request in requests:
            # 1. Trích xuất Tensor từ Triton và chuyển thẳng sang PyTorch GPU (ZERO-COPY)
            in_dets = pb_utils.get_input_tensor_by_name(request, "dets")
            in_labels = pb_utils.get_input_tensor_by_name(request, "labels")
            in_orig_shape = pb_utils.get_input_tensor_by_name(request, "orig_shape")

            # to_dlpack() giúp PyTorch đọc thẳng bộ nhớ GPU hiện tại của Triton
            dets = torch.from_dlpack(in_dets.to_dlpack()).to(self.device)
            labels = torch.from_dlpack(in_labels.to_dlpack()).to(self.device)
            orig_shape = torch.from_dlpack(in_orig_shape.to_dlpack()).to(self.device)

            batch_size = dets.shape[0]
            
            # Khởi tạo mảng lưu trữ kết quả của từng ảnh trong batch
            out_boxes = []
            out_scores = []
            out_classes = []
            out_num_dets = []

            # 2. Xử lý từng ảnh trong Batch
            for b in range(batch_size):
                # Tính Sigmoid và tạo mặt nạ lọc Threshold
                scores = torch.sigmoid(labels[b, :, 0])
                mask = scores >= self.conf_threshold

                valid_scores = scores[mask]
                valid_dets = dets[b, mask, :]
                
                orig_h, orig_w = orig_shape[b, 0], orig_shape[b, 1]

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
                    x1 = torch.clamp(x1, min=0)
                    y1 = torch.clamp(y1, min=0)
                    x2 = torch.clamp(x2, max=orig_w)
                    y2 = torch.clamp(y2, max=orig_h)

                    boxes = torch.stack([x1, y1, x2, y2], dim=1)

                    # ÁP DỤNG NMS TRÊN GPU
                    keep_idx = torchvision.ops.nms(boxes, valid_scores, self.nms_threshold)
                    
                    boxes = boxes[keep_idx]
                    valid_scores = valid_scores[keep_idx]
                    classes = torch.zeros_like(valid_scores, dtype=torch.int32)
                else:
                    # Nếu không có đối tượng nào vượt qua threshold
                    boxes = torch.empty((0, 4), dtype=torch.float32, device=self.device)
                    valid_scores = torch.empty((0,), dtype=torch.float32, device=self.device)
                    classes = torch.empty((0,), dtype=torch.int32, device=self.device)

                out_boxes.append(boxes)
                out_scores.append(valid_scores)
                out_classes.append(classes)
                out_num_dets.append(len(boxes))

            # 3. Đệm (Padding) mảng về cùng kích thước cho cả Batch
            # Lấy số lượng hộp nhiều nhất trong batch này (Tối thiểu là 1 để tránh lỗi tensor rỗng)
            max_dets = max(out_num_dets) if max(out_num_dets) > 0 else 1
            
            batch_boxes = torch.zeros((batch_size, max_dets, 4), dtype=torch.float32, device=self.device)
            batch_scores = torch.zeros((batch_size, max_dets), dtype=torch.float32, device=self.device)
            batch_classes = torch.zeros((batch_size, max_dets), dtype=torch.int32, device=self.device)
            batch_num_dets = torch.tensor(out_num_dets, dtype=torch.int32, device=self.device).view(batch_size, 1)

            # Gán dữ liệu hợp lệ vào mảng đã đệm số 0
            for b in range(batch_size):
                n = out_num_dets[b]
                if n > 0:
                    batch_boxes[b, :n] = out_boxes[b]
                    batch_scores[b, :n] = out_scores[b]
                    batch_classes[b, :n] = out_classes[b]

            # 4. Trả kết quả về Triton (Zero-copy GPU -> Triton)
            out_tensor_boxes = pb_utils.Tensor.from_dlpack("FINAL_BOXES", torch.to_dlpack(batch_boxes))
            out_tensor_scores = pb_utils.Tensor.from_dlpack("FINAL_SCORES", torch.to_dlpack(batch_scores))
            out_tensor_classes = pb_utils.Tensor.from_dlpack("FINAL_CLASSES", torch.to_dlpack(batch_classes))
            out_tensor_num = pb_utils.Tensor.from_dlpack("NUM_DETECTIONS", torch.to_dlpack(batch_num_dets))

            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_boxes, out_tensor_scores, out_tensor_classes, out_tensor_num]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Dọn dẹp bộ nhớ khi model bị unload."""
        pass