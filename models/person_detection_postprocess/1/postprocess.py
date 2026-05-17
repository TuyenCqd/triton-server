import torch
import torch.nn as nn

class Postprocess(nn.Module):
    def __init__(self, threshold=0.15):
        super().__init__()
        self.threshold = threshold

    def forward(self, dets, labels, orig_shape):
        # dets: [B, 300, 4]
        # labels: [B, 300, 2]
        # orig_shape: [B, 2]

        # 1. Chạy Sigmoid cho Raw Logits trên toàn bộ Batch
        # Giữ nguyên cấu trúc chiều bằng cách không dùng index 0
        scores = torch.sigmoid(labels[:, :, 0]) # Shape: [B, 300]
        
        # 2. Tạo mặt nạ lọc Confidence
        mask = scores >= self.threshold # Shape: [B, 300]
        
        # LƯU Ý: Để chạy được batch động mượt mà trong TensorRT, 
        # nên tránh dùng mảng mặt nạ làm thay đổi shape trực tiếp nếu không cần thiết.
        # Dưới đây là cách tính toán giữ nguyên chiều Batch:

        # 4. Tách chiều ảnh, chuyển về shape [B, 1] để broadcast toán học
        orig_h = orig_shape[:, 0:1] # Shape: [B, 1]
        orig_w = orig_shape[:, 1:2] # Shape: [B, 1]

        # 5. Khôi phục tỷ lệ (Thực hiện trên toàn bộ [B, 300])
        cx = dets[:, :, 0] * orig_w
        cy = dets[:, :, 1] * orig_h
        w  = dets[:, :, 2] * orig_w
        h  = dets[:, :, 3] * orig_h

        # 6. Chuyển về dạng Left, Top
        left = cx - (w / 2.0)
        top  = cy - (h / 2.0)

        # 7. Clamping
        zero_tensor = torch.tensor(0.0, device=dets.device)
        left = torch.clamp(left, min=zero_tensor)
        left = torch.min(left, orig_w)
        
        top = torch.clamp(top, min=zero_tensor)
        top = torch.min(top, orig_h)
        
        w = torch.min(w, orig_w - left)
        h = torch.min(h, orig_h - top)

        # 8. Đóng gói đầu ra giữ nguyên cấu trúc Batch [B, 300, X]
        boxes = torch.stack([left, top, w, h], dim=-1) # Shape: [B, 300, 4]
        
        # Thay vì filter cứng bằng mask làm mất chiều batch, ta nhân mask vào score 
        # Các box không đạt threshold sẽ nhận score = 0.0
        final_scores = scores * mask.float() # Shape: [B, 300]
        classes = torch.zeros_like(final_scores, dtype=torch.int32) # Shape: [B, 300]

        return boxes, final_scores, classes

model = Postprocess()
model.eval()

# Dummy inputs giữ nguyên batch = 1 mẫu
dummy_dets = torch.randn(1, 300, 4)
dummy_labels = torch.randn(1, 300, 2)
dummy_orig = torch.tensor([[1080.0, 1920.0]])

torch.onnx.export(
    model, 
    (dummy_dets, dummy_labels, dummy_orig), 
    "postprocess.onnx",
    input_names=["dets", "labels", "orig_shape"],
    output_names=["FINAL_BOXES", "FINAL_SCORES", "FINAL_CLASSES"],
    # BẮT BUỘC: Khai báo chiều 0 (Batch) là dynamic cho cả INPUT và OUTPUT
    dynamic_axes={
        "dets": {0: "batch_size"},
        "labels": {0: "batch_size"},
        "orig_shape": {0: "batch_size"},
        "FINAL_BOXES": {0: "batch_size"},
        "FINAL_SCORES": {0: "batch_size"},
        "FINAL_CLASSES": {0: "batch_size"}
    },
    opset_version=14
)
print("Xuất thành công: postprocess.onnx")
