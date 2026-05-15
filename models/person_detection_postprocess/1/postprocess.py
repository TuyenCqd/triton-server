import torch
import torch.nn as nn

class Postprocess(nn.Module):
    def __init__(self, threshold=0.15):
        super().__init__()
        self.threshold = threshold

    def forward(self, dets, labels, orig_shape):
        # dets: [1, 300, 4]
        # labels: [1, 300, 2]
        # orig_shape: [1, 2] (Nhận từ Preprocess)

        # 1. Chạy Sigmoid cho Raw Logits
        scores = torch.sigmoid(labels[0, :, 0]) # Lấy batch 0, class index 0
        
        # 2. Tạo mặt nạ (mask) lọc Confidence
        mask = scores >= self.threshold
        
        # 3. Lọc lấy những giá trị hợp lệ
        valid_scores = scores[mask]       # Shape: [N]
        valid_dets = dets[0, mask, :]     # Shape: [N, 4]
        
        # 4. Lấy chiều dài/rộng gốc
        orig_h = orig_shape[0, 0]
        orig_w = orig_shape[0, 1]

        # 5. Khôi phục tỷ lệ cx, cy, w, h
        cx = valid_dets[:, 0] * orig_w
        cy = valid_dets[:, 1] * orig_h
        w  = valid_dets[:, 2] * orig_w
        h  = valid_dets[:, 3] * orig_h

        # 6. Chuyển về dạng Left, Top
        left = cx - (w / 2.0)
        top  = cy - (h / 2.0)

        # 7. Clamping (Giới hạn ranh giới không vượt qua mép ảnh)
        zero_tensor = torch.tensor(0.0, device=dets.device)
        left = torch.clamp(left, min=zero_tensor, max=orig_w)
        top  = torch.clamp(top, min=zero_tensor, max=orig_h)
        w    = torch.min(w, orig_w - left)
        h    = torch.min(h, orig_h - top)

        # 8. Đóng gói đầu ra
        boxes = torch.stack([left, top, w, h], dim=-1) # Shape: [N, 4]
        classes = torch.zeros_like(valid_scores, dtype=torch.int32) # Shape: [N]

        return boxes, valid_scores, classes

model = Postprocess()
model.eval()

# Dummy inputs
dummy_dets = torch.randn(1, 300, 4)
dummy_labels = torch.randn(1, 300, 2)
dummy_orig = torch.tensor([[1080.0, 1920.0]])

torch.onnx.export(
    model, 
    (dummy_dets, dummy_labels, dummy_orig), 
    "postprocess.onnx",
    input_names=["dets", "labels", "orig_shape"],
    output_names=["FINAL_BOXES", "FINAL_SCORES", "FINAL_CLASSES"],
    dynamic_axes={
        "FINAL_BOXES": {0: "num_boxes"},
        "FINAL_SCORES": {0: "num_boxes"},
        "FINAL_CLASSES": {0: "num_boxes"}
    },
    opset_version=14
)
print("Xuất thành công: postprocess.onnx")