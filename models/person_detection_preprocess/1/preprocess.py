import torch
import torch.nn as nn
import torch.nn.functional as F

class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        # ImageNet Mean & Std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # x shape: [Batch, 3, H, W], dtype: uint8 (từ Client gửi lên)
        batch_size = x.shape[0]
        h = x.shape[2]
        w = x.shape[3]
        
        # 1. Lưu lại kích thước gốc và nhân bản (repeat) cho đồng bộ với kích thước Batch
        # Tạo tensor [1, 2] chứa [h, w], sau đó repeat thành [Batch, 2]
        orig_shape = torch.tensor([[h, w]], dtype=torch.float32, device=x.device).repeat(batch_size, 1)

        # 2. Resize ảnh về 560x560
        x = F.interpolate(x.float(), size=(560, 560), mode='bilinear', align_corners=False)

        # 3. Chuẩn hóa (0-1 và trừ Mean/Std)
        x = x / 255.0
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x, orig_shape

model = Preprocess()
model.eval()

# Dummy input với kích thước giả định (Lúc export chỉ cần batch=1 làm mẫu)
dummy_input = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8)

torch.onnx.export(
    model, 
    dummy_input, 
    "preprocess.onnx",
    input_names=["raw_input"],
    output_names=["input", "orig_shape"],
    # BẮT BUỘC: Khai báo chiều 0 (Batch) là dynamic cho cả INPUT và OUTPUT
    dynamic_axes={
        "raw_input": {0: "batch_size", 2: "height", 3: "width"}, 
        "input": {0: "batch_size"},
        "orig_shape": {0: "batch_size"}
    },
    opset_version=17
)
print("Xuất thành công: preprocess.onnx (Đã hỗ trợ Dynamic Batching)")