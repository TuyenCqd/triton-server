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
        # x shape: [1, 3, H, W], dtype: uint8 (từ Client gửi lên)
        
        # 1. Lưu lại kích thước gốc để đưa cho Postprocess
        orig_h = torch.tensor([x.shape[2]], dtype=torch.float32)
        orig_w = torch.tensor([x.shape[3]], dtype=torch.float32)
        orig_shape = torch.stack([orig_h, orig_w], dim=-1) # Shape: [1, 2]

        # 2. Resize ảnh về 560x560
        x = F.interpolate(x.float(), size=(560, 560), mode='bilinear', align_corners=False)

        # 3. Chuẩn hóa (0-1 và trừ Mean/Std)
        x = x / 255.0
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x, orig_shape

model = Preprocess()
model.eval()

# Dummy input với kích thước giả định (Client có thể gửi size khác)
dummy_input = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8)

torch.onnx.export(
    model, 
    dummy_input, 
    "preprocess.onnx",
    input_names=["raw_input"],
    output_names=["input", "orig_shape"],
    dynamic_axes={
        "raw_input": {2: "height", 3: "width"} # Cho phép H và W đầu vào thay đổi thoải mái
    },
    opset_version=14
)
print("Xuất thành công: preprocess.onnx")