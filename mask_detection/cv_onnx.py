import torch
import torch.nn as nn
from torchvision import models

# 1. Khai báo lại cấu trúc mô hình
NUM_CLASSES = 3
# Khởi tạo MobileNetV2 (không tải lại trọng số pre-trained từ đầu)
model = models.mobilenet_v2(pretrained=False)
# Gắn Custom Head 3 classes như lúc train
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# 2. Tải trọng số (weights) đã huấn luyện thành công
# File này được lưu ở bước trước
WEIGHTS_PATH = 'face_mask_mobilenetv2.pth'
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval() # Chuyển sang chế độ suy luận (Inference mode)

# 3. Tạo Dummy Input (Batch_Size=1, Channels=3, Width=224, Height=224)
# LƯU Ý: Nếu bạn train với input 112x112, hãy đổi số 224 thành 112
dummy_input = torch.randn(1, 3, 224, 224)

# 4. Thực hiện xuất ra file ONNX
ONNX_PATH = 'face_mask_mobilenetv2.onnx'
torch.onnx.export(
    model,                      # Mô hình cần xuất
    dummy_input,                # Input mẫu
    ONNX_PATH,                  # Tên file đầu ra
    export_params=True,         # Lưu trữ trọng số bên trong file
    opset_version=11,           # Chuẩn ONNX opset (11 thường rất ổn định)
    do_constant_folding=True,   # Tối ưu hóa mô hình
    input_names=['input'],      # Đặt tên cho input layer
    output_names=['output'],    # Đặt tên cho output layer
    # Cấu hình để model có thể nhận batch_size linh hoạt (1 ảnh hoặc nhiều ảnh cùng lúc)
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Thành công! Mô hình ONNX đã được lưu tại: {ONNX_PATH}")