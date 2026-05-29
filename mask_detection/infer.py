import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import os

# ==========================================
# 1. CẤU HÌNH BAN ĐẦU
# ==========================================
NUM_CLASSES = 3
# Thứ tự nhãn phải khớp hoàn toàn với thứ tự thư mục trong tập Train
CLASS_NAMES = ['incorrect_mask', 'with_mask', 'without_mask']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = 'face_mask_mobilenetv2.pth'

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH VÀ TIỀN XỬ LÝ
# ==========================================
# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform bắt buộc phải giống hệt lúc train
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Đổi thành 112x112 nếu bạn dùng size nhỏ
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. HÀM DỰ ĐOÁN VÀ LƯU ẢNH
# ==========================================
def predict_and_save_image(image_path, output_dir="results"):
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Đọc ảnh gốc
    img = Image.open(image_path).convert('RGB')

    # Tiền xử lý để đưa vào model
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Thêm chiều Batch: (1, 3, 224, 224)

    # Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    label_name = CLASS_NAMES[predicted_idx.item()]
    conf_score = confidence.item() * 100

    # Xử lý đồ họa: Vẽ text lên ảnh
    draw = ImageDraw.Draw(img)

    # Thiết lập màu sắc: Xanh lá (Đúng), Đỏ (Không đeo), Cam (Sai cách)
    if label_name == 'with_mask':
        color = "green"
    elif label_name == 'without_mask':
        color = "red"
    else:
        color = "orange"

    # Tạo chuỗi text kết quả
    text_result = f"{label_name} ({conf_score:.1f}%)"

    # Vẽ hộp nền đen nhỏ phía sau text để dễ đọc
    draw.rectangle([(5, 5), (200, 25)], fill="black")
    # Viết chữ lên ảnh (dùng font mặc định của PIL)
    draw.text((10, 10), text_result, fill=color)

    # Lưu file
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"predicted_{filename}")
    img.save(save_path)
    print(f" Đã dự đoán [{text_result}] và lưu ảnh tại: {save_path}")

# ==========================================
# 4. CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Thay 'test_image.jpg' bằng đường dẫn tới một bức ảnh bất kỳ của bạn
    test_img_path = 'test_image.jpg'
    if os.path.exists(test_img_path):
        predict_and_save_image(test_img_path, output_dir="./output_results")
    else:
        print("Vui lòng cập nhật biến test_img_path bằng đường dẫn ảnh hợp lệ để thử nghiệm.")