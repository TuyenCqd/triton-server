import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# ==========================================
# 1. THIẾT LẬP THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
BATCH_SIZE = 32
EPOCHS = 100  
INITIAL_LR = 1e-4       # LR cho giai đoạn đầu (chỉ train classifier)
FINE_TUNE_LR = 1e-5     # LR siêu nhỏ cho giai đoạn Fine-tuning
NUM_CLASSES = 3         # incorrect_mask, with_mask, without_mask
DATA_DIR = './dataset'
CHECKPOINT_PATH = './model/checkpoint/mobilenet_v2_mask_3classes.pth' 

# Cấu hình chuyển giai đoạn & Early Stopping
UNFREEZE_EPOCH = 10     # Mở khóa các lớp cuối từ Epoch thứ 10
PATIENCE = 7             
best_val_loss = float('inf')
patience_counter = 0
is_fine_tuning = False  # Biến đánh dấu trạng thái fine-tuning

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Đang huấn luyện trên thiết bị: {device}")

# ==========================================
# 2. TIỀN XỬ LÝ & DATA AUGMENTATION
# ==========================================
class EnsureRGB(object):
    def __call__(self, img):
        return img.convert('RGB')

data_transforms = {
    'train': transforms.Compose([
        EnsureRGB(), # <--- Thêm vào đây để xử lý triệt để ảnh lỗi trước khi Resize
        transforms.Resize((112, 112)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        EnsureRGB(), # <--- Thêm vào đây cho tập validation
        transforms.Resize((112, 112)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train'])
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. KHỞI TẠO HOẶC LOAD MÔ HÌNH (SỬA LỖI SIZE MISMATCH)
# ==========================================
# Khởi tạo khung mô hình MobileNet V2 trống
model = mobilenet_v2(weights=None)

# Kiểm tra nếu bạn ĐÃ TRAIN TIẾP và có checkpoint 3 lớp lưu từ trước
if os.path.exists(CHECKPOINT_PATH):
    print(f"[*] Tìm thấy file checkpoint {CHECKPOINT_PATH} (Mô hình 3 lớp), tiến hành load để train tiếp...")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    
else:
    print("[*] Không tìm thấy checkpoint cũ, tiến hành nạp trọng số ImageNet gốc để khởi tạo...")
    
    # Quét xem trong thư mục pretrain của bạn đang có file gốc nào
    possible_pretrains = [
        './model/pretrain/mobilenet_v2-7ebf99e0.pth',
        './model/pretrain/mobilenet_v2-b0353104.pth'
    ]
    
    pretrain_file_found = None
    for path in possible_pretrains:
        if os.path.exists(path):
            pretrain_file_found = path
            break
            
    if pretrain_file_found is None:
        raise FileNotFoundError("Lỗi: Không tìm thấy bất kỳ file trọng số ImageNet gốc nào trong thư mục ./model/pretrain/")
        
    print(f"==> Đang nạp trọng số gốc 1000 lớp từ file: {pretrain_file_found}")
    
    # Bước 1: Khởi tạo cấu hình 1000 lớp mặc định của mô hình trống để khớp với file gốc
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 1000)
    )
    # Bước 2: Nạp trọng số gốc 1000 lớp vào
    model.load_state_dict(torch.load(pretrain_file_found, map_location=device))
    
    # Bước 3: Khóa toàn bộ các lớp Feature Extractor để chạy Warm-up
    for param in model.parameters():
        param.requires_grad = False
        
    # Bước 4: Thay thế lớp phân loại sang 3 lớp khẩu trang (lớp mới này sẽ tự động requires_grad=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )

# Đảm bảo phần classifier luôn được mở khóa
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LR)

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    current_epoch = epoch + 1
    
    # --- LOGIC CHUYỂN SANG GIAI ĐOẠN FINE-TUNING ---
    if current_epoch == UNFREEZE_EPOCH and not is_fine_tuning:
        print(f"\n[>>>] Epoch {current_epoch}: Kích hoạt Fine-tuning! Mở khóa các lớp Convolution cuối...")
        
        # Mở khóa 2 block cuối cùng của MobileNet V2 (features[17] và features[18])
        for param in model.features[17].parameters():
            param.requires_grad = True
        for param in model.features[18].parameters():
            param.requires_grad = True
            
        # Cập nhật lại Optimizer: Nhận các tham số mới mở khóa + Hạ thấp Learning Rate
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR)
        is_fine_tuning = True
        
        # Reset lại Early Stopping khi sang giai đoạn mới để mô hình có cơ hội tối ưu tiếp
        best_val_loss = float('inf')
        patience_counter = 0
    # --- Chế độ Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # Sử dụng tqdm để bọc train_loader lại
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Cập nhật hiển thị loss thời gian thực lên màn hình/file log
        train_bar.set_postfix(loss=loss.item())

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct / total

    # --- Chế độ Validation ---
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)

            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item() * val_inputs.size(0)

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    epoch_val_loss = val_running_loss / len(val_dataset)
    epoch_val_acc = val_correct / val_total

    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    mode_str = "Fine-Tuning" if is_fine_tuning else "Warm-Up"
    print(f"Epoch {current_epoch}/{EPOCHS} [{mode_str}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    # --- Logic Early Stopping & Model Checkpoint ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        
        # SỬA Ở ĐÂY: Đảm bảo thư mục cha luôn tồn tại trước khi ghi file
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        
        # Tiến hành lưu checkpoint
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"==> Khởi sắc! Đã lưu lại checkpoint tốt nhất vào file: {CHECKPOINT_PATH}")
    else:
        patience_counter += 1
        print(f"==> Không cải thiện. Kích hoạt Early Stopping: {patience_counter}/{PATIENCE}")


print("\nQuá trình huấn luyện hoàn tất!")

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

# ==========================================
# 5. ĐÁNH GIÁ MÔ HÌNH VÀ VẼ BIỂU ĐỒ
# ==========================================
print("Đang đánh giá mô hình tốt nhất trên tập Validation...")

all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = train_dataset.classes

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Vẽ biểu đồ Loss và Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
plt.axvline(x=UNFREEZE_EPOCH-1, color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Validation Loss', marker='o')
plt.axvline(x=UNFREEZE_EPOCH-1, color='r', linestyle='--', label='Fine-tuning Start')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("Đã lưu biểu đồ Loss/Accuracy vào file: training_curves.png")

# Vẽ Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Thực tế (True Label)')
plt.xlabel('Dự đoán (Predicted Label)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Đã lưu biểu đồ Ma trận nhầm lẫn vào file: confusion_matrix.png")
