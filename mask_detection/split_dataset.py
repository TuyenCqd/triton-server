import os
import shutil
import random

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TỶ LỆ
# ==========================================
SOURCE_DIR = './dataset/FMD_DATASET'    # Thư mục gốc chứa dữ liệu hiện tại
DEST_DIR = 'dataset'          # Thư mục đích sau khi chia xong
TRAIN_RATIO = 0.8             # 80% ảnh dùng để Train, 20% dùng để Valid

# Định nghĩa cấu trúc mapping để gộp thư mục con vào 3 class chính
CLASS_MAPPING = {
    'incorrect_mask': ['mc', 'mmc'],
    'with_mask': ['complex', 'simple'],
    'without_mask': ['complex', 'simple']
}

# ==========================================
# 2. HÀM TẠO THƯ MỤC ĐÍCH
# ==========================================
def create_dest_folders():
    # Tạo thư mục train và val
    for split in ['train', 'val']:
        for main_class in CLASS_MAPPING.keys():
            folder_path = os.path.join(DEST_DIR, split, main_class)
            os.makedirs(folder_path, exist_ok=True)
    print(f" Đã tạo xong cấu trúc thư mục mới tại: {DEST_DIR}/")

# ==========================================
# 3. HÀM XỬ LÝ CHIA VÀ COPY DỮ LIỆU
# ==========================================
def process_and_split_data():
    for main_class, sub_folders in CLASS_MAPPING.items():
        all_images = []

        # 3.1. Gom toàn bộ đường dẫn ảnh từ các thư mục con
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(SOURCE_DIR, main_class, sub_folder)

            if not os.path.exists(sub_folder_path):
                print(f" Cảnh báo: Không tìm thấy {sub_folder_path}")
                continue

            # Lấy tất cả tên file trong thư mục con (bỏ qua các thư mục ẩn)
            images = [f for f in os.listdir(sub_folder_path) if os.path.isfile(os.path.join(sub_folder_path, f))]

            # Lưu lại đường dẫn đầy đủ của từng ảnh
            for img in images:
                all_images.append(os.path.join(sub_folder_path, img))

        # 3.2. Trộn ngẫu nhiên danh sách ảnh để đảm bảo tính khách quan
        random.shuffle(all_images)

        # 3.3. Tính toán số lượng cho Train và Valid
        total_images = len(all_images)
        train_count = int(total_images * TRAIN_RATIO)

        train_images = all_images[:train_count]
        val_images = all_images[train_count:]

        # 3.4. Copy ảnh sang thư mục mới
        print(f" Đang copy class '{main_class}': {len(train_images)} Train | {len(val_images)} Valid...")

        # Copy Train
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            # Thêm prefix tên thư mục con vào ảnh để tránh trùng lặp tên file giữa các thư mục con
            parent_folder = os.path.basename(os.path.dirname(img_path))
            new_img_name = f"{parent_folder}_{img_name}"
            shutil.copy2(img_path, os.path.join(DEST_DIR, 'train', main_class, new_img_name))

        # Copy Valid
        for img_path in val_images:
            img_name = os.path.basename(img_path)
            parent_folder = os.path.basename(os.path.dirname(img_path))
            new_img_name = f"{parent_folder}_{img_name}"
            shutil.copy2(img_path, os.path.join(DEST_DIR, 'val', main_class, new_img_name))

# ==========================================
# 4. THỰC THI SCRIPT
# ==========================================
if __name__ == '__main__':
    # Đặt seed để mỗi lần chạy lại đều cho ra kết quả random giống nhau
    random.seed(42)

    print("Bắt đầu xử lý dữ liệu...")
    create_dest_folders()
    process_and_split_data()
    print("\n Hoàn tất! Dữ liệu của bạn đã sẵn sàng trong thư mục 'dataset/'.")