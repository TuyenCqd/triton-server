import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os

@dali.pipeline_def(batch_size=16, num_threads=4, device_id=0)
def face_detection_preprocess_pipeline():
    # 1. Nhận ảnh từ Client
    images = fn.external_source(device="cpu", name="raw_image", dtype=types.UINT8)
    
    # 2. Lấy kích thước ảnh gốc [H, W, C]
    shapes = fn.shapes(images) 
    shapes_f = fn.cast(shapes, dtype=types.FLOAT)
    
    # Cắt lấy mảng chứa cả H và W (bỏ C ở vị trí index 2)
    # axes=[0] ở đây nghĩa là cắt theo trục của tensor shape [H, W, C] từ index 0 đến 2
    hw_dimensions = fn.slice(shapes_f, 0, 2, axes=[0])
    
    # 3. Tính ratio = 320 / max(h, w) cho TỪNG ảnh độc lập trong batch
    # axes=[0] bên trong reductions sẽ tìm max giữa H và W của từng ảnh đơn lẻ
    max_dim = fn.reductions.max(hw_dimensions, axes=[0]) 
    ratio = 320.0 / max_dim
    
    # 4. Đưa lên GPU để xử lý ảnh
    images_gpu = images.gpu()
    
    # 5. Resize giữ nguyên tỷ lệ (Letterbox)
    resized = fn.resize(images_gpu, 
                        resize_longer=320.0, 
                        interp_type=types.INTERP_LINEAR)
    
    # 6. Pad thêm viền màu 114 (đưa về đúng 320x320)
    padded = fn.pad(resized, fill_value=114.0, shape=(320, 320, 3))
    
    # 7. Normalize và Transpose sang NCHW
    preprocessed = fn.crop_mirror_normalize(padded,
                                            dtype=types.FLOAT,
                                            mean=[0.0, 0.0, 0.0],
                                            std=[255.0, 255.0, 255.0],
                                            output_layout="CHW")
    
    return preprocessed, ratio

# Thực hiện serialize lại
save_path = "model.dali"
pipe = face_detection_preprocess_pipeline()
pipe.serialize(filename=save_path)
print(f"Tạo file thành công tại: {save_path}")
