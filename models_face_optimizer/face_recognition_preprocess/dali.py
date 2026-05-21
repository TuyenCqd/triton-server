import os
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types

def create_dali_pipeline(batch_size=32, num_threads=4, device_id=0):
    pipe = dali.Pipeline(
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=device_id
    )
    
    with pipe:
        images = fn.external_source(
            name="raw_image", 
            device="cpu", 
            dtype=types.UINT8,
            layout="HWC"
        )
        
        images_gpu = images.gpu()
        
        images_rgb = fn.color_space_conversion(
            images_gpu, 
            image_type=types.BGR, 
            output_type=types.RGB
        )
        
        resized = fn.resize(
            images_rgb, 
            resize_x=112, 
            resize_y=112, 
            interp_type=types.INTERP_LANCZOS3
        )
        normalized = resized * (1.0 / 127.5) - 1.0
        
        output = fn.transpose(normalized, perm=[2, 0, 1])
        
        pipe.set_outputs(output)
        
    return pipe

if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_THREADS = 4
    DEVICE_ID = 0

    print("Đang khởi tạo và build DALI pipeline...")
    pipeline = create_dali_pipeline(batch_size=BATCH_SIZE, num_threads=NUM_THREADS, device_id=DEVICE_ID)
    pipeline.build()
    
    output_dir = "1"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.dali")
    
    pipeline.serialize(filename=output_path)
    print(f"Đã tạo thành công file: {output_path}")
