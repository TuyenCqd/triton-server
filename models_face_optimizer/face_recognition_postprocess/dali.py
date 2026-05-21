import os
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dmath 

def create_postprocess_pipeline(batch_size=32, num_threads=1, device_id=0):
    pipe = dali.Pipeline(
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=device_id
    )
    
    with pipe:
        embeddings = fn.external_source(
            name="embeddings",
            device="cpu",
            dtype=types.FLOAT,
            layout="C" 
        )
        
        embeddings_gpu = embeddings.gpu()
        
        emb_squared = embeddings_gpu * embeddings_gpu
        
        norm_squared = fn.reductions.sum(emb_squared, axes=[0], keep_dims=True)
        
        norm = dmath.sqrt(norm_squared)
        norm_safe = norm + 1e-8
        
        output = embeddings_gpu / norm_safe
        
        pipe.set_outputs(output)
        
    return pipe

if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_THREADS = 1
    DEVICE_ID = 0

    print("Đang khởi tạo và build DALI Postprocess pipeline...")
    try:
        pipeline = create_postprocess_pipeline(batch_size=BATCH_SIZE, num_threads=NUM_THREADS, device_id=DEVICE_ID)
        pipeline.build()
        
        output_dir = "1"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "model.dali")
        
        pipeline.serialize(filename=output_path)
        print(f"\n Đã tạo thành công file nhị phân: {output_path}")
    except Exception as e:
        print(f"\n Lỗi trong quá trình build pipeline: {e}")
