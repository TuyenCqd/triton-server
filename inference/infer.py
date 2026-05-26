import os
import sys
import logging
import cv2
import numpy as np
import glog as logger

from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
from client.face_alignment import FaceExtPreClient
from client.face_emmbedding import FaceRegClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchFacePipeline:
    def __init__(self, triton_host="localhost:9187", person_threshold=0.45, max_person_batch=16):
        self.triton_host = triton_host
        self.person_threshold = person_threshold
        self.max_person_batch = max_person_batch
        
        logger.info(f"Connecting Triton Server (Batch Mode) at: {self.triton_host}")
        
        self.person_det_ensemble = PersonDetectionClient(
            triton_host=self.triton_host, triton_model_name="person_detection_ensemble",
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_det_ensemble = FaceEnsembleClient(
            triton_host=self.triton_host, triton_model_name="pipeline_ensemble_Det", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_align_model = FaceExtPreClient(
            triton_host=self.triton_host, triton_model_name="face_alignment_op", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_recog_ensemble = FaceRegClient(
            triton_host=self.triton_host, triton_model_name="pipeline_reg", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        logger.info("Init Batch Client Triton successfully!")

    def _crop_person_images_batch(self, frames, results):
        batch_cropped_images = []
        frame_ids = []
        for idx, (frame, res) in enumerate(zip(frames, results)):

            det_data = res.get("detection", res) if isinstance(res, dict) else res
            actual_frame_idx = res.get("frame_idx", idx) if isinstance(res, dict) else idx

            cropped_images_single_frame = []
            h_img, w_img, _ = frame.shape
            # boxes = res.get("BOXES", [])
            boxes = det_data.get("BOXES", det_data.get("tmp_boxes", []))
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    cropped_images_single_frame.append(frame[y1:y2, x1:x2].copy())
            batch_cropped_images.append(cropped_images_single_frame)
            frame_ids.append(actual_frame_idx)
        return batch_cropped_images, frame_ids 

    def _crop_and_align_face(self, person_img, idx, frame_id):
        if person_img is None or person_img.size == 0:
            return None

        img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results = self.face_det_ensemble.predict([img_rgb])
        results = results[0]

        landmarks = results.get("final_landmarks", [])
        bboxes = results.get("final_boxes", [])

        if len(bboxes) == 0:
            return None

        try:
            bbox = bboxes[0] if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes
            x1, y1, x2, y2 = map(int, bbox[:4])
            w, h = x2 - x1, y2 - y1

            mx, my = int(w * .3), int(h * .3)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(img_rgb.shape[1], x2 + mx)
            y2 = min(img_rgb.shape[0], y2 + my)

            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            white_bg[y1:y2, x1:x2] = face_crop

            results_align = self.face_align_model.predict([white_bg], landmarks, bboxes)
            
            aligned_112 = np.clip(
                results_align[0]["face_aligned_112"] * 128.0 + 127.5, 0, 255
            ).astype(np.uint8)

            
           
            # out_face_path = os.path.join("./output_crops", f"_frame_{frame_id}_aligned_{idx}.jpg")
            # cv2.imwrite(out_face_path,  aligned_112.transpose(1, 2, 0))
            # print(f"Saved aligned face to: {out_face_path}")
            
            return aligned_112.transpose(1, 2, 0)

        except Exception as e:
            logger.error(f"Error when alignment face: {e}")
            return None

    def process_batch(self, frames_list):
        if not frames_list:
            return []

        results = self.person_det_ensemble.predict(frames_list, self.person_threshold, verbose=False)
        if not isinstance(results, list):
            results = [results]

        batch_person_crops, frame_id = self._crop_person_images_batch(frames_list, results)
        
        batch_output = []

        for img_idx, person_crops in enumerate(batch_person_crops):
            img_results = []

            current_frame_id = frame_id[img_idx]
            
            for idx, person_img in enumerate(person_crops):
                if person_img.size == 0:
                    continue
                    
                aligned_face = self._crop_and_align_face(person_img, idx, current_frame_id)
            
                if aligned_face is not None:
                    try:
                        recog_results = self.face_recog_ensemble.predict([aligned_face])
                        recog_results = recog_results[0]
                        embeddings = recog_results.get("norm_embeddings", None)
                        
                        if embeddings is not None:
                            img_results.append({
                                "person_idx": idx,
                                "person_crop": person_img,
                                "aligned_face": aligned_face,
                                "embeddings": embeddings
                            })
                    except Exception as e:
                        logger.error(f"Lỗi trích xuất embedding cho Ảnh {img_idx}, Người {idx}: {e}")
            
            batch_output.append(img_results)
                        
        return batch_output


if __name__ == "__main__":
    pipeline = BatchFacePipeline(triton_host="localhost:9187", person_threshold=0.45)

    img_paths = [
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/test.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000910.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05202026/14581214_165159_20052026_IMG_20260520_165031.jpg"
    ]
    
    batch_frames = []
    for p in img_paths:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                batch_frames.append(img)
    
    logger.info(f"Loading batch iclude: {len(batch_frames)} imgs input into Class.")        
    
    all_batch_results = pipeline.process_batch(batch_frames)
    # print("all_batch_results: ", all_batch_results)
    
    for img_idx, objects_in_image in enumerate(all_batch_results):
        print(f"\n================ KẾT QUẢ ẢNH THỨ {img_idx} ================")
        print(f"-> Detected {len(objects_in_image)} valid face.")
        
        for obj in objects_in_image:
            print(f"  [Person {obj['person_idx']}]")
            print(f"  - Shape of vector Embedding: {obj['embeddings'].shape}")
            print(f"  - L2 Norm: {np.linalg.norm(obj['embeddings']):.4f}")
