import os
import sys
import logging
import cv2
import numpy as np
import glog as logger
import requests 

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

    def _crop_and_align_face(self, person_img):
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
            
            return aligned_112.transpose(1, 2, 0)

        except Exception as e:
            logger.error(f"Error when alignment face: {e}")
            return None

    def _extract_embedding(self, aligned_face):
        """Hàm phụ trợ: Nhận ảnh mặt đã align, trả về list embeddings hoặc None."""
        if aligned_face is None:
            return None
        try:
            recog_results = self.face_recog_ensemble.predict([aligned_face])[0]
            embeddings = recog_results.get("norm_embeddings", None)
            return embeddings.tolist() if embeddings is not None else None
        except Exception as e:
            logger.error(f"Lỗi trích xuất embedding: {e}")
            return None

    def process_single_crop(self, person_img):
        if person_img is None or person_img.size == 0:
            return None
        return self._extract_embedding(self._crop_and_align_face(person_img))

    def process_batch(self, frames_list):
        if not frames_list:
            return {}  

        results = self.person_det_ensemble.predict(frames_list, self.person_threshold, verbose=False)
        results = results if isinstance(results, list) else [results]
        batch_person_crops, frame_id = self._crop_person_images_batch(frames_list, results)
        
        batch_output = {}
        for img_idx, person_crops in enumerate(batch_person_crops):
            img_results = []
            for idx, person_img in enumerate(person_crops):
                if person_img.size == 0:
                    continue
                
                emb = self._extract_embedding(self._crop_and_align_face(person_img))
                if emb is not None:
                    img_results.append({"person_idx": int(idx), "embeddings": emb})
                    
            batch_output[str(frame_id[img_idx])] = img_results
                        
        return batch_output



if __name__ == "__main__":
    pipeline = BatchFacePipeline(triton_host="localhost:9187", person_threshold=0.45)

    img_paths = [
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/test.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000910.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_133735_21052026_20260521_133652.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05202026/14581214_165159_20052026_IMG_20260520_165031.jpg",
        "http://107.120.93.24:9122/employee-faces/faces/26507931/26507931_1.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_135524_21052026_20260521_135452.jpg"
    ]
    
    batch_frames = []
    for p in img_paths:
        print(p)
        if p.startswith("http://") or p.startswith("https://"):
            try:
                response = requests.get(p, timeout=5) 
                if response.status_code == 200:
                    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        batch_frames.append(img)
                else:
                    logger.warning(f"Không thể tải ảnh từ URL (Status code: {response.status_code}): {p}")
            except Exception as e:
                logger.error(f"Lỗi khi download ảnh từ URL {p}: {e}")
        
        else:
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    batch_frames.append(img)
    
    logger.info(f"Loading batch iclude: {len(batch_frames)} imgs input into Class.")        
    
    all_batch_results = pipeline.process_batch(batch_frames)
    # print("all_batch_results: ", all_batch_results)
    
    # for f_id, objects_in_image in all_batch_results.items():
    #     print(f"\n================ KẾT QUẢ CỦA FRAME ID: {f_id} ================")
    #     print(f"-> Detected {len(objects_in_image)} valid face.")
        
    #     for obj in objects_in_image:
    #         print(f"  [Person {obj['person_idx']}]")
    #         print(f"  - Shape of vector Embedding: {obj['embeddings'].shape}")
    #         print(f"  - L2 Norm: {np.linalg.norm(obj['embeddings']):.4f}")
    import json
    with open("face_features_output.json", "w", encoding="utf-8") as f:
        json.dump(all_batch_results, f, ensure_ascii=False, indent=4)
        
    print("✓ Đã xuất cấu trúc JSON mong muốn ra file thành công!")
