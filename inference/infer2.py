import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import json
import requests

from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
from client.face_alignment import FaceExtPreClient
from client.face_emmbedding import FaceRegClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BatchFacePipeline")

@dataclass
class FaceEmbeddingResult:
    """Encapsulates one face embedding in an image."""
    person_idx: int
    embeddings: List[float]

class BatchFacePipeline:
    def __init__(
        self,
        triton_host: str = "localhost:9187",
        person_model: str = "person_detection_ensemble",
        face_det_model: str = "pipeline_ensemble_Det",
        face_align_model: str = "face_alignment_op",
        face_emb_model: str = "pipeline_reg",
        person_threshold: float = 0.45,
        max_person_batch: int = 16
    ):
        """
        Khởi tạo client kết nối các model trên Triton server.
        Cho phép truyền động toàn bộ tham số cấu hình, model.
        """
        self.triton_host = triton_host
        self.person_threshold = person_threshold
        self.max_person_batch = max_person_batch

        logger.info(f"Connecting Triton Server (Batch Mode) at: {self.triton_host}")

        self.person_det_ensemble = PersonDetectionClient(
            triton_host=self.triton_host, triton_model_name=person_model,
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_det_ensemble = FaceEnsembleClient(
            triton_host=self.triton_host, triton_model_name=face_det_model,
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_align_model = FaceExtPreClient(
            triton_host=self.triton_host, triton_model_name=face_align_model,
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_recog_ensemble = FaceRegClient(
            triton_host=self.triton_host, triton_model_name=face_emb_model,
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        logger.info("Init Batch Client Triton successfully!")

    def _crop_person_images_batch(
        self,
        frames: List[np.ndarray],
        results: List[Dict[str, Any]]
    ) -> Tuple[List[List[np.ndarray]], List[int]]:
        """
        Cắt crop các vùng phát hiện người trên từng frame.
        Trả về list các ảnh cropped và id frame tương ứng.
        """
        batch_cropped_images = []
        frame_ids = []
        for idx, (frame, res) in enumerate(zip(frames, results)):
            det_data = res.get("detection", res) if isinstance(res, dict) else res
            actual_frame_idx = res.get("frame_idx", idx) if isinstance(res, dict) else idx

            cropped_images_single_frame = []
            if frame is not None and hasattr(frame, "shape"):
                h_img, w_img = frame.shape[:2]
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

    def _crop_and_align_face(self, person_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Cắt vùng mặt và căn chỉnh khuôn mặt trên ảnh person đó.
        Kết quả trả về là ảnh đã align kích thước (112,112,3), None nếu fail.
        """
        if person_img is None or person_img.size == 0:
            return None

        try:
            img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Could not convert image to RGB: {e}")
            return None

        try:
            results = self.face_det_ensemble.predict([img_rgb])
            results = results[0] if results and isinstance(results, list) else {}
            landmarks = results.get("final_landmarks", [])
            bboxes = results.get("final_boxes", [])
            
            # Khắc phục lỗi mảng NumPy khi kiểm tra điều kiện 'if not bboxes'
            if bboxes is None or (isinstance(bboxes, np.ndarray) and bboxes.size == 0) or (isinstance(bboxes, list) and len(bboxes) == 0): 
                return None

            # Đảm bảo bboxes và landmarks được chuyển về Python List để tránh lỗi Ambiguous bên trong face_align_model
            if isinstance(bboxes, np.ndarray):
                bboxes = bboxes.tolist()
            if isinstance(landmarks, np.ndarray):
                landmarks = landmarks.tolist()

            bbox = bboxes[0] if isinstance(bboxes, list) else bboxes

            x1, y1, x2, y2 = map(int, bbox[:4])
            w, h = x2 - x1, y2 - y1
            mx, my = int(.3 * w), int(.3 * h)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(img_rgb.shape[1], x2 + mx), min(img_rgb.shape[0], y2 + my)

            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            # Nhúng lên nền trắng
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

    def _extract_embedding(self, aligned_face: np.ndarray) -> Optional[List[float]]:
        """
        Từ ảnh mặt đã được align trả về embedding (chuẩn hóa), hoặc None nếu lỗi.
        """
        if aligned_face is None:
            return None
        try:
            recog_results = self.face_recog_ensemble.predict([aligned_face])
            recog_results = recog_results[0] if recog_results and isinstance(recog_results, list) else {}
            embeddings = recog_results.get("norm_embeddings", None)
            return embeddings.tolist() if embeddings is not None else None
        except Exception as e:
            logger.error(f"Lỗi trích xuất embedding: {e}")
            return None

    def process_single_crop(self, person_img: np.ndarray) -> Optional[List[float]]:
        """
        Pipeline cho 1 ảnh người: align, extract embedding.
        """
        return self._extract_embedding(self._crop_and_align_face(person_img))

    def process_batch(self, frames_list: List[np.ndarray]) -> Dict[str, List[Dict]]:
        """
        Pipeline batch hóa cho list frame. Trả về dict {frame_id: [FaceEmbeddingResult...]}.
        """
        if not frames_list:
            return {}

        try:
            results = self.person_det_ensemble.predict(
                frames_list, self.person_threshold, verbose=False
            )
            results = results if isinstance(results, list) else [results]
            batch_person_crops, frame_ids = self._crop_person_images_batch(frames_list, results)
            batch_output: Dict[str, List[Dict]] = {}
            for img_idx, person_crops in enumerate(batch_person_crops):
                img_results = []
                for idx, person_img in enumerate(person_crops):
                    if person_img is None or person_img.size == 0:
                        continue
                    emb = self.process_single_crop(person_img)
                    if emb is not None:
                        res = FaceEmbeddingResult(person_idx=int(idx), embeddings=emb)
                        img_results.append(asdict(res))
                batch_output[str(frame_ids[img_idx])] = img_results
            return batch_output
        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
            return {}

def load_images(img_paths: List[str], timeout: int = 8) -> List[np.ndarray]:
    """
    Load a list of images from local paths or HTTP(S) URLs.
    """
    images = []
    for path in img_paths:
        logger.info(f"Loading image: {path}")
        if path.startswith("http://") or path.startswith("https://"):
            try:
                response = requests.get(path, timeout=timeout)
                if response.status_code == 200:
                    arr = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
                    else:
                        logger.warning(f"Không thể giải mã ảnh từ URL: {path}")
                else:
                    logger.warning(f"Không thể tải ảnh từ URL (Status code: {response.status_code}): {path}")
            except Exception as e:
                logger.error(f"Lỗi khi download ảnh từ URL {path}: {e}")
        else:
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                else:
                    logger.warning(f"Không thể đọc ảnh từ file: {path}")
            else:
                logger.warning(f"File không tồn tại: {path}")
    return images

def save_results_to_json(results: Dict, output_path: str):
    """
    Lưu dict kết quả ra file JSON.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"✓ Đã xuất cấu trúc JSON mong muốn ra file: {output_path}")
    except Exception as e:
        logger.error(f"Lỗi khi ghi file JSON kết quả: {e}")

def main():
    img_paths = [
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000910.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_133735_21052026_20260521_133652.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05202026/14581214_165159_20052026_IMG_20260520_165031.jpg",
        "http://107.120.93.24:9122/employee-faces/faces/26507931/26507931_1.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_135524_21052026_20260521_135452.jpg"
    ]
    images = load_images(img_paths)
    logger.info(f"Đã load được {len(images)} ảnh đầu vào.")

    pipeline = BatchFacePipeline()
    all_batch_results = pipeline.process_batch(images)
    save_results_to_json(all_batch_results, "face_features_output.json")

if __name__ == "__main__":
    main()