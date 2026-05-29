import os
import sys
import logging
import cv2
import numpy as np
import glog as logger
import requests 
import math

from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
from client.face_alignment import FaceExtPreClient
from client.face_emmbedding import FaceRegClient
from client.face_mask import FaceMaskClient

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
        self.face_mask_model = FaceMaskClient(
            triton_host=self.triton_host, triton_model_name="mask", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_recog_ensemble = FaceRegClient(
            triton_host=self.triton_host, triton_model_name="pipeline_reg", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.min_face_w = 30
        self.min_face_h = 30

        logger.info("Init Batch Client Triton successfully!")

    def _quick_yaw_estimate(self, landmarks):
        try:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            eye_dist = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
            mouth_dist = math.hypot(right_mouth[0] - left_mouth[0], right_mouth[1] - left_mouth[1])

            dist_left_eye_to_nose = math.hypot(nose[0] - left_eye[0], nose[1] - left_eye[1])
            dist_right_eye_to_nose = math.hypot(nose[0] - right_eye[0], nose[1] - right_eye[1])
            mean_eye_to_nose = (dist_left_eye_to_nose + dist_right_eye_to_nose) / 2.0

            # mắt quá sát nhau < 5 pixel
            if mean_eye_to_nose == 0 or eye_dist < 5.0: 
                return False, 0.0, 999.0
                
            geometry_ratio = eye_dist / mean_eye_to_nose
            
            # khoảng cách 2 mắt nhỏ hơn 40% khoảng cách từ mắt đến mũi
            if geometry_ratio < 0.4: 
                return False, 0.0, 999.0

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll_angle = math.degrees(math.atan2(dy, dx))

            if dist_right_eye_to_nose == 0:
                return True, roll_angle, 999.0

            yaw_ratio = dist_left_eye_to_nose / dist_right_eye_to_nose
            yaw_score = (yaw_ratio - 1.0) if yaw_ratio >= 1.0 else ((1.0 / yaw_ratio) - 1.0)

            return True, roll_angle, yaw_score
        except Exception as e:
            print(f"Lỗi hệ thống khi tính toán góc: {e}", flush=True)
            return 0.0
        
    def _check_face_size(self, w, h):
        ok = (
            w >= self.min_face_w and
            h >= self.min_face_h
        )
        return ok, w, h

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

    def _select_topmost_face(self, bboxes, landmarks):
        """
        Chọn khuôn mặt nằm ở vị trí trên cùng nhất (y1 nhỏ nhất).
        Trả về: selected_bbox, selected_landmarks, best_idx
        """
        if len(bboxes) == 0:
            return None, None, None
        
        if isinstance(bboxes, list) and not isinstance(bboxes[0], (list, np.ndarray)):
            bboxes = [bboxes]
            landmarks = [landmarks]
        elif hasattr(bboxes, 'ndim') and bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
            landmarks = np.expand_dims(landmarks, axis=0)
            
        best_idx = 0
        if len(bboxes) > 1:
            min_y1 = float('inf')
            for i, bbox in enumerate(bboxes):
                current_y1 = bbox[1]  # y1 là phần tử thứ 2 trong [x1, y1, x2, y2]
                if current_y1 < min_y1:
                    min_y1 = current_y1
                    best_idx = i
            print(f"Detected {len(bboxes)} faces. Selected {best_idx} top face.")
            
        return bboxes[best_idx], landmarks[best_idx], best_idx


    def _crop_and_align_face(self, person_img, idx=0, frame_id="single"):
        if person_img is None or person_img.size == 0:
            return None
        
        out_face_path = os.path.join("./output_crops", f"frame_{frame_id}_person_{idx}.jpg")
        cv2.imwrite(out_face_path,  person_img)
        print(f"Saved aligned face to: {out_face_path}")

        img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

        results = self.face_det_ensemble.predict([img_rgb])
        results = results[0]
        print("results face:", results)

        landmarks = results.get("final_landmarks", [])
        bboxes = results.get("final_boxes", [])

        if len(bboxes) == 0:
            print("None face")
            return None, None
        
        selected_bbox, selected_landmarks, _ = self._select_topmost_face(bboxes, landmarks)
        
        yal_valid, roll_angle, yaw_score = self._quick_yaw_estimate(selected_landmarks)
        # print("roll_angle: ", roll_angle)
        # print("yaw_score: ", yaw_score)
        # print("yal_valid: ", yal_valid)
        if not yal_valid:
            print("invalid yaw")
            return None, None

        try:
            # bbox = bboxes[0] if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes
            x1, y1, x2, y2 = map(int, selected_bbox[:4])
            w, h = x2 - x1, y2 - y1
            if not self._check_face_size(w, h):
                print(f"Invalid size face {w}, {h} (w,h) !")
                return None, None
            # print(f"Valid face {w}, {h} (w,h) !")
            mx, my = int(w * .3), int(h * .3)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(img_rgb.shape[1], x2 + mx)
            y2 = min(img_rgb.shape[0], y2 + my)

            # # 1. Cắt khuôn mặt từ ảnh gốc (đang là RGB)
            # face_crop_1 = img_rgb[y1:y2, x1:x2].copy() # Dùng .copy() để tránh ghi đè ảnh gốc
            
            # # 2. Vẽ landmark lên ảnh khuôn mặt đã cắt
            # # Giả định landmarks[0] là danh sách điểm [(x, y), (x, y), ...] của khuôn mặt đầu tiên
            # if len(landmarks) > 0:
            #     for pt in landmarks:
            #         # pt chắc chắn là một cặp tọa độ [x, y] nhờ cấu trúc thực tế của bạn
            #         pt_x = int(pt[0]) - x1
            #         pt_y = int(pt[1]) - y1
                    
            #         # Kiểm tra điểm vẽ có nằm trong vùng ảnh khuôn mặt đã cắt không
            #         if 0 <= pt_x < face_crop_1.shape[1] and 0 <= pt_y < face_crop_1.shape[0]:
            #             # Vẽ hình tròn màu Đỏ=(255, 0, 0) lên ảnh RGB, bán kính=2, độ dày=-1 (tô đặc)
            #             cv2.circle(face_crop_1, (pt_x, pt_y), 2, (255, 0, 0), -1)

            # # 3. Chuyển từ RGB về BGR để lưu bằng OpenCV không bị sai màu
            # face_crop_bgr = cv2.cvtColor(face_crop_1, cv2.COLOR_RGB2BGR)
            
            # # 4. Lưu ảnh
            # os.makedirs("./output_crops", exist_ok=True)
            # out_face_path = os.path.join("./output_crops", f"face_frame_{frame_id}_idx_{idx}.jpg")
            # cv2.imwrite(out_face_path, face_crop_bgr)

            face_crop = img_rgb[y1:y2, x1:x2]
            
            # out_face_path = os.path.join("./output_crops", f"test_face.jpg")
            # cv2.imwrite(out_face_path,  face_crop)
            # print(f"Saved aligned face to: {out_face_path}")

            if face_crop.size == 0:
                return None, None

            white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
            white_bg[y1:y2, x1:x2] = face_crop

            results_align = self.face_align_model.predict([white_bg], landmarks, bboxes)
            
            aligned_112 = np.clip(
                results_align[0]["face_aligned_112"] * 128.0 + 127.5, 0, 255
            ).astype(np.uint8).transpose(1, 2, 0)

            # out_face_path = os.path.join("./output_crops", f"_frame_{frame_id}_aligned_{idx}.jpg")
            # cv2.imwrite(out_face_path,  aligned_112)
            # print(f"Saved aligned face to: {out_face_path}")

            # 1. Lấy ảnh gốc dạng float32
            face_fp32 = results_align[0]["face_aligned_nhwc"]

            # align_mask_save = np.clip(face_fp32 * 255.0, 0, 255).astype(np.uint8)
            # out_face_path = os.path.join("./output_crops", f"_frame_{frame_id}_inputMask_{idx}.jpg")
            # cv2.imwrite(out_face_path, align_mask_save)
            # print(f"Saved aligned face to: {out_face_path}")

            # # 3. CHUẨN BỊ ẢNH CHO MODEL MASK: Đảo kênh màu sang BGR (vì OpenCV đọc ảnh dạng BGR)
            # # Giữ nguyên cấu trúc 3 chiều [112, 112, 3] không dùng expand_dims vì predict() tự thêm batch
            # img_bgr = face_fp32[:, :, ::-1] 

            # # --- THỬ NGHIỆM ĐẦU VÀO MODEL MASK ---
            # # TRƯỜNG HỢP A: Nếu model Face Mask của bạn nhận dải [0.0 - 1.0] (Mặc định thử cách này trước)
            # # input_for_mask = img_bgr.astype(np.float32)

            # # TRƯỜNG HỢP B: Nếu kết quả VẪN RA MASK, hãy bỏ comment dòng dưới đây (Model nhận dải 0-255)
            # input_for_mask = (img_bgr * 255.0).astype(np.float32)

            # 4. DỰ ĐOÁN
            res_final = self.face_mask_model.predict(face_fp32)
            print("res_final: ", res_final)

            # Trích xuất kết quả từ dictionary bên trong list
            (mask, withoutMask) = res_final[0]['dense_1']
            label = "Mask" if mask > withoutMask else "No Mask"
            print(f"Kết quả phân loại: {label} (Mask: {mask:.4f} | NoMask: {withoutMask:.4f})")
                        
            return aligned_112, selected_bbox[:4]

        except Exception as e:
            logger.error(f"Error when alignment face: {e}")
            return None, None

    def _extract_embedding(self, aligned_face):
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
        aligned_face, _ = self._crop_and_align_face(person_img)
        # aligned_face = self._crop_and_align_face(person_img)
        return self._extract_embedding(aligned_face)

    def process_batch(self, frames_list):
        if not frames_list:
            return {}  

        MAX_TRITON_BATCH = 8
        batch_output = {}
        
        # Chia toàn bộ frames_list thành các cụm nhỏ tối đa 8 ảnh
        for chunk_idx in range(0, len(frames_list), MAX_TRITON_BATCH):
            sub_frames = frames_list[chunk_idx : chunk_idx + MAX_TRITON_BATCH]
            
            # 1. Dự đoán detector cho sub-batch hiện tại
            sub_results = self.person_det_ensemble.predict(
                sub_frames, self.person_threshold, verbose=False
            )
            if not isinstance(sub_results, list):
                sub_results = [sub_results]
                
            # 2. Cắt ảnh người theo đúng sub-batch để đảm bảo frame_id đồng bộ, không bị trùng
            sub_person_crops, sub_frame_ids = self._crop_person_images_batch(sub_frames, sub_results)
            
            # 3. Xử lý trích xuất khuôn mặt cho từng ảnh trong sub-batch
            for img_idx, person_crops in enumerate(sub_person_crops):
                img_results = []
                
                # Tính toán frame_id thực tế dựa trên vị trí chunk gốc (nếu sub_frame_ids bị reset về 0)
                # Hoặc dùng trực tiếp sub_frame_ids[img_idx] nếu hàm của bạn sinh ID động
                actual_frame_id = str(chunk_idx + img_idx) 
                
                for idx, person_img in enumerate(person_crops):
                    if person_img.size == 0:
                        continue
                    
                    # Cắt và căn chỉnh khuôn mặt
                    aligned_face, face_box = self._crop_and_align_face(person_img, idx, actual_frame_id)
                    emb = self._extract_embedding(aligned_face)
                    
                    if emb is not None:
                        img_results.append({
                            "person_idx": int(idx), 
                            "face_box": face_box.tolist(),
                            "embeddings": emb
                        })
                        
                batch_output[actual_frame_id] = img_results
                        
        return batch_output




if __name__ == "__main__":
    pipeline = BatchFacePipeline(triton_host="localhost:9187", person_threshold=0.45)

    img_paths = [
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/test.jpg"
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/vlcsnap-2026-03-05-15h40m18s599.png"
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000900.jpg",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách bổ sung MM + Vendor/Vendor File 2/26052026_019203002089-Nguyrn Tu Viet Hung- SRtech rear/05262026/14581214_153133_26052026_IMG_20260526_153035.jpg"
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000960.jpg",
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/vlcsnap-2026-05-20-14h56m16s042.png"
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/Screenshot 2026-05-28 140815.png",
        # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/Screenshot 2026-05-28 141606.png",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_140935_21052026_IMG_20260521_140807.jpg",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 3/05212026/14581214_134031_21052026_IMG_20260521_133940.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000960.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/vlcsnap-2026-03-05-15h40m18s599.png"
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_202232_21052026_IMG_20260521_202157.jpg",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_143237_21052026_20260521_142906.jpg",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05202026/14581214_165159_20052026_IMG_20260520_165012.jpg",
        # "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_104539_21052026_IMG_20260521_104509.jpg"
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
    # all_batch_results = pipeline.process_single_crop(batch_frames[0])
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
