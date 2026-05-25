import os
import sys
import logging
import cv2
import numpy as np
import glog as logger

from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
# from client.face_alignment import FaceExtPreClient
# from client.face_emmbedding import FaceRegClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class BatchFacePipeline:
#     def __init__(self, triton_host="localhost:9187", person_threshold=0.45, max_person_batch=32):
#         self.triton_host = triton_host
#         self.person_threshold = person_threshold
#         self.max_person_batch = max_person_batch
        
#         logger.info(f"Connecting Triton Server (Batch Mode) at: {self.triton_host}")
        
#         self.person_det_ensemble = PersonDetectionClient(
#             triton_host=self.triton_host, triton_model_name="person_detection_ensemble",
#             max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
#         )
#         self.face_det_ensemble = FaceEnsembleClient(
#             triton_host=self.triton_host, triton_model_name="pipeline_ensemble_Det", 
#             max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
#         )
#         # self.face_align_model = FaceExtPreClient(
#         #     triton_host=self.triton_host, triton_model_name="face_alignment_op", 
#         #     max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
#         # )
#         # self.face_recog_ensemble = FaceRegClient(
#         #     triton_host=self.triton_host, triton_model_name="pipeline_reg", 
#         #     max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
#         # )
#         logger.info("Init Batch Client Triton successfully!")

#     def _chunk_list(self, data_list, chunk_size):
#         for i in range(0, len(data_list), chunk_size):
#             yield data_list[i:i + chunk_size]

#     def _crop_person_images_batch(self, frames, results):
#         batch_cropped_images = []
#         for frame, res in zip(frames, results):
#             cropped_images_single_frame = []
#             h_img, w_img, _ = frame.shape
#             boxes = res.get("BOXES", [])

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w_img, x2), min(h_img, y2)
                
#                 if x2 > x1 and y2 > y1:
#                     cropped_images_single_frame.append(frame[y1:y2, x1:x2].copy())
#             batch_cropped_images.append(cropped_images_single_frame)
#         return batch_cropped_images  

#     def get_person(self, frames_list):
#         if not frames_list:
#             return []
        
#         all_results = []
#         for chunk in self._chunk_list(frames_list, self.max_person_batch):
#             results = self.person_det_ensemble.predict(chunk, self.person_threshold, verbose=False)
#             if isinstance(results, list):
#                 all_results.extend(results)
#             else:
#                 all_results.append(results)
                
#         return self._crop_person_images_batch(frames_list, all_results)

#     def get_face_batch(self, person_imgs_rgb):
#         if not person_imgs_rgb:
#             return []
        
#         all_face_results = []
#         for chunk in self._chunk_list(person_imgs_rgb, self.max_person_batch):
#             results = self.face_det_ensemble.predict(chunk)
#             if isinstance(results, list):
#                 all_face_results.extend(results)
#             else:
#                 all_face_results.append(results)
#         return all_face_results

#     def get_face_align_batch(self, person_imgs_rgb, face_det_results):
#         white_bg_batch = []
#         landmarks_batch = []
#         bboxes_batch = []
#         valid_indices = []

#         for idx, (img_rgb, res) in enumerate(zip(person_imgs_rgb, face_det_results)):
#             landmarks = res.get("final_landmarks", [])
#             bboxes = res.get("final_boxes", [])

#             if len(bboxes) == 0:
#                 continue

#             try:
#                 bbox = bboxes if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes
#                 x1, y1, x2, y2 = map(int, bbox[:4])
#                 w, h = x2 - x1, y2 - y1

#                 mx, my = int(w * .3), int(h * .3)
#                 x1 = max(0, x1 - mx)
#                 y1 = max(0, y1 - my)
#                 x2 = min(img_rgb.shape, x2 + mx)
#                 y2 = min(img_rgb.shape, y2 + my)

#                 face_crop = img_rgb[y1:y2, x1:x2]
#                 if face_crop.size == 0:
#                     continue

#                 white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
#                 white_bg[y1:y2, x1:x2] = face_crop

#                 white_bg_batch.append(white_bg)
#                 landmarks_batch.append(landmarks)
#                 bboxes_batch.append(bboxes)
#                 valid_indices.append(idx)
#             except Exception as e:
#                 logger.error(f"Error preparing align data at index {idx}: {e}")

#         if not white_bg_batch:
#             return [], []

#         aligned_faces = []
#         os.makedirs("./output_crops", exist_ok=True)
        
#         bg_chunks = self._chunk_list(white_bg_batch, self.max_person_batch)
#         lm_chunks = self._chunk_list(landmarks_batch, self.max_person_batch)
#         bx_chunks = self._chunk_list(bboxes_batch, self.max_person_batch)

#         global_idx = 0
#         for bg_c, lm_c, bx_c in zip(bg_chunks, lm_chunks, bx_chunks):
#             try:
#                 results_align = self.face_align_model.predict(bg_c, lm_c, bx_c)
#                 if not isinstance(results_align, list):
#                     results_align = [results_align]

#                 for res_align in results_align:
#                     aligned_112 = np.clip(
#                         res_align["face_aligned_112"] * 128.0 + 127.5, 0, 255
#                     ).astype(np.uint8)
                    
#                     aligned_face_bgr = aligned_112.transpose(1, 2, 0)
                    
#                     original_idx = valid_indices[global_idx]
#                     out_face_path = os.path.join("./output_crops", f"aligned_{original_idx}.jpg")
#                     cv2.imwrite(out_face_path, aligned_face_bgr)
                    
#                     aligned_faces.append(aligned_face_bgr)
#                     global_idx += 1
#             except Exception as e:
#                 logger.error(f"Error when batch alignment face chunk: {e}")
#                 global_idx += len(bg_c) 

#         return aligned_faces, valid_indices

#     def get_embedding_batch(self, aligned_faces):
#         if not aligned_faces:
#             return []
        
#         all_embeddings = []
#         for chunk in self._chunk_list(aligned_faces, self.max_person_batch):
#             try:
#                 recog_results = self.face_recog_ensemble.predict(chunk)
#                 if not isinstance(recog_results, list):
#                     recog_results = [recog_results]
#                 all_embeddings.extend([res.get("norm_embeddings", None) for res in recog_results])
#             except Exception as e:
#                 logger.error(f"Error when batch extracting embedding chunk: {e}")
#                 all_embeddings.extend([None] * len(chunk))
                
#         return all_embeddings

#     def process_batch(self, frames_list):
#         batch_person_crops = self.get_person(frames_list)
        
#         all_person_imgs = []
#         meta_mapping = [] 

#         for img_idx, person_crops in enumerate(batch_person_crops):
#             for idx, person_img in enumerate(person_crops):
#                 if person_img.size > 0:
#                     all_person_imgs.append(person_img)
#                     meta_mapping.append((img_idx, idx))

#         if not all_person_imgs:
#             return [[] for _ in range(len(frames_list))]

#         all_person_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_person_imgs]

#         face_det_results = self.get_face_batch(all_person_rgb)
#         aligned_faces, valid_indices = self.get_face_align_batch(all_person_rgb, face_det_results)

#         valid_meta = [meta_mapping[i] for i in valid_indices]
#         valid_person_imgs = [all_person_imgs[i] for i in valid_indices]

#         embeddings_list = self.get_embedding_batch(aligned_faces)

#         batch_output = [[] for _ in range(len(frames_list))]

#         for face_idx, embedding in enumerate(embeddings_list):
#             if embedding is not None:
#                 img_idx, person_idx = valid_meta[face_idx]
#                 batch_output[img_idx].append({
#                     "person_idx": person_idx,
#                     "person_crop": valid_person_imgs[face_idx],
#                     "aligned_face": aligned_faces[face_idx],
#                     "embeddings": embedding
#                 })

#         return batch_output


class BatchFacePipeline:
    def __init__(self, triton_host="localhost:9187", person_threshold=0.45, max_person_batch=32):
        self.triton_host = triton_host
        self.person_threshold = person_threshold
        self.max_person_batch = max_person_batch
        
        logger.info(f"Connecting Triton Server (Batch Mode) at: {self.triton_host}")
        
        # Giả định các class Client của bạn đã được định nghĩa trước đó
        self.person_det_ensemble = PersonDetectionClient(
            triton_host=self.triton_host, triton_model_name="person_detection_ensemble",
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        self.face_det_ensemble = FaceEnsembleClient(
            triton_host=self.triton_host, triton_model_name="pipeline_ensemble_Det", 
            max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        )
        # self.face_align_model = FaceExtPreClient(
        #     triton_host=self.triton_host, triton_model_name="face_alignment_op", 
        #     max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        # )
        # self.face_recog_ensemble = FaceRegClient(
        #     triton_host=self.triton_host, triton_model_name="pipeline_reg", 
        #     max_batch_size=self.max_person_batch, shared_memory=False, shared_cuda_memory=False
        # )
        logger.info("Init Batch Client Triton successfully!")

    def _chunk_list(self, data_list, chunk_size):
        for i in range(0, len(data_list), chunk_size):
            yield data_list[i:i + chunk_size]

    def _crop_person_images_batch(self, frames, results):
        batch_cropped_images = []
        for frame, res in zip(frames, results):
            cropped_images_single_frame = []
            h_img, w_img, _ = frame.shape
            boxes = res.get("BOXES", [])

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    cropped_images_single_frame.append(frame[y1:y2, x1:x2].copy())
            batch_cropped_images.append(cropped_images_single_frame)
        return batch_cropped_images  

    # 1. Get Person
    def get_person(self, frames_list):
        if not frames_list:
            return []
        
        all_results = []
        for chunk in self._chunk_list(frames_list, self.max_person_batch):
            results = self.person_det_ensemble.predict(chunk, self.person_threshold, verbose=False)
            if isinstance(results, list):
                all_results.extend(results)
            else:
                all_results.append(results)

        return self._crop_person_images_batch(frames_list, all_results)

    # 2. Get Face
    def get_face_batch(self, person_imgs_rgb):
        if not person_imgs_rgb:
            return []
        
        all_face_results = []
        for chunk in self._chunk_list(person_imgs_rgb, self.max_person_batch):
            results = self.face_det_ensemble.predict(chunk)
            if isinstance(results, list):
                all_face_results.extend(results)
            else:
                all_face_results.append(results)
        return all_face_results

    # 3. Get Face Align
    def get_face_align_batch(self, person_imgs_rgb, face_det_results):
        white_bg_batch = []
        landmarks_batch = []
        bboxes_batch = []
        valid_indices = []

        for idx, (img_rgb, res) in enumerate(zip(person_imgs_rgb, face_det_results)):
            landmarks = res.get("final_landmarks", [])
            bboxes = res.get("final_boxes", [])

            if len(bboxes) == 0:
                continue

            try:
                bbox = bboxes if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes
                x1, y1, x2, y2 = map(int, bbox[:4])
                w, h = x2 - x1, y2 - y1

                mx, my = int(w * .3), int(h * .3)
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(img_rgb.shape[1], x2 + mx)
                y2 = min(img_rgb.shape[0], y2 + my)

                face_crop = img_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
                white_bg[y1:y2, x1:x2] = face_crop

                white_bg_batch.append(white_bg)
                landmarks_batch.append(landmarks)
                bboxes_batch.append(bboxes)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Error preparing align data at index {idx}: {e}")

        if not white_bg_batch:
            return [], []

        aligned_faces = []
        os.makedirs("./output_crops", exist_ok=True)
        
        bg_chunks = self._chunk_list(white_bg_batch, self.max_person_batch)
        lm_chunks = self._chunk_list(landmarks_batch, self.max_person_batch)
        bx_chunks = self._chunk_list(bboxes_batch, self.max_person_batch)

        global_idx = 0
        for bg_c, lm_c, bx_c in zip(bg_chunks, lm_chunks, bx_chunks):
            try:
                results_align = self.face_align_model.predict(bg_c, lm_c, bx_c)
                if not isinstance(results_align, list):
                    results_align = [results_align]

                for res_align in results_align:
                    aligned_112 = np.clip(
                        res_align["face_aligned_112"] * 128.0 + 127.5, 0, 255
                    ).astype(np.uint8)
                    
                    aligned_face_bgr = aligned_112.transpose(1, 2, 0)
                    
                    original_idx = valid_indices[global_idx]
                    out_face_path = os.path.join("./output_crops", f"aligned_{original_idx}.jpg")
                    cv2.imwrite(out_face_path, aligned_face_bgr)
                    
                    aligned_faces.append(aligned_face_bgr)
                    global_idx += 1
            except Exception as e:
                logger.error(f"Error when batch alignment face chunk: {e}")
                global_idx += len(bg_c)

        return aligned_faces, valid_indices

    # 4. Get Embedding
    def get_embedding_batch(self, aligned_faces):
        if not aligned_faces:
            return []
        
        all_embeddings = []
        for chunk in self._chunk_list(aligned_faces, self.max_person_batch):
            try:
                recog_results = self.face_recog_ensemble.predict(chunk)
                if not isinstance(recog_results, list):
                    recog_results = [recog_results]
                all_embeddings.extend([res.get("norm_embeddings", None) for res in recog_results])
            except Exception as e:
                logger.error(f"Error when batch extracting embedding chunk: {e}")
                all_embeddings.extend([None] * len(chunk))
                
        return all_embeddings

    # Pipeline xử lý chính
    def process_batch(self, frames_list):
        batch_person_crops = self.get_person(frames_list)
        
        all_person_imgs = []
        meta_mapping = [] 

        # DEBUG_CROP_DIR = "debug_person_crops"
        # os.makedirs(DEBUG_CROP_DIR, exist_ok=True)

        for img_idx, person_crops in enumerate(batch_person_crops):
            for idx, person_img in enumerate(person_crops):
                if person_img.size > 0:
                    all_person_imgs.append(person_img)
                    meta_mapping.append((img_idx, idx))

                    # filename = f"crop_frame{img_idx}_person{idx}.jpg"
                    # filepath = os.path.join(DEBUG_CROP_DIR, filename)
                    # cv2.imwrite(filepath, person_img)

        if not all_person_imgs:
            return [[] for _ in range(len(frames_list))]

        all_person_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_person_imgs]
        print(len(all_person_rgb))

        face_det_results = self.get_face_batch(all_person_rgb)
        print("face_det_results: ", face_det_results)

        # aligned_faces, valid_indices = self.get_face_align_batch(all_person_rgb, face_det_results)

        # valid_meta = [meta_mapping[i] for i in valid_indices]
        # valid_person_imgs = [all_person_imgs[i] for i in valid_indices]

        # embeddings_list = self.get_embedding_batch(aligned_faces)

        # batch_output = [[] for _ in range(len(frames_list))]

        # for face_idx, embedding in enumerate(embeddings_list):
        #     if embedding is not None:
        #         img_idx, person_idx = valid_meta[face_idx]
        #         batch_output[img_idx].append({
        #             "person_idx": person_idx,
        #             "person_crop": valid_person_imgs[face_idx],
        #             "aligned_face": aligned_faces[face_idx],
        #             "embeddings": np.array(embedding) # Ép kiểu numpy array để in shape/norm không lỗi
        #         })

        # return batch_output


if __name__ == "__main__":
    pipeline = BatchFacePipeline(triton_host="localhost:9187", person_threshold=0.45)

    img_paths = [
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000910.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg"
    ]
    
    batch_frames = []
    for p in img_paths:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                batch_frames.append(img)
    
    logger.info(f"Loading batch include: {len(batch_frames)} imgs input into Class.")        
    
    all_batch_results = pipeline.process_batch(batch_frames)
    
    # for img_idx, objects_in_image in enumerate(all_batch_results):
    #     print(f"\n================ KẾT QUẢ ẢNH THỨ {img_idx} ================")
    #     print(f"-> Detected {len(objects_in_image)} valid face.")
        
    #     for obj in objects_in_image:
    #         print(f"  [Person {obj['person_idx']}]")
    #         print(f"  - Shape of vector Embedding: {obj['embeddings'].shape}")
    #         print(f"  - L2 Norm: {np.linalg.norm(obj['embeddings']):.4f}")
            