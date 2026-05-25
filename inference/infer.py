import cv2
import numpy as np
import glog as logger
import sys
import os
import logging

from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
from client.face_alignment import FaceExtPreClient
from client.face_emmbedding import FaceRegClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crop_person_images_batch(frames, results):
    batch_cropped_images = []

    for frame, res in zip(frames, results):
        cropped_images_single_frame = []
        h_img, w_img, _ = frame.shape
        boxes = res.get("BOXES", [])

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_img, int(x2))
            y2 = min(h_img, int(y2))
            
            if x2 > x1 and y2 > y1:
                person_crop = frame[y1:y2, x1:x2]
                cropped_images_single_frame.append(person_crop)
                
        batch_cropped_images.append(cropped_images_single_frame)
            
    return batch_cropped_images  


def crop_and_align_face(person_img, face_det_ensemble, face_align_model, save_dir, debug_name="face"):
    if person_img is None or person_img.size == 0:
        print("Batch images are none / null !")
        return None

    img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    batchs = [img_rgb]

    results = face_det_ensemble.predict(batchs)
    results = results[0]

    landmarks = results.get("final_landmarks", [])
    bboxes = results.get("final_boxes", [])

    if len(bboxes) == 0:
        return None

    try:
        bbox = bboxes[0] if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes

        x1, y1, x2, y2 = map(int, bbox[:4])
        w = x2 - x1
        h = y2 - y1

        mx = int(w * .3)
        my = int(h * .3)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(img_rgb.shape[1], x2 + mx)
        y2 = min(img_rgb.shape[0], y2 + my)

        face_crop = img_rgb[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
        white_bg[y1:y2, x1:x2] = face_crop
        batchs_align = [white_bg]
        
        # white_bg_bgr = cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR)
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(save_dir, f"debug_white_bg_{debug_name}.jpg"), white_bg_bgr)

        results_align = face_align_model.predict(
            batchs_align,
            landmarks,
            bboxes
        )

        aligned_112 = np.clip(
            results_align[0]["face_aligned_112"] * 128.0 + 127.5,
            0,
            255
        ).astype(np.uint8)

        aligned_112 = aligned_112.transpose(1, 2, 0)
           
        # out_face_path = os.path.join(save_dir, f"aligned_{debug_name}.jpg")
        # cv2.imwrite(out_face_path, aligned_112)
        # print(f"Saved aligned face to: {out_face_path}")

        return aligned_112

    except Exception as e:
        print(f"Failed to align face for: {debug_name}")
        print(e)
        return None
    
def visualize_detections(frame, results, output_path=None):
    boxes = results.get("BOXES", [])
    scores = results.get("SCORES", [])

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person: {score:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, frame)
    return frame

def load_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return frame

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Person and Face Detection Pipeline")
    parser.add_argument("--host", type=str, default="localhost:9187", help="Triton server host")
    parser.add_argument("--output", type=str, default="result_person_detection.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--save-dir", type=str, default="output_crops", help="Directory to save crops and aligned faces")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        person_det_ensemble = PersonDetectionClient(
            triton_host=args.host, triton_model_name="person_detection_ensemble",
            max_batch_size=32, shared_memory=False, shared_cuda_memory=False
        )
        face_det_ensemble = FaceEnsembleClient(
            triton_host=args.host, triton_model_name="pipeline_ensemble_Det", 
            max_batch_size=1, shared_memory=False, shared_cuda_memory=False
        )
        face_align_model = FaceExtPreClient(
            triton_host=args.host, triton_model_name="face_alignment_op", 
            max_batch_size=1, shared_memory=False, shared_cuda_memory=False
        )
        face_recog_ensemble = FaceRegClient(
            triton_host=args.host, triton_model_name="pipeline_reg", 
            max_batch_size=1, shared_memory=False, shared_cuda_memory=False
        )

        img_paths = [
            "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg"
            # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/test.jpg",
            # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/aa.png",
            # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/11516413_anh B.jpg",
            # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000900/frame_000900_3.jpg",
            # "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_007215/frame_007215_1.jpg"
        ]
        
        batch_frames = [load_image(p) for p in img_paths]
        logger.info(f"Processing batch iclude: {len(batch_frames)} images.")        
        
        results = person_det_ensemble.predict(batch_frames, args.threshold, verbose=True)
        # print("results: ", results)
        if not isinstance(results, list):
            results = [results]

        batch_person_crops = crop_person_images_batch(batch_frames, results)

        for img_idx, person_crops in enumerate(batch_person_crops):
            # logger.info(f"Ảnh gốc thứ {img_idx} cắt được {len(person_crops)} ảnh người.")
            
            for idx, person_img in enumerate(person_crops):
                if person_img.size == 0:
                    continue
                    
                # person_img_name = f"frame_{img_idx}_person_{idx}.jpg"
                # person_img_path = os.path.join(args.save_dir, person_img_name)
                # cv2.imwrite(person_img_path, person_img)

                aligned_face = crop_and_align_face(
                    person_img=person_img,            
                    face_det_ensemble=face_det_ensemble,
                    face_align_model=face_align_model,
                    save_dir=args.save_dir,
                    debug_name=f"frame_{img_idx}_person_{idx}"
                )
            
                if aligned_face is not None:
                    logger.info(f"Processed Face Align sucessfully for person {idx}")
                
                    batch = [aligned_face]
                    results = face_recog_ensemble.predict(batch)
                    results = results[0]
                    print(results["norm_embeddings"].shape)
                    print(np.linalg.norm(results["norm_embeddings"]))

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
