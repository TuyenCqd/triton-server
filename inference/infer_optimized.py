import os
import sys
import logging
import cv2
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_BATCH_ALIGNMENT = 16
MAX_BATCH_EMBEDDING = 32
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# ==================== DATA CLASSES ====================
@dataclass
class FaceInfo:
    """Metadata for a face"""
    frame_id: str
    person_idx: int
    img: np.ndarray
    
@dataclass
class AlignedFace:
    """Aligned face with metadata"""
    frame_id: str
    person_idx: int
    aligned_img: np.ndarray
    
@dataclass
class EmbeddingResult:
    """Final embedding result"""
    frame_id: str
    person_idx: int
    embeddings: List[float]

# ==================== UTILITIES ====================
def create_session_with_retries(max_retries=MAX_RETRIES):
    """Create requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def validate_image(img: np.ndarray, img_name: str = "") -> bool:
    """Validate image shape and content"""
    if img is None:
        logger.warning(f"Image {img_name} is None")
        return False
    
    if img.size == 0:
        logger.warning(f"Image {img_name} is empty")
        return False
    
    if len(img.shape) != 3:
        logger.warning(f"Image {img_name} must be 3D, got shape {img.shape}")
        return False
    
    return True

def load_image_from_file(path: str) -> Optional[np.ndarray]:
    """Load image from local file"""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    
    if Path(path).suffix.lower() not in ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file extension: {path}")
        return None
    
    img = cv2.imread(path)
    if not validate_image(img, path):
        return None
    
    return img

def load_image_from_url(url: str, session: requests.Session) -> Optional[np.ndarray]:
    """Load image from URL with retry"""
    try:
        response = session.get(url, timeout=DEFAULT_TIMEOUT, verify=True)
        
        if response.status_code != 200:
            logger.warning(f"Failed to load URL (Status {response.status_code}): {url}")
            return None
        
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if not validate_image(img, url):
            return None
        
        return img
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading URL {url}: {e}", exc_info=True)
        return None

# ==================== MAIN PIPELINE ====================
from client.person_detection import PersonDetectionClient
from client.face_detection import FaceEnsembleClient
from client.face_alignment import FaceExtPreClient
from client.face_emmbedding import FaceRegClient  # Note: typo in original

class BatchFacePipeline:
    def __init__(
        self,
        triton_host: str = "localhost:9187",
        person_threshold: float = 0.45,
        max_person_batch: int = 16,
        max_batch_alignment: int = MAX_BATCH_ALIGNMENT,
        max_batch_embedding: int = MAX_BATCH_EMBEDDING
    ):
        """Initialize face pipeline with Triton clients"""
        self.triton_host = triton_host
        self.person_threshold = person_threshold
        self.max_person_batch = max_person_batch
        self.max_batch_alignment = max_batch_alignment
        self.max_batch_embedding = max_batch_embedding
        
        logger.info(f"Connecting Triton Server (Batch Mode) at: {self.triton_host}")
        
        try:
            self.person_det_ensemble = PersonDetectionClient(
                triton_host=self.triton_host,
                triton_model_name="person_detection_ensemble",
                max_batch_size=self.max_person_batch,
                shared_memory=False,
                shared_cuda_memory=False
            )
            self.face_det_ensemble = FaceEnsembleClient(
                triton_host=self.triton_host,
                triton_model_name="pipeline_ensemble_Det",
                max_batch_size=self.max_person_batch,
                shared_memory=False,
                shared_cuda_memory=False
            )
            self.face_align_model = FaceExtPreClient(
                triton_host=self.triton_host,
                triton_model_name="face_alignment_op",
                max_batch_size=self.max_batch_alignment,
                shared_memory=False,
                shared_cuda_memory=False
            )
            self.face_recog_ensemble = FaceRegClient(
                triton_host=self.triton_host,
                triton_model_name="pipeline_reg",
                max_batch_size=self.max_batch_embedding,
                shared_memory=False,
                shared_cuda_memory=False
            )
            logger.info("✓ Successfully initialized all Triton clients")
        except Exception as e:
            logger.error(f"Failed to initialize Triton clients: {e}", exc_info=True)
            raise

    def _crop_person_images_batch(
        self,
        frames: List[np.ndarray],
        results: List[Dict]
    ) -> Tuple[List[List[np.ndarray]], List]:
        """
        Crop person regions from detection results
        
        Args:
            frames: List of input frames
            results: Person detection results
            
        Returns:
            (batch_cropped_images, frame_ids)
        """
        batch_cropped_images = []
        frame_ids = []
        
        for idx, (frame, res) in enumerate(zip(frames, results)):
            # Validate frame
            if not validate_image(frame, f"frame_{idx}"):
                batch_cropped_images.append([])
                frame_ids.append(idx)
                continue
            
            # Extract detection data with fallback
            if isinstance(res, dict):
                det_data = res.get("detection", res)
                actual_frame_idx = res.get("frame_idx", idx)
            else:
                det_data = res
                actual_frame_idx = idx
            
            # Extract boxes with validation
            if isinstance(det_data, dict):
                boxes = det_data.get("BOXES", det_data.get("tmp_boxes", []))
            else:
                logger.warning(f"Invalid detection data format for frame {idx}")
                boxes = []
            
            cropped_images_single_frame = []
            h_img, w_img = frame.shape[:2]
            
            for box in boxes:
                try:
                    # Validate box
                    if not isinstance(box, (list, tuple, np.ndarray)) or len(box) < 4:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_img = frame[y1:y2, x1:x2]
                        if cropped_img.size > 0:
                            cropped_images_single_frame.append(cropped_img)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing box {box}: {e}")
                    continue
            
            batch_cropped_images.append(cropped_images_single_frame)
            frame_ids.append(actual_frame_idx)
        
        return batch_cropped_images, frame_ids

    def _batch_crop_and_align_faces(
        self,
        faces_info: List[FaceInfo]
    ) -> List[AlignedFace]:
        """
        Batch alignment: Process multiple faces at once
        
        Args:
            faces_info: List of FaceInfo objects with image data
            
        Returns:
            List of aligned faces
        """
        if not faces_info:
            return []
        
        logger.info(f"Starting batch alignment for {len(faces_info)} faces")
        start_time = time.time()
        
        rgb_images = []
        detections = []
        valid_indices = []
        
        # Step 1: Convert to RGB and detect faces
        for i, face in enumerate(faces_info):
            try:
                if not validate_image(face.img, f"face_{i}"):
                    continue
                
                img_rgb = cv2.cvtColor(face.img, cv2.COLOR_BGR2RGB)
                rgb_images.append(img_rgb)
                
                # Single face detection
                det_result = self.face_det_ensemble.predict([img_rgb])[0]
                detections.append(det_result)
                valid_indices.append(i)
                
            except Exception as e:
                logger.error(f"Error in face detection for face {i}: {e}")
                continue
        
        if not rgb_images:
            logger.warning("No valid faces to align")
            return []
        
        logger.info(f"Detected faces in {time.time() - start_time:.2f}s")
        
        # Step 2: Prepare alignment inputs
        white_bgs = []
        landmarks_list = []
        bboxes_list = []
        face_metadata = []
        
        for i, det_result in enumerate(detections):
            try:
                landmarks = det_result.get("final_landmarks", [])
                bboxes = det_result.get("final_boxes", [])
                
                if len(bboxes) == 0:
                    logger.debug(f"No face detected in person crop {i}")
                    continue
                
                bbox = bboxes[0] if isinstance(bboxes, list) or len(bboxes.shape) > 1 else bboxes
                x1, y1, x2, y2 = map(int, bbox[:4])
                w, h = x2 - x1, y2 - y1
                
                # Expand region by 30%
                mx, my = int(w * 0.3), int(h * 0.3)
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(rgb_images[i].shape[1], x2 + mx)
                y2 = min(rgb_images[i].shape[0], y2 + my)
                
                face_crop = rgb_images[i][y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # Create white background
                white_bg = np.ones_like(rgb_images[i], dtype=np.uint8) * 255
                white_bg[y1:y2, x1:x2] = face_crop
                
                white_bgs.append(white_bg)
                landmarks_list.append(landmarks)
                bboxes_list.append(bboxes)
                face_metadata.append({
                    "frame_id": faces_info[valid_indices[i]].frame_id,
                    "person_idx": faces_info[valid_indices[i]].person_idx
                })
                
            except Exception as e:
                logger.error(f"Error preparing alignment for face {i}: {e}")
                continue
        
        if not white_bgs:
            logger.warning("No valid faces prepared for alignment")
            return []
        
        # Step 3: BATCH ALIGNMENT
        try:
            logger.info(f"Batch aligning {len(white_bgs)} faces")
            align_start = time.time()
            
            results_align = self.face_align_model.predict(
                white_bgs,
                landmarks_list,
                bboxes_list
            )
            
            logger.info(f"Batch alignment completed in {time.time() - align_start:.2f}s")
            
            aligned_faces = []
            for i, metadata in enumerate(face_metadata):
                try:
                    aligned_112 = np.clip(
                        results_align[i]["face_aligned_112"] * 128.0 + 127.5,
                        0,
                        255
                    ).astype(np.uint8)
                    
                    aligned_faces.append(
                        AlignedFace(
                            frame_id=metadata["frame_id"],
                            person_idx=metadata["person_idx"],
                            aligned_img=aligned_112.transpose(1, 2, 0)
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing aligned face {i}: {e}")
                    continue
            
            logger.info(f"Successfully aligned {len(aligned_faces)} faces")
            return aligned_faces
            
        except Exception as e:
            logger.error(f"Error in batch alignment: {e}", exc_info=True)
            return []

    def _batch_get_embeddings(
        self,
        aligned_faces: List[AlignedFace]
    ) -> List[EmbeddingResult]:
        """
        Batch embedding: Extract embeddings for multiple faces
        
        Args:
            aligned_faces: List of aligned faces
            
        Returns:
            List of embedding results
        """
        if not aligned_faces:
            return []
        
        logger.info(f"Starting batch embedding for {len(aligned_faces)} faces")
        start_time = time.time()
        
        images = [face.aligned_img for face in aligned_faces]
        
        try:
            logger.info(f"Batch predicting embeddings for {len(images)} faces")
            embed_start = time.time()
            
            recog_results = self.face_recog_ensemble.predict(images)
            
            logger.info(f"Batch embedding completed in {time.time() - embed_start:.2f}s")
            
            results = []
            for i, face in enumerate(aligned_faces):
                try:
                    embeddings = recog_results[i].get("norm_embeddings", None)
                    
                    if embeddings is not None:
                        results.append(
                            EmbeddingResult(
                                frame_id=face.frame_id,
                                person_idx=face.person_idx,
                                embeddings=embeddings.tolist()
                            )
                        )
                    else:
                        logger.warning(f"No embeddings returned for face {i}")
                        
                except Exception as e:
                    logger.error(f"Error extracting embedding for face {i}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}", exc_info=True)
            return []

    def process_batch(self, frames_list: List[np.ndarray]) -> Dict:
        """
        Main pipeline: Person detection → Face alignment batch → Embedding batch
        
        Args:
            frames_list: List of input frames
            
        Returns:
            Dictionary with results organized by frame_id
        """
        if not frames_list:
            logger.warning("Empty frames list")
            return {}
        
        pipeline_start = time.time()
        logger.info(f"Processing batch of {len(frames_list)} frames")
        
        # Step 1: Person detection
        try:
            logger.info("Starting person detection")
            det_start = time.time()
            
            results = self.person_det_ensemble.predict(
                frames_list,
                self.person_threshold,
                verbose=False
            )
            if not isinstance(results, list):
                results = [results]
            
            logger.info(f"Person detection completed in {time.time() - det_start:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}", exc_info=True)
            return {}
        
        # Step 2: Crop person images
        try:
            batch_person_crops, frame_ids = self._crop_person_images_batch(
                frames_list, results
            )
            total_people = sum(len(crops) for crops in batch_person_crops)
            logger.info(f"Cropped {total_people} people from {len(frames_list)} frames")
        except Exception as e:
            logger.error(f"Error cropping person images: {e}", exc_info=True)
            return {}
        
        # Step 3: Collect all faces for batch alignment
        all_faces_for_alignment = []
        
        for img_idx, person_crops in enumerate(batch_person_crops):
            current_frame_id = str(frame_ids[img_idx])
            
            for person_idx, person_img in enumerate(person_crops):
                if not validate_image(person_img, f"frame_{img_idx}_person_{person_idx}"):
                    continue
                
                all_faces_for_alignment.append(
                    FaceInfo(
                        frame_id=current_frame_id,
                        person_idx=person_idx,
                        img=person_img
                    )
                )
        
        logger.info(f"Total faces to process: {len(all_faces_for_alignment)}")
        
        if not all_faces_for_alignment:
            logger.warning("No valid faces to process")
            return {}
        
        # Step 4: BATCH ALIGNMENT
        try:
            aligned_faces = self._batch_crop_and_align_faces(all_faces_for_alignment)
            if not aligned_faces:
                logger.warning("No faces successfully aligned")
                return {}
        except Exception as e:
            logger.error(f"Error in batch alignment: {e}", exc_info=True)
            return {}
        
        # Step 5: BATCH EMBEDDING
        try:
            embedding_results = self._batch_get_embeddings(aligned_faces)
            if not embedding_results:
                logger.warning("No embeddings successfully extracted")
                return {}
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}", exc_info=True)
            return {}
        
        # Step 6: Organize results by frame
        batch_output = {}
        for result in embedding_results:
            if result.frame_id not in batch_output:
                batch_output[result.frame_id] = []
            
            batch_output[result.frame_id].append({
                "person_idx": result.person_idx,
                "embeddings": result.embeddings
            })
        
        total_time = time.time() - pipeline_start
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        logger.info(f"Processed {len(embedding_results)} faces with embeddings")
        
        return batch_output


def load_images_from_paths(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load images from local files and URLs
    
    Args:
        image_paths: List of file paths or URLs
        
    Returns:
        List of loaded images
    """
    batch_frames = []
    session = create_session_with_retries()
    
    logger.info(f"Loading {len(image_paths)} images")
    
    for path in image_paths:
        if not path:
            continue
        
        img = None
        
        if path.startswith("http://") or path.startswith("https://"):
            logger.info(f"Loading from URL: {path}")
            img = load_image_from_url(path, session)
        else:
            logger.info(f"Loading from file: {path}")
            img = load_image_from_file(path)
        
        if img is not None:
            batch_frames.append(img)
    
    logger.info(f"Successfully loaded {len(batch_frames)}/{len(image_paths)} images")
    return batch_frames


def save_results_with_metadata(
    results: Dict,
    output_path: str = "face_features_output.json",
    triton_host: str = "localhost:9187",
    person_threshold: float = 0.45
) -> bool:
    """
    Save results to JSON with metadata
    
    Args:
        results: Pipeline results
        output_path: Output file path
        triton_host: Triton server host
        person_threshold: Person detection threshold
        
    Returns:
        Success status
    """
    try:
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "triton_host": triton_host,
                "person_threshold": person_threshold,
                "total_frames": len(results),
                "total_faces": sum(len(v) for v in results.values()),
                "version": "2.0 - Optimized Batch Pipeline"
            },
            "results": results
        }
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✓ Results saved to: {output_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {output['metadata']['total_frames']}")
        logger.info(f"Total faces detected: {output['metadata']['total_faces']}")
        logger.info(f"Timestamp: {output['metadata']['timestamp']}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)
        return False


# ==================== MAIN ====================
if __name__ == "__main__":
    # Configuration
    TRITON_HOST = "localhost:9187"
    PERSON_THRESHOLD = 0.45
    OUTPUT_PATH = "face_features_output.json"
    
    img_paths = [
        # Local files
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000910.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_133735_21052026_20260521_133652.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05202026/14581214_165159_20052026_IMG_20260520_165031.jpg",
        # URL
        "http://107.120.93.24:9122/employee-faces/faces/26507931/26507931_1.jpg",
        "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/frames_test4/frame_000970.jpg",
        "/mnt/data/tuyenmb/datasets/data_cctv_face/sev/Danh sách nhân viên có quyền vào MM Comp1 - 2F/File 1/05212026/14581214_135524_21052026_20260521_135452.jpg"
    ]
    
    try:
        # Load images
        batch_frames = load_images_from_paths(img_paths)
        
        if not batch_frames:
            logger.error("No images loaded. Exiting.")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = BatchFacePipeline(
            triton_host=TRITON_HOST,
            person_threshold=PERSON_THRESHOLD,
            max_batch_alignment=16,
            max_batch_embedding=32
        )
        
        # Process batch
        all_batch_results = pipeline.process_batch(batch_frames)
        
        # Save results
        save_results_with_metadata(
            all_batch_results,
            OUTPUT_PATH,
            TRITON_HOST,
            PERSON_THRESHOLD
        )
        
        print("\n✓ Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
