import tritonclient.grpc as grpcclient
import sys
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import sys
import time
import cv2
import numpy as np
import glog as logger
import time
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from itertools import combinations

import random
from typing import List, Tuple, Dict, Optional
from skimage import transform as trans

# # Lấy đường dẫn tuyệt đối của thư mục chứa code
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# from config.const import MILVUS_URI, DATABASE_NAME, TABLE_NAME
# from processor.vector_search_milvus import VectorSearch


class TritonBaseClient:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = 'localhost:8001', # default gRPC port
                triton_model_name = '',
                connection = 'GRPC',
                verbose = False,
                ssl = False,
                root_certificates = None,
                private_key = None,
                certificate_chain = None,
                max_batch_size = 1, 
                shared_memory = False,
                shared_cuda_memory = False):
        
        assert connection in ['GRPC', 'HTTP'], "Current support only connection type GRPC or HTTP"
        logger.info('Init connection from Triton-inference-server')
        logger.info('- Host: {}'.format(triton_host))
        logger.info('- Model: {}'.format(triton_model_name))
        logger.info('- Connection: {}'.format(connection))
        logger.info('- Shared memory: {}'.format(shared_memory))

        self.triton_host = triton_host
        self.triton_model_name = triton_model_name
        self.connection = connection
        if self.connection == 'GRPC':
            self.model = grpcclient.InferenceServerClient(url = self.triton_host,
                                                        verbose = verbose,
                                                        ssl = ssl,
                                                        root_certificates = root_certificates,
                                                        private_key = private_key,
                                                        certificate_chain = certificate_chain)
        else:
            self.model = httpclient.InferenceServerClient(url = self.triton_host)
        if not self.model.is_server_live():
            logger.info("[ERROR] Server not found: {}".format(self.triton_host))
            sys.exit(1)
        
        if not self.model.is_model_ready(self.triton_model_name):
            logger.info("[ERROR] Model not ready: {}".format(self.triton_model_name))
            sys.exit(1)
        
        self.max_batch_size = max_batch_size
        self.shared_memory = shared_memory
        self.shared_cuda_memory = shared_cuda_memory

    def preprocess(self, imgs):
        """
            Preprocess image
            Input: List of image
            Output: Batch image normalization
        """
        pass
    
    def postprocess(self, batch_result):
        pass
        
    def run(self, batch_data, meta_inputs, meta_outputs, verbose = False):
        
        if verbose:
            tik = time.time()

        # Nếu batch_data là list (nhiều input), lấy len của phần tử đầu tiên
        # Nếu là numpy array (1 input), lấy len của chính nó
        if isinstance(batch_data, list):
            total_images = len(batch_data[0]) 
        else:
            total_images = len(batch_data)

        # Tính toán số lượng batch dựa trên số lượng ảnh thực tế
        total_batchs = int(total_images/self.max_batch_size) if total_images % self.max_batch_size == 0 else int(total_images/self.max_batch_size) + 1
        batch_results = []
        
        for ib in range(total_batchs):
            inputs = []
            outputs = []
            lower = ib * self.max_batch_size
            higher = min((ib+1)*self.max_batch_size, total_images)
            # if verbose:
            #     logger.info(' --> Infer batch {} from data range {}-{}'.format(ib, lower, higher))
            if isinstance(batch_data, list) and len(batch_data) == len(meta_inputs):
                # Trường hợp Ensemble có nhiều input (INPUT_IMAGE, INPUT_LANDMARKS)
                data = batch_data
            else:
                # Trường hợp Model cũ chỉ có 1 input (như Face Detection)
                data = [batch_data[lower:higher]]
            if self.connection == 'GRPC':
                for ix, input_tuple in enumerate(meta_inputs):
                    inputs.append(grpcclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
                    inputs[ix].set_data_from_numpy(data[ix])
                    
                for ix, output_tuple in enumerate(meta_outputs):
                    outputs.append(grpcclient.InferRequestedOutput(output_tuple[0]))
            else:
                for ix, input_tuple in enumerate(meta_inputs):
                    inputs.append(httpclient.InferInput(input_tuple[0], data[ix].shape, input_tuple[1])) # <name, shape, dtype>
                    inputs[ix].set_data_from_numpy(data[ix])

                for ix, output_tuple in enumerate(meta_outputs):
                    outputs.append(httpclient.InferRequestedOutput(output_tuple[0]))

            results = self.model.infer(
                model_name=self.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=None)
                
            results_dict = {}
            for ix, output_tuple in enumerate(meta_outputs):
                output_np = results.as_numpy(output_tuple[0])
                results_dict[output_tuple[0]] = output_np.copy()

            for i in range(higher - lower):
                result_per_image = {}
                for ix, output_tuple in enumerate(meta_outputs):
                    output_name = output_tuple[0]
                    data_from_server = results_dict[output_name]

                    if data_from_server.size > 0:
                        result_per_image[output_name] = data_from_server[i]
                    else:
                        # Trả về mảng rỗng thay vì truy cập index
                        result_per_image[output_name] = np.array([]) 
                    # print('output_tuple: ', output_tuple)
                    # result_per_image[output_tuple[0]] = results_dict[output_tuple[0]][i]
                    # # print('result_per_image: ',result_per_image)
                batch_results.append(result_per_image)

        if verbose:
            time_infer = time.time() - tik
            logger.info(f'[INFO] Inference cost: {int(time_infer * 1000)}ms')
        
        return batch_results
    
class FaceEnsembleClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            ("INPUT_IMAGE", "FP32")
        ]
        
        self.meta_outputs = [
            ("tmp_detections", "INT32", [-1]),
            ("final_boxes", "FP32", [-1, 4]),
            ("final_scores", "FP32", [ -1]),
            ("tmp_classes", "FP32", [-1]),
            ("final_landmarks", "FP32", [-1, 5, 2]),
        ]

    def preprocess(self, frame):
        img_input = np.array(frame, dtype=np.float32)
        # Thêm chiều batch để thành [1, H, W, 3]
        # img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def predict(self, frame, verbose= False):
        # 1. Tiền xử lý tại local (chỉ chuyển kiểu dữ liệu và thêm batch dim)
        img_blob = self.preprocess(frame)
        
        # if verbose:
            # tik = time.time()

        batch_result = self.run(
            [img_blob], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose = verbose
        )

        return batch_result
    
class FaceRegClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            ("INPUT_IMAGE", "FP32")
        ]
        
        self.meta_outputs = [
            ("norm_embeddings", "FP32", [-1, 512]),
        ]

    def preprocess(self, frame):
        img_input = np.array(frame, dtype=np.float32)
        # Thêm chiều batch để thành [1, H, W, 3]
        # img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def predict(self, frame, verbose= False):
        # 1. Tiền xử lý tại local (chỉ chuyển kiểu dữ liệu và thêm batch dim)
        img_blob = self.preprocess(frame)
        
        # if verbose:
            # tik = time.time()

        batch_result = self.run(
            [img_blob], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose = verbose
        )

        return batch_result
    
class FaceExtPreClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            ("person_image", "FP32"),
            ("landmarks", "FP32"),
            ("bboxes", "FP32")
        ]
        
        self.meta_outputs = [
            ("face_aligned_112", "FP32", [-1, 3, 112, 112]),
            ("face_aligned_224", "FP32", [-1, 3, 224, 224]),
            ("face_aligned_nhwc", "FP32", [-1, 112, 112, 3]),
        ]

    def preprocess(self, frame, landmarks, bboxes):
        img_input = np.array(frame, dtype=np.float32)
        landmarks_input = np.expand_dims(landmarks, axis=0)
        bboxes_input = np.expand_dims(bboxes, axis=0)
        # Thêm chiều batch để thành [1, H, W, 3]
        # img_input = np.expand_dims(img_input, axis=0)
        return img_input, landmarks_input, bboxes_input

    def predict(self, frame, landmarks, bboxes, verbose= False):
        # 1. Tiền xử lý tại local (chỉ chuyển kiểu dữ liệu và thêm batch dim)
        img_blob, landmark_input, bboxes_input = self.preprocess(frame, landmarks, bboxes)
        
        # if verbose:
            # tik = time.time()

        batch_result = self.run(
            [img_blob, landmark_input, bboxes_input], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose = verbose
        )

        return batch_result

def crop_and_align_face(person_img_path, save_dir):

    img_bgr = cv2.imread(str(person_img_path))

    if img_bgr is None:
        print(f"Cannot read image: {person_img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    batchs = [img_rgb]

    results = model_ensemble.predict(batchs)
    results = results[0]

    landmarks = results["final_landmarks"]
    bboxes = results["final_boxes"]

    if len(bboxes) == 0:
        print(f"No face detected: {person_img_path}")
        return

    try:
        bbox = bboxes

        x1, y1, x2, y2 = bbox[:4]

        w = x2 - x1
        h = y2 - y1

        mx = int(w * .3)
        my = int(h * .3)

        x1 = int(max(0, x1 - mx))
        y1 = int(max(0, y1 - my))
        x2 = int(min(img_rgb.shape[1], x2 + mx))
        y2 = int(min(img_rgb.shape[0], y2 + my))

        face_crop = img_rgb[y1:y2, x1:x2]


        if face_crop.size == 0:
            print(f"Invalid crop: {person_img_path}")
            return

        white_bg = np.ones_like(img_rgb, dtype=np.uint8) * 255
        white_bg[y1:y2, x1:x2] = face_crop
        batchs_align = [white_bg]

        results_align = align_model.predict(
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

        # RGB -> BGR để save
        aligned_112_bgr = cv2.cvtColor(
            aligned_112,
            cv2.COLOR_RGB2BGR
        )

        return aligned_112_bgr

    except Exception as e:
        print(f"Failed: {person_img_path}")
        print(e)
        print("Landmarks:", landmarks)
        print("BBoxes:", bboxes)

if __name__ == "__main__":
    # from sklearn.metrics.pairwise import cosine_similarity

    model_ensemble = FaceEnsembleClient(triton_host = "localhost:8001",
                                triton_model_name="pipeline_ensemble_Det", 
                                max_batch_size=1,
                                shared_memory = False, 
                                shared_cuda_memory = False)
    
    align_model = FaceExtPreClient(
        triton_host = "localhost:8001",
        triton_model_name="face_alignment", 
        max_batch_size=1,
        shared_memory = False, 
        shared_cuda_memory = False
    )

    recog_model = FaceRegClient(
        triton_host = "localhost:8001",
        triton_model_name="pipeline_reg", 
        max_batch_size=1,
        shared_memory = False, 
        shared_cuda_memory = False
    )

    img_paths = "/app/cctv/frames_test3/frame_002830/frame_002830_1.jpg"
    img = cv2.imread(img_paths)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aligned = crop_and_align_face(
        person_img_path=img_paths,
        save_dir=None
    )

    cv2.imwrite("test.jpg", cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    batch = [aligned]
    results = recog_model.predict(batch)
    results = results[0]
    print(results["norm_embeddings"].shape)
    print(np.linalg.norm(results["norm_embeddings"]))
