import tritonclient.grpc as grpcclient
import sys
import os
import time
import cv2
import numpy as np
import glog as logger

class TritonBaseClient:
    '''
        Sample model-request triton-inference-server with gRPC
    '''
    def __init__(self,
                triton_host = 'localhost:8001', 
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
        logger.info(f'- Host: {triton_host}')
        logger.info(f'- Model: {triton_model_name}')
        logger.info(f'- Connection: {connection}')
        logger.info(f'- Shared memory: {shared_memory}')

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
            from tritonclient import http as httpclient
            self.model = httpclient.InferenceServerClient(url = self.triton_host)
            
        if not self.model.is_server_live():
            logger.error(f"[ERROR] Server not found: {self.triton_host}")
            sys.exit(1)
        
        if not self.model.is_model_ready(self.triton_model_name):
            logger.error(f"[ERROR] Model not ready: {self.triton_model_name}")
            sys.exit(1)
        
        self.max_batch_size = max_batch_size
        self.shared_memory = shared_memory
        self.shared_cuda_memory = shared_cuda_memory

    def preprocess(self, imgs):
        pass
    
    def postprocess(self, batch_result):
        pass
        
    def run(self, batch_data, meta_inputs, meta_outputs, verbose = False):
        if verbose:
            tik = time.time()

        if not isinstance(batch_data, list):
            batch_data = [batch_data]

        total_images = len(batch_data[0]) 
        total_batchs = int(total_images/self.max_batch_size) if total_images % self.max_batch_size == 0 else int(total_images/self.max_batch_size) + 1
        batch_results = []
        
        for ib in range(total_batchs):
            inputs = []
            outputs = []
            lower = ib * self.max_batch_size
            higher = min((ib+1)*self.max_batch_size, total_images)
            
            data = [d[lower:higher] for d in batch_data]
            ClientModule = grpcclient if self.connection == 'GRPC' else httpclient
            
            for ix, input_tuple in enumerate(meta_inputs):
                input_name = input_tuple[0]
                input_type = input_tuple[1]
                infer_in = ClientModule.InferInput(input_name, data[ix].shape, input_type)
                infer_in.set_data_from_numpy(data[ix])
                inputs.append(infer_in)
                
            for ix, output_tuple in enumerate(meta_outputs):
                output_name = output_tuple[0] 
                outputs.append(ClientModule.InferRequestedOutput(output_name))

            results = self.model.infer(
                model_name=self.triton_model_name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=None
            )
            
            result_per_batch = {}
            for ix, output_tuple in enumerate(meta_outputs):
                output_name = output_tuple[0]
                data_from_server = results.as_numpy(output_name)
                
                if len(data_from_server.shape) > 0 and data_from_server.shape[0] == 1 and output_name != "NUM_DETECTIONS":
                    data_from_server = np.squeeze(data_from_server, axis=0)
                    
                result_per_batch[output_name] = data_from_server
            
            batch_results.append(result_per_batch)

        if verbose:
            time_infer = time.time() - tik
            logger.info(f'[INFO] Inference Server cost: {int(time_infer * 1000)}ms')
        
        if total_images <= self.max_batch_size and len(batch_results) == 1:
            return batch_results[0]
            
        return batch_results


class PersonDetectionClient(TritonBaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_inputs = [
            ("IMAGE_IN", "UINT8")
        ]
        
        self.meta_outputs = [
            ("BOXES", "FP32"),
            ("SCORES", "FP32"),
            ("CLASSES", "INT32")
        ]

    def preprocess(self, frame):
        # 1. Chuyển BGR (OpenCV) sang RGB 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Đổi trục từ HWC sang CHW
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        
        # 3. Ép kiểu về uint8
        img_input = np.array(img_chw, dtype=np.uint8)
        
        # 4. Thêm chiều Batch Dimension [1, 3, H, W]
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input

    def postprocess(self, batch_result, conf_threshold=0.5, nms_threshold=0.4):
        """Lọc bỏ background và xử lý chồng chéo bằng NMS (nếu mô hình chưa hội tụ tốt)"""
        boxes = batch_result.get("BOXES", [])
        scores = batch_result.get("SCORES", [])
        classes = batch_result.get("CLASSES", [])

        # 1. Lọc theo Confidence Threshold
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        for box, score, cls_id in zip(boxes, scores, classes):
            if score >= conf_threshold:
                filtered_boxes.append(box.tolist()) # Chuyển đổi sang list cho OpenCV
                filtered_scores.append(float(score))
                filtered_classes.append(int(cls_id))

        if len(filtered_boxes) == 0:
            return {"BOXES": [], "SCORES": [], "CLASSES": []}

        # 2. Áp dụng Non-Maximum Suppression (NMS)
        # cv2.dnn.NMSBoxes yêu cầu format box là [x_min, y_min, width, height]
        # (Dữ liệu của bạn hiện tại đang là [left, top, width, height] -> Rất khớp!)
        indices = cv2.dnn.NMSBoxes(
            bboxes=filtered_boxes, 
            scores=filtered_scores, 
            score_threshold=conf_threshold, 
            nms_threshold=nms_threshold
        )

        # 3. Lấy ra kết quả cuối cùng
        final_boxes = []
        final_scores = []
        final_classes = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(filtered_boxes[i])
                final_scores.append(filtered_scores[i])
                final_classes.append(filtered_classes[i])

        return {
            "BOXES": final_boxes,
            "SCORES": final_scores,
            "CLASSES": final_classes
        }

    def predict(self, frame, threshold=0.5, verbose=False):
        # 1. Preprocess
        img_blob = self.preprocess(frame)
        
        # 2. Gửi request lên Triton Server
        batch_result = self.run(
            [img_blob], 
            meta_inputs=self.meta_inputs, 
            meta_outputs=self.meta_outputs,
            verbose=verbose
        )

        # 3. Postprocess (Lọc 300 boxes thô)
        final_result = self.postprocess(batch_result, threshold=threshold)

        return final_result

if __name__ == "__main__":
    # Lưu ý: Sửa lại tên triton_model_name cho khớp với tên ensemble_model trên server của bạn
    model_ensemble = PersonDetectionClient(
        triton_host="localhost:9187",
        triton_model_name="rtdetr_ensemble", 
        max_batch_size=1,
        shared_memory=False, 
        shared_cuda_memory=False
    )

    img_path = "/mnt/data/tuyenmb/projects/cctv-face-demo/vi-cctv-inference/infer/test_imgs/test.jpg"
    print(f"Đường dẫn ảnh: {img_path}")
    frame = cv2.imread(img_path)
    
    if frame is None:
        logger.error("Failed to load image")
    else:
        # Bạn có thể thay đổi threshold tại đây (ví dụ: 0.15 hoặc 0.5)
        # results = model_ensemble.predict(frame, threshold=0.5, verbose=True)# Gọi Inference
        results = model_ensemble.predict(frame, verbose=True)

        # 1. Lấy số lượng bounding box thực tế (bỏ qua các box đệm số 0)
        # Vì cấu hình Triton trả về mảng 1 chiều, ta lấy phần tử [0]
        num_dets = int(results.get("NUM_DETECTIONS")[0])
        
        # 2. Trích xuất dữ liệu của ảnh đầu tiên (batch index 0) và cắt mảng tới vị trí num_dets
        boxes = results.get("BOXES")[0][:num_dets]
        scores = results.get("SCORES")[0][:num_dets]
        classes = results.get("CLASSES")[0][:num_dets]

        print(f"[INFO] Tìm thấy {num_dets} người vượt qua ngưỡng.")

        # 3. Vẽ hộp giới hạn
        for box, score, cls_id in zip(boxes, scores, classes):
            # Mảng boxes từ Python Backend hiện trả về trực tiếp tọa độ [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            
            # Ép kiểu về số nguyên để vẽ bằng OpenCV
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Vẽ hộp giới hạn (Màu xanh lục, độ dày 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ghi text Score
            label = f"Person: {score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Lưu ảnh kết quả ra file
        output_path = "result_test.jpg"
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Đã lưu ảnh kết quả tại: {output_path}")
        
        # boxes = results.get("BOXES", [])
        # scores = results.get("SCORES", [])
        # classes = results.get("CLASSES", [])

        # print(f"[INFO] Tìm thấy {len(boxes)} người vượt qua ngưỡng.")

        # for box, score, cls_id in zip(boxes, scores, classes):
        #     # Giải mã tọa độ
        #     left, top, w, h = box
        #     x1, y1 = int(left), int(top)
        #     x2, y2 = int(left + w), int(top + h)
            
        #     # Vẽ hộp giới hạn
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        #     # Ghi text Score
        #     label = f"Person: {score:.2f}"
        #     cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # Lưu ảnh kết quả ra file
        # output_path = "result_test.jpg"
        # cv2.imwrite(output_path, frame)
        # print(f"[INFO] Đã lưu ảnh kết quả tại: {output_path}")