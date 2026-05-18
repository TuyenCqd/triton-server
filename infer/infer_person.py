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

                # Ép kích thước batch, ngoại trừ NUM_DETECTIONS thường trả về shape tĩnh
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

        # Chỉ đòi đúng 3 outputs (Bỏ NUM_DETECTIONS đi)
        self.meta_outputs = [
            ("BOXES", "FP32"),
            ("SCORES", "FP32"),
            ("CLASSES", "INT32")
        ]

    def preprocess(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        img_input = np.array(img_chw, dtype=np.uint8)
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def postprocess(self, batch_result, threshold=0.15, nms_threshold=0.45):
        """Hàm thực hiện NMS tại Client"""
        # Server trả về đủ 300 boxes
        boxes = batch_result.get("BOXES", np.array([]))
        scores = batch_result.get("SCORES", np.array([]))
        classes = batch_result.get("CLASSES", np.array([]))

        # 1. Lọc Nhanh (Thresholding) để vứt bỏ các hộp "Nền" (Background)
        valid_idx = np.where(scores >= threshold)[0]

        if len(valid_idx) == 0:
             return {"BOXES": [], "SCORES": [], "CLASSES": []}

        valid_boxes = boxes[valid_idx]
        valid_scores = scores[valid_idx]
        valid_classes = classes[valid_idx]

        # 2. Đẩy vào NMS của OpenCV (OpenCV cần input list dạng [x, y, w, h])
        cv2_boxes = valid_boxes.tolist()
        cv2_scores = valid_scores.tolist()

        keep_indices = cv2.dnn.NMSBoxes(cv2_boxes, cv2_scores, score_threshold=threshold, nms_threshold=nms_threshold)

        # 3. Gom kết quả cuối cùng và chuyển lại sang [x1, y1, x2, y2] để dễ vẽ
        final_boxes, final_scores, final_classes = [], [], []

        if len(keep_indices) > 0:
            keep_indices = keep_indices.flatten()
            for i in keep_indices:
                x, y, w, h = valid_boxes[i]
                # Trả về [x1, y1, x2, y2]
                final_boxes.append([int(x), int(y), int(x + w), int(y + h)])
                final_scores.append(valid_scores[i])
                final_classes.append(valid_classes[i])

        return {
            "BOXES": final_boxes,
            "SCORES": final_scores,
            "CLASSES": final_classes
        }

    def predict(self, frame, threshold=0.45, verbose=False):
        img_blob = self.preprocess(frame)

        batch_result = self.run(
            [img_blob],
            meta_inputs=self.meta_inputs,
            meta_outputs=self.meta_outputs,
            verbose=verbose
        )

        # Truyền threshold vào postprocess để Client tự do tinh chỉnh
        final_result = self.postprocess(batch_result, threshold=threshold, nms_threshold=0.45)

        return final_result


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

    parser = argparse.ArgumentParser(description="Person Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--host", type=str, default="localhost:9187", help="Triton server host")
    parser.add_argument("--model", type=str, default="person_detection_ensemble", help="Model name")
    parser.add_argument("--output", type=str, default="result_person_detection.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    try:
        model_ensemble = PersonDetectionClient(
            triton_host=args.host,
            triton_model_name=args.model,
            max_batch_size=args.batch_size,
            shared_memory=False,
            shared_cuda_memory=False
        )

        frame = load_image(args.image)
        logger.info(f"Image shape: {frame.shape}")

        results = model_ensemble.predict(frame, args.threshold ,verbose=True)
        print("Detection Results:", results)
        # print(f"Boxes: {results['BOXES']}")
        # scores = results['CORE_LABELS']
        # confidence_scores_pct = np.round(sigmoid(scores) * 100, 2)
        # print(f"Scores: {confidence_scores_pct}")

        # print(f"Classes: {results['CLASSES']}")
        visualize_detections(frame, results, output_path=args.output)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
