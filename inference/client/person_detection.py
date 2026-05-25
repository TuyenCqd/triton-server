import cv2
import numpy as np
from common.triton_base import TritonBaseClient

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

    def preprocess(self, batch_frames):
        processed_imgs = []
        
        self.model_w = 560
        self.model_h = 560
        
        self.org_sizes = [] 

        for frame in batch_frames:
            h_org, w_org, _ = frame.shape
            self.org_sizes.append((w_org, h_org))
            
            resized_frame = cv2.resize(frame, (self.model_w, self.model_h))
            
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            processed_imgs.append(img_chw)
        
        img_input = np.stack(processed_imgs, axis=0).astype(np.uint8)
        return img_input
    
    def postprocess(self, batch_results, threshold=0.15, nms_threshold=0.45):
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
            
        final_batch_output = []

        for single_result, (w_org, h_org) in zip(batch_results, self.org_sizes):
            boxes = single_result.get("BOXES", np.array([]))
            scores = single_result.get("SCORES", np.array([]))
            classes = single_result.get("CLASSES", np.array([]))

            scale_x = w_org / self.model_w
            scale_y = h_org / self.model_h

            valid_idx = np.where(scores >= threshold)[0]

            if len(valid_idx) == 0:
                final_batch_output.append({"BOXES": [], "SCORES": [], "CLASSES": []})
                continue

            valid_boxes = boxes[valid_idx]
            valid_scores = scores[valid_idx]
            valid_classes = classes[valid_idx]

            scaled_boxes = []
            for box in valid_boxes:
                x, y, w, h = box
                x_real = x * scale_x
                y_real = y * scale_y
                w_real = w * scale_x
                h_real = h * scale_y
                scaled_boxes.append([x_real, y_real, w_real, h_real])

            cv2_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in scaled_boxes]
            cv2_scores = valid_scores.tolist()

            keep_indices = cv2.dnn.NMSBoxes(cv2_boxes, cv2_scores, score_threshold=threshold, nms_threshold=nms_threshold)

            final_boxes, final_scores, final_classes = [], [], []

            if len(keep_indices) > 0:
                keep_indices = keep_indices.flatten()
                for i in keep_indices:
                    x, y, w, h = scaled_boxes[i]
                    final_boxes.append([int(x), int(y), int(x + w), int(y + h)])
                    final_scores.append(valid_scores[i])
                    final_classes.append(valid_classes[i])

            final_batch_output.append({
                "BOXES": final_boxes,
                "SCORES": final_scores,
                "CLASSES": final_classes
            })

        return final_batch_output


    def predict(self, batch_frames, threshold=0.45, verbose=False):
        if not isinstance(batch_frames, list):
            batch_frames = [batch_frames]

        img_blob = self.preprocess(batch_frames)

        batch_result = self.run(
            [img_blob],
            meta_inputs=self.meta_inputs,
            meta_outputs=self.meta_outputs,
            verbose=verbose
        )

        final_result = self.postprocess(batch_result, threshold=threshold, nms_threshold=0.45)

        return final_result
