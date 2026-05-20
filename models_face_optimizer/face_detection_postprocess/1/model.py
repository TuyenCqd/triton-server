import numpy as np
import json
import triton_python_backend_utils as pb_utils

# class TritonPythonModel:
#     def initialize(self, args):
#         self.model_config = json.loads(args["model_config"])
#         # Bạn có thể lấy ngưỡng threshold từ config nếu muốn
#         self.conf_threshold = 0.15

#     def execute(self, requests):
#         responses = []
        
#         for request in requests:
#             boxes_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_boxes")
#             scores_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_scores")
#             landmarks_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_landmarks")
#             ratio_tensor = pb_utils.get_input_tensor_by_name(request, "ratio_value")

#             if None in [boxes_tensor, scores_tensor, landmarks_tensor, ratio_tensor]:
#                 # Quan trọng: Vẫn phải trả về response rỗng để không làm treo Pipeline Ensemble
#                 responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError("Input tensors missing")))
#                 continue

#             boxes = boxes_tensor.as_numpy()
#             scores = scores_tensor.as_numpy()    
#             landmarks = landmarks_tensor.as_numpy() 
#             ratio = ratio_tensor.as_numpy().flatten()[0]

#             if boxes.ndim == 3:
#                 boxes = boxes[0]
#                 scores = scores[0]
#                 landmarks = landmarks[0]

#             # 3. Lọc theo ngưỡng điểm số
#             mask = scores >= self.conf_threshold
#             f_boxes = boxes[mask]
#             f_scores = scores[mask]
#             f_landmarks = landmarks[mask]

#             # KIỂM TRA NẾU RỖNG SAU KHI MASK
#             if len(f_scores) == 0:
#                 final_boxes = np.empty((0, 4), dtype=np.float32)
#                 final_scores = np.empty((0,), dtype=np.float32)
#                 final_landmarks = np.empty((0, 5, 2), dtype=np.float32)
#             else:
#                 # 4. Scale tọa độ
#                 final_boxes = (f_boxes / ratio).astype(np.float32)
#                 final_scores = f_scores.astype(np.float32)
#                 # 5. Reshape và scale landmarks
#                 final_landmarks = (f_landmarks / ratio).reshape(-1, 5, 2).astype(np.float32)

#             # 7. Tạo output tensors với đúng tên trong config.pbtxt
#             out_boxes = pb_utils.Tensor("final_boxes", final_boxes)
#             out_scores = pb_utils.Tensor("final_scores", final_scores)
#             out_landmarks = pb_utils.Tensor("final_landmarks", final_landmarks)

#             inference_response = pb_utils.InferenceResponse(
#                 output_tensors=[out_boxes, out_scores, out_landmarks]
#             )
#             responses.append(inference_response)

#         return responses


#     def _check_face_size(self, )

#     def finalize(self):
#         pass


import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.conf_threshold = 0.5
        # =====================================================
        # FACE FILTER CONFIG
        # =====================================================
        self.min_face_w = 40
        self.min_face_h = 40

        # yaw heuristic
        # nose lệch quá nhiều khỏi center mắt
        self.max_yaw_score = 0.35

    # =========================================================
    # FACE SIZE CHECK
    # =========================================================
    def _check_face_size(self, box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        ok = (
            w >= self.min_face_w and
            h >= self.min_face_h
        )
        return ok, w, h

    # =========================================================
    # YAW CHECK
    # =========================================================
    def _yaw_score(self, landmarks):
        """
        landmarks:
            0: left eye
            1: right eye
            2: nose
            3: left mouth
            4: right mouth
        """

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # center giữa 2 mắt
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        # khoảng cách 2 mắt
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return 999.0

        # nose lệch khỏi center
        offset = abs(nose[0] - eye_center_x)
        score = offset / eye_dist
        return float(score)

    # =========================================================
    # FILTER FACE
    # =========================================================
    def _is_good_face(self, box, landmarks):
        # ---------------------------------------------
        # size
        # ---------------------------------------------
        ok_size, w, h = self._check_face_size(box)

        if not ok_size: return False

        # ---------------------------------------------
        # yaw
        # ---------------------------------------------
        yaw = self._yaw_score(landmarks)
        if yaw > self.max_yaw_score:
            return False
        return True
    
    # =========================================================
    # EXECUTE
    # =========================================================
    def execute(self, requests):
        responses = []
        for request in requests:
            boxes_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_boxes")
            scores_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_scores")
            landmarks_tensor = pb_utils.get_input_tensor_by_name(request, "tmp_landmarks")
            ratio_tensor = pb_utils.get_input_tensor_by_name(request, "ratio_value")
            # =================================================
            # CHECK INPUT
            # =================================================
            if None in [boxes_tensor, scores_tensor, landmarks_tensor, ratio_tensor]:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "Input tensors missing"
                        )
                    )
                )
                continue
            # =================================================
            # LOAD INPUT
            # =================================================
            boxes = boxes_tensor.as_numpy()
            scores = scores_tensor.as_numpy()
            landmarks = landmarks_tensor.as_numpy()
            ratio = ratio_tensor.as_numpy().flatten()[0]

            # =================================================
            # REMOVE BATCH
            # =================================================
            if boxes.ndim == 3:
                boxes = boxes[0]
                scores = scores[0]
                landmarks = landmarks[0]

            # =================================================
            # SCORE FILTER
            # =================================================
            mask = scores >= self.conf_threshold
            f_boxes = boxes[mask]
            f_scores = scores[mask]
            f_landmarks = landmarks[mask]
            # =================================================
            # EMPTY
            # =================================================
            if len(f_scores) == 0:
                final_boxes = np.empty((0, 4),dtype=np.float32)
                final_scores = np.empty((0,),dtype=np.float32)
                final_landmarks = np.empty((0, 5, 2),dtype=np.float32)
            else:

                # ---------------------------------------------
                # scale
                # ---------------------------------------------
                scaled_boxes = (f_boxes / ratio).astype(np.float32)
                scaled_scores = (f_scores.astype(np.float32))
                scaled_landmarks = (f_landmarks / ratio).reshape(-1, 5, 2).astype(np.float32)

                # ---------------------------------------------
                # custom filter
                # ---------------------------------------------
                keep_boxes = []
                keep_scores = []
                keep_landmarks = []

                for box, score, lmk in zip(scaled_boxes, scaled_scores, scaled_landmarks):
                    # if self._is_good_face(box, lmk):
                    keep_boxes.append(box)
                    keep_scores.append(score)
                    keep_landmarks.append(lmk)
                # ---------------------------------------------
                # to numpy
                # ---------------------------------------------
                if len(keep_boxes) > 0:
                    final_boxes = np.asarray(keep_boxes, dtype=np.float32)
                    final_scores = np.asarray(keep_scores,dtype=np.float32)
                    final_landmarks = np.asarray(keep_landmarks, dtype=np.float32)
                else:
                    final_boxes = np.empty((0, 4), dtype=np.float32)
                    final_scores = np.empty((0,),dtype=np.float32)
                    final_landmarks = np.empty((0, 5, 2), dtype=np.float32)

            # =================================================
            # OUTPUT
            # =================================================
            
            out_boxes = pb_utils.Tensor("final_boxes", final_boxes)

            out_scores = pb_utils.Tensor("final_scores",final_scores)
            out_landmarks = pb_utils.Tensor("final_landmarks", final_landmarks)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_boxes,
                    out_scores,
                    out_landmarks
                ]
            )
            responses.append(inference_response)
        return responses