import numpy as np
import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        
        for request in requests:
            embeddings_tensor = pb_utils.get_input_tensor_by_name(request, "embeddings")

            if embeddings_tensor is None:
                # Quan trọng: Vẫn phải trả về response rỗng để không làm treo Pipeline Ensemble
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError("Input tensors missing")))
                continue

            embeddings = embeddings_tensor.as_numpy()      
            embeddings = embeddings / np.linalg.norm(embeddings)
            embeddings = np.expand_dims(embeddings, axis=0)

            # 7. Tạo output tensors với đúng tên trong config.pbtxt
            norm_embeddings = pb_utils.Tensor("norm_embeddings", embeddings)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[norm_embeddings]
            )
            responses.append(inference_response)

        return responses


    def finalize(self):
        pass
