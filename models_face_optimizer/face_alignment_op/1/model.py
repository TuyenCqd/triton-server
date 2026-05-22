import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from face_alignment_gpu import FaceAlignmentGPU
from typing import List

class TritonPythonModel:
    """
    PyTorch GPU-accelerated Face Alignment Backend for Triton
    
    This backend implements face alignment using PyTorch GPU kernels,
    providing significantly faster inference than the original Python implementation.
    
    Features:
    - GPU-accelerated processing with PyTorch
    - Batch processing support
    - Multiple output formats (NCHW, NHWC)
    - Automatic device management
    - Performance optimizations (batching, caching)
    """
    
    def initialize(self, args):
        """
        Initialize the model
        
        Args:
            args: Dictionary containing model configuration and runtime parameters
        """
        self.model_config = json.loads(args["model_config"])
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Face Alignment PyTorch] Using device: {self.device}")
        
        if self.device.type == 'cpu':
            print("[WARNING] GPU not available, falling back to CPU (slower)")
        
        # Output specifications
        self.output_specs = {
            "face_aligned_112": {
                "size": 112,
                "mean": 127.5,
                "std": 128.0,
                "is_nchw": True
            },
            "face_aligned_224": {
                "size": 224,
                "mean": 127.5,
                "std": 127.5,
                "is_nchw": True
            },
            "face_aligned_nhwc": {
                "size": 112,
                "mean": 0.0,
                "std": 255.0,
                "is_nchw": False
            }
        }
        
        # Initialize GPU model
        self.model = FaceAlignmentGPU(
            device=str(self.device),
            output_specs=self.output_specs,
            dtype=torch.float32
        )
        self.model.eval()  # Set to evaluation mode
        
        # Compile to TorchScript for better performance (optional)
        self.use_torchscript = False
        try:
            # Try to script the model
            self.scripted_model = torch.jit.script(self.model)
            self.use_torchscript = True
            print("[Face Alignment PyTorch] TorchScript compilation successful")
        except Exception as e:
            print(f"[WARNING] TorchScript compilation failed: {e}")
            print("         Using regular PyTorch model")
        
        print(f"[Face Alignment PyTorch] Model initialized successfully")
    
    def execute(self, requests: List) -> List:
        """
        Execute inference on batch of requests
        
        Args:
            requests: List of Triton InferenceRequest objects
        
        Returns:
            List of Triton InferenceResponse objects
        """
        responses = []
        
        for request in requests:
            try:
                response = self._process_request(request)
                responses.append(response)
            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(f"[ERROR] {error_msg}")
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(error_msg)
                    )
                )
        
        return responses
    
    def _process_request(self, request):
        """
        Process a single inference request
        
        Args:
            request: Triton InferenceRequest object
        
        Returns:
            Triton InferenceResponse object
        """
        # Extract input tensors
        image_input = pb_utils.get_input_tensor_by_name(request, "person_image")
        landmarks_input = pb_utils.get_input_tensor_by_name(request, "landmarks")
        bboxes_input = pb_utils.get_input_tensor_by_name(request, "bboxes")
        
        # Convert to numpy
        img_np = image_input.as_numpy()
        lmk_np = landmarks_input.as_numpy()
        bbox_np = bboxes_input.as_numpy()
        
        # Remove batch dimension if present
        if len(img_np.shape) == 4:
            img_np = np.squeeze(img_np, axis=0)
        
        # Reshape landmarks and bboxes
        batch_size = lmk_np.shape[0]
        lmk_np = lmk_np.reshape(batch_size, 5, 2)
        bbox_np = bbox_np.reshape(batch_size, 4)
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img_np).to(self.device)
        lmk_tensor = torch.from_numpy(lmk_np).float().to(self.device)
        bbox_tensor = torch.from_numpy(bbox_np).float().to(self.device)
        
        # Run inference with no_grad for faster computation
        with torch.no_grad():
            if self.use_torchscript:
                outputs_dict = self.scripted_model(img_tensor, lmk_tensor, bbox_tensor)
            else:
                outputs_dict = self.model(img_tensor, lmk_tensor, bbox_tensor)
        
        # Convert outputs to numpy and prepare response tensors
        output_tensors = []
        
        for output_name in self.output_specs.keys():
            if output_name in outputs_dict:
                output_tensor = outputs_dict[output_name]
                
                # Convert to numpy
                output_np = output_tensor.cpu().numpy()
                
                # Create Triton tensor
                triton_output = pb_utils.Tensor(output_name, output_np)
                output_tensors.append(triton_output)
        
        return pb_utils.InferenceResponse(output_tensors=output_tensors)