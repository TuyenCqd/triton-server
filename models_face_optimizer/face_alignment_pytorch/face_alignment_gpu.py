import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import cv2

# ARCFACE standard landmarks
ARCFACE_DST = torch.tensor(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=torch.float32
)


class FaceAlignmentGPU(nn.Module):
    """
    PyTorch GPU-accelerated Face Alignment Module
    
    Performs face alignment using affine transformation and bilinear interpolation
    on GPU with support for batch processing and multiple output formats.
    
    Features:
    - GPU-accelerated ROI cropping
    - PyTorch grid_sample for affine warping
    - Bilinear interpolation with TorchScript support
    - Multi-format output (NCHW, NHWC)
    - Differentiable operations (can be used in training pipeline)
    
    Example:
        >>> model = FaceAlignmentGPU(device='cuda')
        >>> image = torch.randn(1, 1080, 1920, 3, dtype=torch.uint8).cuda()
        >>> landmarks = torch.randn(1, 5, 2).cuda()
        >>> bboxes = torch.tensor([[100, 100, 600, 700]], dtype=torch.float32).cuda()
        >>> output = model(image, landmarks, bboxes)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        output_specs: Optional[Dict] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize Face Alignment GPU module
        
        Args:
            device: CUDA device ('cuda', 'cuda:0', etc.)
            output_specs: Dictionary of output configurations
                Example: {
                    'face_aligned_112': {'size': 112, 'mean': 127.5, 'std': 128.0, 'is_nchw': True},
                    'face_aligned_224': {'size': 224, 'mean': 127.5, 'std': 127.5, 'is_nchw': True},
                    'face_aligned_nhwc': {'size': 112, 'mean': 0.0, 'std': 255.0, 'is_nchw': False},
                }
            dtype: Floating point dtype for computation
        """
        super(FaceAlignmentGPU, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.margin = 0.3
        
        # Default output specifications matching original config
        if output_specs is None:
            output_specs = {
                'face_aligned_112': {
                    'size': 112, 'mean': 127.5, 'std': 128.0, 'is_nchw': True
                },
                'face_aligned_224': {
                    'size': 224, 'mean': 127.5, 'std': 127.5, 'is_nchw': True
                },
                'face_aligned_nhwc': {
                    'size': 112, 'mean': 0.0, 'std': 255.0, 'is_nchw': False
                },
            }
        
        self.output_specs = output_specs
        
        # Register ARCFACE_DST as buffer (GPU-accessible constant)
        self.register_buffer('arcface_dst', ARCFACE_DST.to(device))
    
    def forward(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        bboxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for face alignment
        
        Args:
            image: Input image tensor
                Shape: (B, H, W, 3) or (H, W, 3)
                Format: uint8 or float32, BGR
                
            landmarks: Facial landmarks tensor
                Shape: (B, 5, 2) or (5, 2)
                Format: float32, pixel coordinates
                
            bboxes: Bounding boxes tensor
                Shape: (B, 4) or (4,)
                Format: float32, [x1, y1, x2, y2]
        
        Returns:
            Dictionary of aligned face tensors:
            {
                'face_aligned_112': (B, 3, 112, 112) or (B, 112, 112, 3),
                'face_aligned_224': (B, 3, 224, 224),
                'face_aligned_nhwc': (B, 112, 112, 3),
            }
        """
        # Ensure tensors are on correct device
        image = image.to(self.device)
        landmarks = landmarks.to(self.device)
        bboxes = bboxes.to(self.device)
        
        # Handle unbatched inputs
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if landmarks.dim() == 2:
            landmarks = landmarks.unsqueeze(0)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)
        
        batch_size = image.shape[0]
        img_height, img_width = image.shape[1:3]
        
        # Convert image to float if needed
        if image.dtype == torch.uint8:
            image = image.float()
        
        # Process each output format
        outputs = {}
        for output_name, spec in self.output_specs.items():
            aligned_faces = []
            
            for b in range(batch_size):
                # Align single face
                aligned = self._align_single_face(
                    image[b],
                    landmarks[b],
                    bboxes[b],
                    img_height,
                    img_width,
                    spec
                )
                aligned_faces.append(aligned)
            
            # Stack batch
            if aligned_faces[0] is not None:
                stacked = torch.stack(aligned_faces, dim=0)
                outputs[output_name] = stacked
            else:
                # Return empty tensor with correct shape
                if spec['is_nchw']:
                    shape = (batch_size, 3, spec['size'], spec['size'])
                else:
                    shape = (batch_size, spec['size'], spec['size'], 3)
                outputs[output_name] = torch.empty(shape, dtype=self.dtype, device=self.device)
        
        return outputs
    
    def _align_single_face(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        bbox: torch.Tensor,
        img_height: int,
        img_width: int,
        spec: Dict
    ) -> Optional[torch.Tensor]:
        """
        Align a single face image
        
        Args:
            image: Single image tensor (H, W, 3)
            landmarks: Single landmarks tensor (5, 2)
            bbox: Single bbox tensor (4,)
            img_height, img_width: Image dimensions
            spec: Output specification
        
        Returns:
            Aligned face tensor or None if alignment fails
        """
        # Compute ROI with margin
        x1, y1, x2, y2 = bbox.long().tolist()
        w = x2 - x1
        h = y2 - y1
        
        mx = int(w * self.margin)
        my = int(h * self.margin)
        
        x1_crop = max(0, x1 - mx)
        y1_crop = max(0, y1 - my)
        x2_crop = min(img_width - 1, x2 + mx)
        y2_crop = min(img_height - 1, y2 + my)
        
        roi_height = y2_crop - y1_crop
        roi_width = x2_crop - x1_crop
        
        if roi_height <= 0 or roi_width <= 0:
            return None
        
        # Crop ROI
        roi = image[y1_crop:y2_crop, x1_crop:x2_crop, :].clone()
        
        # Adjust landmarks to ROI local coordinates
        landmarks_local = landmarks.clone()
        landmarks_local[:, 0] -= x1_crop
        landmarks_local[:, 1] -= y1_crop
        
        # Compute affine transformation matrix
        M = self._estimate_affine_transform(landmarks_local, self.arcface_dst, spec['size'])
        
        if M is None:
            return None
        
        # Apply affine transformation using grid_sample
        aligned = self._warp_affine(roi, M, spec['size'])
        
        # Normalize
        aligned = (aligned - spec['mean']) / spec['std']
        
        # Format output
        if spec['is_nchw']:
            # NCHW: (3, H, W)
            aligned = aligned.permute(2, 0, 1)
        # else: NHWC already (H, W, 3)
        
        return aligned
    
    def _estimate_affine_transform(
        self,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
        output_size: int
    ) -> Optional[torch.Tensor]:
        """
        Estimate 2D affine transformation using least squares
        
        Solves: A @ x = b, where x is the transformation matrix
        
        Args:
            src_points: Source landmarks (5, 2)
            dst_points: Target landmarks (5, 2)
            output_size: Output image size
        
        Returns:
            Affine transformation matrix (2, 3) or None if estimation fails
        """
        # Scale target landmarks to output size
        scale_ratio = output_size / 112.0
        dst_scaled = dst_points * scale_ratio
        
        # Build normal equations: A^T @ A @ x = A^T @ b
        # For each point: [x_i, y_i, 1] @ [a, b, c]^T = u_i
        
        batch_size = src_points.shape[0]
        
        # Construct design matrix A (batch_size * 2, 6)
        A = torch.zeros((batch_size * 2, 6), dtype=self.dtype, device=self.device)
        b = torch.zeros((batch_size * 2, 1), dtype=self.dtype, device=self.device)
        
        for i in range(batch_size):
            x, y = src_points[i]
            u, v = dst_scaled[i]
            
            # Row for u-coordinate: [x, y, 1, 0, 0, 0]
            A[2*i, 0] = x
            A[2*i, 1] = y
            A[2*i, 2] = 1.0
            b[2*i, 0] = u
            
            # Row for v-coordinate: [0, 0, 0, x, y, 1]
            A[2*i+1, 3] = x
            A[2*i+1, 4] = y
            A[2*i+1, 5] = 1.0
            b[2*i+1, 0] = v
        
        # Solve using least squares: x = (A^T @ A)^-1 @ A^T @ b
        try:
            # PyTorch's lstsq is more stable
            solution = torch.linalg.lstsq(A, b).solution
            
            # Reshape to 2x3 matrix
            M = solution.squeeze(-1).reshape(2, 3)
            
            return M
        except Exception as e:
            print(f"Failed to estimate affine transform: {e}")
            return None
    
    def _warp_affine(
        self,
        src_image: torch.Tensor,
        M: torch.Tensor,
        output_size: int
    ) -> torch.Tensor:
        """
        Apply affine transformation using PyTorch's grid_sample
        
        Args:
            src_image: Source image (H, W, 3)
            M: Affine transformation matrix (2, 3)
            output_size: Output image size
        
        Returns:
            Warped image (output_size, output_size, 3)
        """
        # Create grid for output image
        grid = F.affine_grid(
            M.unsqueeze(0),  # (1, 2, 3)
            size=(1, 3, output_size, output_size),
            align_corners=True
        )
        
        # Prepare source image: (1, 3, H, W) - channels first for grid_sample
        src_img_chw = src_image.permute(2, 0, 1).unsqueeze(0)
        
        # Apply grid_sample (bilinear interpolation)
        warped = F.grid_sample(
            src_img_chw,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Convert back to HWC format
        warped = warped.squeeze(0).permute(1, 2, 0)
        
        # Convert BGR to RGB
        warped = warped[:, :, [2, 1, 0]]
        
        return warped
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.device = device
        return self


class FaceAlignmentTorchScript(nn.Module):
    """
    TorchScript-compatible version for deployment
    """
    
    def __init__(self, model: FaceAlignmentGPU):
        super().__init__()
        self.model = model
    
    def forward(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        bboxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.model(image, landmarks, bboxes)
