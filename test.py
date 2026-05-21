import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class OptimizedFaceAligner(nn.Module):
    """
    Tối ưu PyTorch Face Alignment:
    - Đơn giản hóa code
    - Mixed precision support (float16)
    - Batch processing optimization
    - Input validation
    - Memory efficient
    """
    
    def __init__(self, use_fp16: bool = False):
        super().__init__()
        self.use_fp16 = use_fp16
        
        # ArcFace standard landmarks (112x112)
        dst_112 = torch.tensor([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=torch.float32)
        
        self.register_buffer("dst_112", dst_112)
        self.register_buffer("dst_224", dst_112 * 2.0)
        
        # Precompute normalization parameters
        self.register_buffer("mean_127", torch.full((1, 3, 1, 1), 127.5))
        self.register_buffer("mean_0", torch.zeros((1, 3, 1, 1)))
        self.register_buffer("std_128", torch.full((1, 3, 1, 1), 128.0))
        self.register_buffer("std_255", torch.full((1, 3, 1, 1), 255.0))

    def _umeyama_transform(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        Tính Similarity Transform (Umeyama algorithm)
        Args:
            src: [N, 5, 2] - Source landmarks
            dst: [5, 2] - Destination landmarks
        Returns:
            M: [N, 2, 3] - Affine transformation matrix
        """
        N = src.shape[0]
        
        # Normalize về tâm
        src_mean = src.mean(dim=1, keepdim=True)  # [N, 1, 2]
        dst_mean = dst.mean(dim=0, keepdim=True)  # [1, 2]
        
        src_c = src - src_mean  # [N, 5, 2]
        dst_c = dst - dst_mean  # [5, 2]
        
        # Tính covariance
        sx, sy = src_c[..., 0], src_c[..., 1]  # [N, 5]
        dx, dy = dst_c[:, 0], dst_c[:, 1]      # [5]
        
        # Umeyama: S1 = <src, dst>, S2 = cross product, S3 = norm
        S1 = torch.sum(sx * dx + sy * dy, dim=1)      # [N]
        S2 = torch.sum(sx * dy - sy * dx, dim=1)      # [N]
        S3 = torch.sum(sx**2 + sy**2, dim=1) + 1e-8   # [N] (eps for stability)
        
        # Scale + Rotation
        a = S1 / S3  # [N]
        b = S2 / S3  # [N]
        
        # Build 2x3 transformation matrix
        M = torch.zeros((N, 2, 3), dtype=src.dtype, device=src.device)
        M[:, 0, 0] = a
        M[:, 0, 1] = -b
        M[:, 1, 0] = b
        M[:, 1, 1] = a
        
        # Translation: t = dst_mean - M_2x2 @ src_mean
        src_mean_2d = src_mean.squeeze(1)  # [N, 2]
        translation = dst_mean - torch.bmm(M[:, :, :2], src_mean_2d.unsqueeze(-1)).squeeze(-1)
        M[:, 0, 2] = translation[:, 0]
        M[:, 1, 2] = translation[:, 1]
        
        return M

    def _coordinate_transform(self, M: torch.Tensor, H: int, W: int, out_size: int) -> torch.Tensor:
        """
        Chuyển đổi ma trận từ pixel space sang normalized space [-1, 1]
        
        PyTorch affine_grid cần: grid_out = M_norm @ pixel_in
        OpenCV warpAffine có: pixel_out = M_pixel @ grid_in
        
        Công thức chuyển đổi:
            M_norm = T_out^-1 @ M_pixel @ T_in
            
        Với: T_out = [2/out_size, 0, -1]
             T_in = [W/2, 0, W/2; 0, H/2, H/2]
        """
        N = M.shape[0]
        
        # Simplified coordinate transformation (derived analytically)
        M_norm = torch.zeros_like(M)
        
        # Rotation/Scale components (with aspect ratio)
        M_norm[:, 0, 0] = M[:, 0, 0] * 2.0 / out_size
        M_norm[:, 0, 1] = M[:, 0, 1] * 2.0 * H / (W * out_size)
        M_norm[:, 1, 0] = M[:, 1, 0] * 2.0 * W / (H * out_size)
        M_norm[:, 1, 1] = M[:, 1, 1] * 2.0 / out_size
        
        # Translation components
        M_norm[:, 0, 2] = (2.0 * M[:, 0, 2] - W) / W + (M[:, 0, 0] + M[:, 0, 1] - 1.0) * 2.0 / out_size
        M_norm[:, 1, 2] = (2.0 * M[:, 1, 2] - H) / H + (M[:, 1, 0] + M[:, 1, 1] - 1.0) * 2.0 / out_size
        
        return M_norm

    def _warp_faces(
        self, 
        img: torch.Tensor, 
        M: torch.Tensor, 
        out_size: int
    ) -> torch.Tensor:
        """
        Áp dụng affine warp + normalize
        Args:
            img: [N, 3, H, W] - Input image NCHW
            M: [N, 2, 3] - Affine matrices
            out_size: 112 or 224
        Returns:
            Warped faces [N, 3, out_size, out_size]
        """
        N = M.shape[0]
        _, _, H, W = img.shape
        
        # Transform matrix to normalized space
        M_norm = self._coordinate_transform(M, H, W, out_size)
        
        # Create grid and sample
        grid = F.affine_grid(M_norm, (N, 3, out_size, out_size), align_corners=False)
        faces = F.grid_sample(img, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
        
        return faces

    def _normalize(
        self, 
        img: torch.Tensor, 
        mean: torch.Tensor, 
        std: torch.Tensor
    ) -> torch.Tensor:
        """Normalize image: (img - mean) / std"""
        return (img - mean) / std

    def forward(
        self, 
        person_image: torch.Tensor,
        landmarks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Args:
            person_image: [1, H, W, 3] NHWC UINT8
            landmarks: [N, 5, 2] Float32
        Returns:
            face_112_nchw: [N, 3, 112, 112] NCHW
            face_224_nchw: [N, 3, 224, 224] NCHW
            face_112_nhwc: [N, 112, 112, 3] NHWC
        """
        # Input validation
        assert person_image.shape[0] == 1, "Batch size must be 1 for image"
        assert landmarks.shape[1] == 5 and landmarks.shape[2] == 2, "Landmarks must be [N, 5, 2]"
        
        N_faces = landmarks.shape[0]
        
        # Convert to float32 and NCHW format
        img_float = person_image.to(torch.float32)  # [1, H, W, 3]
        img_nchw = img_float.permute(0, 3, 1, 2)    # [1, 3, H, W]
        
        # Use mixed precision if enabled
        if self.use_fp16:
            img_nchw = img_nchw.half()
            landmarks = landmarks.half()
        
        # Compute affine matrices
        M112 = self._umeyama_transform(landmarks, self.dst_112)
        M224 = self._umeyama_transform(landmarks, self.dst_224)
        
        # Expand image for batch
        img_batch = img_nchw.expand(N_faces, -1, -1, -1)
        
        # Warp faces
        face_112 = self._warp_faces(img_batch, M112, 112)
        face_224 = self._warp_faces(img_batch, M224, 224)
        face_112_dup = self._warp_faces(img_batch, M112, 112)
        
        # Normalize
        face_112 = self._normalize(face_112, self.mean_127, self.std_128)
        face_224 = self._normalize(face_224, self.mean_127, self.std_128)
        face_nhwc = self._normalize(face_112_dup, self.mean_0, self.std_255)
        
        # Convert NHWC
        face_nhwc = face_nhwc.permute(0, 2, 3, 1)
        
        return face_112, face_224, face_nhwc


# ==================== TEST & EXPORT ====================

if __name__ == "__main__":
    # Initialize model
    model = OptimizedFaceAligner(use_fp16=False).cuda()
    model.eval()
    
    # Test data
    img = torch.randint(0, 255, (1, 480, 640, 3), dtype=torch.uint8).cuda()
    lmks = torch.randn(2, 5, 2).cuda()
    
    print("Testing optimized model...")
    with torch.no_grad():
        face_112, face_224, face_nhwc = model(img, lmks)
    
    print(f"✓ face_112 shape: {face_112.shape}")    # [2, 3, 112, 112]
    print(f"✓ face_224 shape: {face_224.shape}")    # [2, 3, 224, 224]
    print(f"✓ face_nhwc shape: {face_nhwc.shape}")  # [2, 112, 112, 3]
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model, 
        (img, lmks), 
        "face_alignment_optimized.onnx",
        input_names=["person_image", "landmarks"],
        output_names=["face_aligned_112", "face_aligned_224", "face_aligned_nhwc"],
        dynamic_axes={
            "person_image": {1: "H", 2: "W"},
            "landmarks": {0: "num_faces"},
            "face_aligned_112": {0: "num_faces"},
            "face_aligned_224": {0: "num_faces"},
            "face_aligned_nhwc": {0: "num_faces"}
        },
        opset_version=19,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    print("✓ ONNX export successful: face_alignment_optimized.onnx")


# --- EXPORT ---
model = OptimizedFaceAligner(use_fp16=True).cuda()  # FP16 mode
model.eval()
img = torch.randint(0, 255, (1, 480, 640, 3), dtype=torch.uint8).cuda()
lmks = torch.randn(2, 5, 2).cuda()

# Inference
with torch.no_grad():
    face_112, face_224, face_nhwc = model(img, lmks)

# Output
print(face_112.shape)    # [2, 3, 112, 112]
print(face_224.shape)    # [2, 3, 224, 224]
print(face_nhwc.shape)   # [2, 112, 112, 3]


torch.onnx.export(
    model, (img, lmks), "face_extraction_preprocess.onnx",
    input_names=["person_image", "landmarks"],
    output_names=["face_aligned_112", "face_aligned_224", "face_aligned_nhwc"],
    dynamic_axes={
        "person_image": {0: "batch", 1: "H", 2: "W"},
        "landmarks": {0: "num_faces"}, # Ép chiều 0 phải là động
        "face_aligned_112": {0: "num_faces"},
        "face_aligned_224": {0: "num_faces"},
        "face_aligned_nhwc": {0: "num_faces"}
    },
    opset_version=19,
    do_constant_folding=True,
    export_params=True,
)
