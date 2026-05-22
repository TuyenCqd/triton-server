import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Bộ điểm mốc chuẩn ArcFace định dạng (5, 2) trên khung 112x112
ARCFACE_DST = torch.tensor(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=torch.float32
)

class FaceAlignmentGPU(nn.Module):
    """
    PyTorch GPU-accelerated Face Alignment Module (Optimized for Triton Server)
    - Xử lý hoàn toàn bằng ma trận song song (Vectorized Batch)
    - Hỗ trợ TorchScript trọn vẹn (Tương thích 100% với torch.jit.script)
    - Sửa lỗi toán học của hệ tọa độ F.affine_grid (-1 đến 1)
    """
    def __init__(
        self,
        device: str = 'cuda',
        output_specs: Dict[str, Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32
    ):
        super(FaceAlignmentGPU, self).__init__()
        self.device_str = device
        self.dtype = dtype
        self.margin = 0.3
        
        if output_specs is None:
            self.output_specs = {
                'face_aligned_112': {'size': 112.0, 'mean': 127.5, 'std': 128.0, 'is_nchw': 1.0},
                'face_aligned_224': {'size': 224.0, 'mean': 127.5, 'std': 127.5, 'is_nchw': 1.0},
                'face_aligned_nhwc': {'size': 112.0, 'mean': 0.0, 'std': 255.0, 'is_nchw': 0.0},
            }
        else:
            # Ép kiểu dữ liệu sang float để TorchScript biên dịch tĩnh dễ dàng
            sanitized_specs: Dict[str, Dict[str, float]] = {}
            for k, v in output_specs.items():
                sanitized_specs[k] = {
                    'size': float(v['size']),
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'is_nchw': 1.0 if bool(v['is_nchw']) else 0.0
                }
            self.output_specs = sanitized_specs
        
        # Đăng ký buffer điểm mốc chuẩn
        self.register_buffer('arcface_dst', ARCFACE_DST.to(device))

    def forward(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        bboxes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        image: Tensor [H, W, C] từ Triton request (ảnh gốc đơn lẻ)
        landmarks: Tensor [B, 5, 2] (B khuôn mặt phát hiện được trên ảnh đó)
        bboxes: Tensor [B, 4]
        """
        # Đảm bảo dữ liệu nằm trên đúng thiết bị và định dạng
        img_h, img_w, img_c = image.shape
        if image.dtype == torch.uint8:
            image = image.float()
            
        # 1. Tính toán tọa độ crop đồng loạt cho cả Batch (Vectorized ROI)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        
        mx = (w * self.margin).long()
        my = (h * self.margin).long()
        
        x1_crop = torch.clamp(x1.long() - mx, min=0, max=img_w - 1)
        y1_crop = torch.clamp(y1.long() - my, min=0, max=img_h - 1)
        x2_crop = torch.clamp(x2.long() + mx, min=0, max=img_w - 1)
        y2_crop = torch.clamp(y2.long() + my, min=0, max=img_h - 1)

        # 2. Xử lý đa định dạng đầu ra song song hóa qua dict
        outputs: Dict[str, torch.Tensor] = {}
        
        for output_name, spec in self.output_specs.items():
            out_size = int(spec['size'])
            
            # Tính toán ma trận Affine trực tiếp từ ảnh gốc sang ảnh đích cho cả Batch
            # Giải phương trình bình phương tối thiểu song song: (A^T * A)^-1 * A^T * B
            M = self._estimate_affine_batch(landmarks, out_size)
            
            # Chuyển đổi ma trận Affine sang hệ tọa độ chuẩn hóa của PyTorch grid_sample [-1, 1]
            M_norm = self._convert_affine_to_grid_coords(M, img_h, img_w, out_size)
            
            # Biến đổi hình học đồng loạt cho cả Batch (Warp Affine song song)
            # Thêm chiều Batch cho ảnh: [1, H, W, C] -> permute sang NCHW để dùng grid_sample
            img_nchw = image.permute(2, 0, 1).unsqueeze(0) 
            
            # Tạo grid lưới tọa độ từ ma trận chuẩn hóa [B, H_out, W_out, 2]
            grid = F.affine_grid(
                M_norm, 
                size=[landmarks.shape[0], img_c, out_size, out_size], 
                align_corners=True
            )
            
            # Mở rộng ảnh gốc tương ứng với kích thước Batch để gom chung xử lý 1 lần đơn duy nhất
            img_expanded = img_nchw.expand(landmarks.shape[0], -1, -1, -1)
            
            # Ép mẫu nội suy đồng loạt cực nhanh trên GPU
            warped = F.grid_sample(
                img_expanded, 
                grid, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
            ) # Output dạng: [B, C, H_out, W_out]
            
            # Đổi hệ màu từ RGB sang BGR bằng indexing chuẩn xác
            warped = warped[:, [2, 1, 0], :, :]
            
            # Chuẩn hóa ảnh: (X - mean) / std
            warped = (warped - spec['mean']) / spec['std']
            
            # Định hình đầu ra theo cấu hình (NCHW hay NHWC)
            if spec['is_nchw'] == 0.0:
                warped = warped.permute(0, 2, 3, 1) # Chuyển thành [B, H_out, W_out, C]
                
            outputs[output_name] = warped

        return outputs

    def _estimate_affine_batch(self, src_points: torch.Tensor, output_size: int) -> torch.Tensor:
        """ Tính toán ma trận biến đổi Affine song song cho toàn bộ các khuôn mặt trong Batch """
        B = src_points.shape[0]
        scale_ratio = float(output_size) / 112.0
        dst_scaled = self.arcface_dst * scale_ratio
        
        # Khởi tạo ma trận hệ số A kích thước [B, 10, 6] và vector b kích thước [B, 10, 1]
        A = torch.zeros((B, 10, 6), dtype=self.dtype, device=src_points.device)
        b = torch.zeros((B, 10, 1), dtype=self.dtype, device=src_points.device)
        
        for i in range(5):
            A[:, 2*i, 0] = src_points[:, i, 0]
            A[:, 2*i, 1] = src_points[:, i, 1]
            A[:, 2*i, 2] = 1.0
            b[:, 2*i, 0] = dst_scaled[i, 0]
            
            A[:, 2*i+1, 3] = src_points[:, i, 0]
            A[:, 2*i+1, 4] = src_points[:, i, 1]
            A[:, 2*i+1, 5] = 1.0
            b[:, 2*i+1, 0] = dst_scaled[i, 1]
            
        # Giải phương trình bình phương tối thiểu song song qua Batch bằng cấu trúc Vector hóa
        # Thay vì try-except, ta cộng epsilon nhỏ vào đường chéo tránh suy biến Ma trận (an toàn cho TorchScript)
        ATA = torch.bmm(A.transpose(1, 2), A)
        epsilon = 1e-6 * torch.eye(6, dtype=self.dtype, device=src_points.device).unsqueeze(0)
        ATA_inv = torch.linalg.inv(ATA + epsilon)
        ATb = torch.bmm(A.transpose(1, 2), b)
        solution = torch.bmm(ATA_inv, ATb)
        
        M = solution.squeeze(-1).reshape(B, 2, 3)
        return M

    def _convert_affine_to_grid_coords(
        self, M: torch.Tensor, img_h: int, img_w: int, out_size: int
    ) -> torch.Tensor:
        """
        Hàm cốt lõi sửa lỗi toán học: Chuyển ma trận Affine Pixel-to-Pixel chuẩn 
        sang ma trận nghịch đảo tương thích với hệ lưới chuẩn hóa [-1, 1] của F.affine_grid.
        """
        B = M.shape[0]
        
        # Tạo ma trận đồng nhất mở rộng kích thước [B, 3, 3]
        M_homo = torch.eye(3, dtype=self.dtype, device=M.device).unsqueeze(0).repeat(B, 1, 1)
        M_homo[:, :2, :] = M
        
        # Tính ma trận nghịch đảo vì affine_grid yêu cầu ánh xạ ngược (Dest -> Source)
        M_inv = torch.linalg.inv(M_homo)
        
        # Ma trận chuẩn hóa tọa độ ảnh gốc sang [-1, 1]
        T_src = torch.tensor([
            [2.0 / (img_w - 1.0), 0.0, -1.0],
            [0.0, 2.0 / (img_h - 1.0), -1.0],
            [0.0, 0.0, 1.0]
        ], dtype=self.dtype, device=M.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Ma trận chuẩn hóa tọa độ ảnh đích sang [-1, 1]
        T_dst = torch.tensor([
            [2.0 / (out_size - 1.0), 0.0, -1.0],
            [0.0, 2.0 / (out_size - 1.0), -1.0],
            [0.0, 0.0, 1.0]
        ], dtype=self.dtype, device=M.device).unsqueeze(0).repeat(B, 1, 1)
        
        # Áp dụng phép chuyển đổi cơ sở hệ tọa độ: M_norm = T_src * M_inv * T_dst^-1
        M_norm = torch.bmm(T_src, torch.bmm(M_inv, torch.linalg.inv(T_dst)))
        
        return M_norm[:, :2, :]
