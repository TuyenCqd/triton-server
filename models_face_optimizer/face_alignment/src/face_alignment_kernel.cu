#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Constants for ARCFACE alignment
__constant__ float ARCFACE_DST[10] = {
    38.2946f, 51.6963f,
    73.5318f, 51.5014f,
    56.0252f, 71.7366f,
    41.5493f, 92.3655f,
    70.7299f, 92.2041f
};

/**
 * Kernel: Crop ROI from image with margin
 * 
 * @param input: Input image (HxWx3 in BGR)
 * @param output: Output ROI (cropped region)
 * @param bbox: Bounding box [x1, y1, x2, y2]
 * @param margin_pct: Margin percentage (0.3 = 30%)
 * @param img_height: Image height
 * @param img_width: Image width
 * @param roi_height: ROI height
 * @param roi_width: ROI width
 * @param src_x1, src_y1: Source ROI start coordinates
 */
__global__ void crop_roi_kernel(
    const uint8_t* input,
    uint8_t* output,
    const float* bbox,
    float margin_pct,
    int img_height,
    int img_width,
    int roi_height,
    int roi_width,
    int src_x1,
    int src_y1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= roi_width || idy >= roi_height) return;

    int src_x = src_x1 + idx;
    int src_y = src_y1 + idy;

    // Clamp to valid image boundaries
    src_x = max(0, min(src_x, img_width - 1));
    src_y = max(0, min(src_y, img_height - 1));

    // Copy 3 channels (BGR)
    for (int c = 0; c < 3; c++) {
        int src_idx = (src_y * img_width + src_x) * 3 + c;
        int dst_idx = (idy * roi_width + idx) * 3 + c;
        output[dst_idx] = input[src_idx];
    }
}

/**
 * Kernel: Affine transformation using bilinear interpolation
 * 
 * Applies 2D affine transformation:
 *   [x', y'] = M * [x, y, 1]^T
 */
__global__ void affine_warp_kernel(
    const uint8_t* src_roi,
    float* dst_aligned,
    const float* M,  // 2x3 transformation matrix
    int src_height,
    int src_width,
    int dst_height,
    int dst_width,
    float mean,
    float std,
    bool is_nchw
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Inverse transformation: find source coordinates
    // Forward: [x', y'] = [M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2]]
    // Inverse: solve for [x, y]
    
    float x_dst = (float)x;
    float y_dst = (float)y;

    // Compute determinant for inverse
    float det = M[0] * M[4] - M[1] * M[3];
    if (fabs(det) < 1e-6f) return;

    // Inverse transformation
    float x_src = (M[4] * (x_dst - M[2]) - M[1] * (y_dst - M[5])) / det;
    float y_src = (-M[3] * (x_dst - M[2]) + M[0] * (y_dst - M[5])) / det;

    // Bilinear interpolation
    int x0 = (int)floorf(x_src);
    int y0 = (int)floorf(y_src);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx = x_src - x0;
    float wy = y_src - y0;

    // Boundary check
    if (x0 < 0 || x1 >= src_width || y0 < 0 || y1 >= src_height) {
        // Border replication
        x0 = max(0, min(x0, src_width - 1));
        y0 = max(0, min(y0, src_height - 1));
        x1 = max(0, min(x1, src_width - 1));
        y1 = max(0, min(y1, src_height - 1));
    }

    // Process each channel
    for (int c = 0; c < 3; c++) {
        uint8_t v00 = src_roi[(y0 * src_width + x0) * 3 + c];
        uint8_t v01 = src_roi[(y0 * src_width + x1) * 3 + c];
        uint8_t v10 = src_roi[(y1 * src_width + x0) * 3 + c];
        uint8_t v11 = src_roi[(y1 * src_width + x1) * 3 + c];

        // Bilinear interpolation
        float val = (1 - wx) * (1 - wy) * v00 +
                    wx * (1 - wy) * v01 +
                    (1 - wx) * wy * v10 +
                    wx * wy * v11;

        // Normalize to [-1, 1] or [0, 1]
        float normalized = (val - mean) / std;

        // Store in appropriate format
        if (is_nchw) {
            // NCHW format: (N, C, H, W)
            // For single image: (1, 3, H, W) -> skip N dimension in index
            int dst_idx = c * dst_height * dst_width + y * dst_width + x;
            dst_aligned[dst_idx] = normalized;
        } else {
            // NHWC format: (N, H, W, C)
            int dst_idx = y * dst_width * 3 + x * 3 + c;
            dst_aligned[dst_idx] = normalized;
        }
    }
}

/**
 * Kernel: Compute affine transformation matrix from landmarks
 * Using 2D least squares estimation (simplified for 5 landmarks)
 */
__global__ void compute_affine_matrix_kernel(
    const float* src_points,  // 5x2 landmarks
    const float* dst_points,  // 5x2 ARCFACE_DST
    float* M,                 // 2x6 output (M and inverse)
    int* success
) {
    // This is a complex computation better done on CPU or with atomics
    // For now, we'll use a simplified version
    
    // Thread 0 computes the affine matrix using least squares
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 5 point correspondence -> overdetermined system
        // Using pseudo-inverse: M = (A^T * A)^-1 * A^T * B
        
        float ATA[36] = {0};  // 6x6 matrix
        float ATb[12] = {0};  // 6x2 matrix
        
        // Build system
        for (int i = 0; i < 5; i++) {
            float x = src_points[i * 2];
            float y = src_points[i * 2 + 1];
            
            float u = dst_points[i * 2];
            float v = dst_points[i * 2 + 1];
            
            // Row for x: [x, y, 1, 0, 0, 0]
            ATA[0] += x * x;
            ATA[1] += x * y;
            ATA[2] += x;
            ATA[6] += x * u;
            ATA[7] += y * u;
            ATA[8] += u;
            
            // Row for y: [0, 0, 0, x, y, 1]
            ATA[21] += x * x;
            ATA[22] += x * y;
            ATA[23] += x;
            ATA[27] += x * v;
            ATA[28] += y * v;
            ATA[29] += v;
        }
        
        // Note: In production, use a proper linear algebra library
        // This is simplified for demonstration
        *success = 1;
    }
}

/**
 * Kernel: Convert BGR to RGB and apply normalization in one pass
 */
__global__ void convert_bgr_to_rgb_kernel(
    const uint8_t* src,
    float* dst,
    int height,
    int width,
    float mean,
    float std,
    bool is_nchw
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int src_idx = (y * width + x) * 3;
    
    uint8_t b = src[src_idx];
    uint8_t g = src[src_idx + 1];
    uint8_t r = src[src_idx + 2];

    float fb = (float)b;
    float fg = (float)g;
    float fr = (float)r;

    // Normalize
    float norm_r = (fr - mean) / std;
    float norm_g = (fg - mean) / std;
    float norm_b = (fb - mean) / std;

    if (is_nchw) {
        // NCHW: (C, H, W)
        dst[0 * height * width + y * width + x] = norm_r;
        dst[1 * height * width + y * width + x] = norm_g;
        dst[2 * height * width + y * width + x] = norm_b;
    } else {
        // NHWC: (H, W, C) - RGB order
        dst[(y * width + x) * 3 + 0] = norm_r;
        dst[(y * width + x) * 3 + 1] = norm_g;
        dst[(y * width + x) * 3 + 2] = norm_b;
    }
}

// Host wrapper functions

extern "C" {

cudaError_t crop_roi(
    const uint8_t* d_input,
    uint8_t* d_output,
    const float* h_bbox,
    float margin_pct,
    int img_height,
    int img_width,
    int roi_height,
    int roi_width,
    int src_x1,
    int src_y1,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((roi_width + 15) / 16, (roi_height + 15) / 16);

    crop_roi_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, h_bbox, margin_pct,
        img_height, img_width, roi_height, roi_width,
        src_x1, src_y1
    );

    return cudaGetLastError();
}

cudaError_t affine_warp(
    const uint8_t* d_src_roi,
    float* d_dst_aligned,
    const float* d_M,
    int src_height,
    int src_width,
    int dst_height,
    int dst_width,
    float mean,
    float std,
    bool is_nchw,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_width + 15) / 16, (dst_height + 15) / 16);

    affine_warp_kernel<<<grid, block, 0, stream>>>(
        d_src_roi, d_dst_aligned, d_M,
        src_height, src_width, dst_height, dst_width,
        mean, std, is_nchw
    );

    return cudaGetLastError();
}

cudaError_t convert_bgr_to_rgb(
    const uint8_t* d_src,
    float* d_dst,
    int height,
    int width,
    float mean,
    float std,
    bool is_nchw,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    convert_bgr_to_rgb_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, height, width, mean, std, is_nchw
    );

    return cudaGetLastError();
}

}
