#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstring>

#include "affine_transform.h"

// Forward declarations of CUDA kernels
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
);

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
);

cudaError_t convert_bgr_to_rgb(
    const uint8_t* d_src,
    float* d_dst,
    int height,
    int width,
    float mean,
    float std,
    bool is_nchw,
    cudaStream_t stream
);
}

// ARCFACE standard landmarks
static constexpr float ARCFACE_DST[10] = {
    38.2946f, 51.6963f,
    73.5318f, 51.5014f,
    56.0252f, 71.7366f,
    41.5493f, 92.3655f,
    70.7299f, 92.2041f
};

namespace triton { namespace backend { namespace face_alignment {

class FaceAlignmentBackend : public BackendModel {
public:
    FaceAlignmentBackend(
        TRITONBACKEND_Model* triton_model,
        const char* model_name,
        const uint64_t version,
        TRITONSERVER_Message* backend_config,
        TRITONBACKEND_ModelState** state
    )
        : BackendModel(triton_model, model_name, version, backend_config, state)
    {
        // Parse output specs from model config
        InitializeOutputSpecs();
    }

    ~FaceAlignmentBackend() = default;

    TRITONSERVER_Error* Execute(
        TRITONBACKEND_ModelInstance* instance,
        const uint32_t request_count,
        TRITONBACKEND_Request** requests,
        TRITONBACKEND_Response** responses
    ) override;

private:
    struct OutputSpec {
        int size;
        float mean;
        float std;
        bool is_nchw;
    };

    std::unordered_map<std::string, OutputSpec> output_specs_;

    void InitializeOutputSpecs() {
        output_specs_["face_aligned_112"] = {112, 127.5f, 128.0f, true};
        output_specs_["face_aligned_224"] = {224, 127.5f, 127.5f, true};
        output_specs_["face_aligned_nhwc"] = {112, 0.0f, 255.0f, false};
    }

    TRITONSERVER_Error* ProcessRequest(
        TRITONBACKEND_Request* request,
        TRITONBACKEND_Response* response
    );

    TRITONSERVER_Error* AlignFace(
        const uint8_t* h_image,
        const float* h_landmarks,
        const float* h_bbox,
        int img_height,
        int img_width,
        const OutputSpec& spec,
        float* h_output
    );
};

TRITONSERVER_Error* FaceAlignmentBackend::Execute(
    TRITONBACKEND_ModelInstance* instance,
    const uint32_t request_count,
    TRITONBACKEND_Request** requests,
    TRITONBACKEND_Response** responses
) {
    // Get CUDA stream from instance
    cudaStream_t cuda_stream = nullptr;
    auto cuda_err = cudaStreamCreate(&cuda_stream);
    if (cuda_err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to create CUDA stream: " + std::string(cudaGetErrorString(cuda_err))).c_str()
        );
    }

    for (uint32_t r = 0; r < request_count; r++) {
        auto err = ProcessRequest(requests[r], responses[r]);
        if (err != nullptr) {
            TRITONBACKEND_ResponseSend(responses[r], err, nullptr);
            TRITONSERVER_ErrorDelete(err);
        }
    }

    cudaStreamDestroy(cuda_stream);
    return nullptr;
}

TRITONSERVER_Error* FaceAlignmentBackend::ProcessRequest(
    TRITONBACKEND_Request* request,
    TRITONBACKEND_Response* response
) {
    // Get input tensors
    uint32_t input_count = 0;
    TRITONBACKEND_RequestInputCount(request, &input_count);

    if (input_count != 3) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Expected 3 inputs: person_image, landmarks, bboxes"
        );
    }

    // Get person_image
    TRITONBACKEND_Input* image_input;
    TRITONBACKEND_RequestInput(request, "person_image", &image_input);

    const void* image_buffer;
    uint64_t image_buffer_size;
    uint32_t image_buffer_count;
    TRITONBACKEND_InputBuffer(image_input, 0, &image_buffer, &image_buffer_size, &image_buffer_count);

    // Get landmarks
    TRITONBACKEND_Input* landmarks_input;
    TRITONBACKEND_RequestInput(request, "landmarks", &landmarks_input);

    const void* landmarks_buffer;
    uint64_t landmarks_buffer_size;
    uint32_t landmarks_buffer_count;
    TRITONBACKEND_InputBuffer(landmarks_input, 0, &landmarks_buffer, &landmarks_buffer_size, &landmarks_buffer_count);

    // Get bboxes
    TRITONBACKEND_Input* bboxes_input;
    TRITONBACKEND_RequestInput(request, "bboxes", &bboxes_input);

    const void* bboxes_buffer;
    uint64_t bboxes_buffer_size;
    uint32_t bboxes_buffer_count;
    TRITONBACKEND_InputBuffer(bboxes_input, 0, &bboxes_buffer, &bboxes_buffer_size, &bboxes_buffer_count);

    // Get shape information from model config
    // (Shape info would come from model.proto/state)
    int img_height = 1080;  // Would be dynamic from input
    int img_width = 1920;
    int num_landmarks = 5;
    int num_bboxes = 1;

    const uint8_t* h_image = static_cast<const uint8_t*>(image_buffer);
    const float* h_landmarks = static_cast<const float*>(landmarks_buffer);
    const float* h_bboxes = static_cast<const float*>(bboxes_buffer);

    // Process each output format
    for (const auto& [output_name, spec] : output_specs_) {
        int output_size = num_bboxes * (spec.is_nchw ? 3 : 3) * spec.size * spec.size;
        std::vector<float> h_output(output_size);

        // Process each face
        for (int i = 0; i < num_bboxes; i++) {
            auto err = AlignFace(
                h_image,
                h_landmarks + i * num_landmarks * 2,
                h_bboxes + i * 4,
                img_height,
                img_width,
                spec,
                h_output.data() + i * (spec.is_nchw ? 3 : 3) * spec.size * spec.size
            );

            if (err != nullptr) {
                return err;
            }
        }

        // Create output tensor
        TRITONBACKEND_Output* output;
        TRITONBACKEND_ResponseOutput(
            response,
            &output,
            output_name.c_str(),
            TRITONSERVER_TYPE_FP32,
            nullptr,
            0  // Batch size
        );

        // Copy data to output
        uint8_t* output_buffer;
        TRITONBACKEND_OutputBuffer(
            output,
            &output_buffer,
            output_size * sizeof(float)
        );

        std::memcpy(output_buffer, h_output.data(), output_size * sizeof(float));
    }

    return nullptr;
}

TRITONSERVER_Error* FaceAlignmentBackend::AlignFace(
    const uint8_t* h_image,
    const float* h_landmarks,
    const float* h_bbox,
    int img_height,
    int img_width,
    const OutputSpec& spec,
    float* h_output
) {
    // Calculate ROI with margin
    const float margin = 0.3f;
    int x1 = static_cast<int>(h_bbox[0]);
    int y1 = static_cast<int>(h_bbox[1]);
    int x2 = static_cast<int>(h_bbox[2]);
    int y2 = static_cast<int>(h_bbox[3]);

    int w = x2 - x1;
    int h = y2 - y1;
    int mx = static_cast<int>(w * margin);
    int my = static_cast<int>(h * margin);

    x1 = std::max(0, x1 - mx);
    y1 = std::max(0, y1 - my);
    x2 = std::min(img_width - 1, x2 + mx);
    y2 = std::min(img_height - 1, y2 + my);

    int roi_width = x2 - x1;
    int roi_height = y2 - y1;

    // Allocate GPU memory
    uint8_t* d_image = nullptr;
    uint8_t* d_roi = nullptr;
    float* d_output = nullptr;
    float* d_landmarks = nullptr;
    float* d_M = nullptr;

    size_t image_size = img_height * img_width * 3 * sizeof(uint8_t);
    size_t roi_size = roi_height * roi_width * 3 * sizeof(uint8_t);
    size_t output_size = spec.size * spec.size * 3 * sizeof(float);

    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_roi, roi_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_landmarks, 5 * 2 * sizeof(float));
    cudaMalloc(&d_M, 6 * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_landmarks, h_landmarks, 5 * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Crop ROI
    crop_roi(
        d_image, d_roi, h_bbox, margin,
        img_height, img_width,
        roi_height, roi_width,
        x1, y1,
        nullptr  // stream
    );

    // Compute affine transformation matrix
    std::array<std::array<float, 2>, 5> src_landmarks;
    std::array<std::array<float, 2>, 5> dst_landmarks;

    // Copy local landmarks (relative to ROI)
    for (int i = 0; i < 5; i++) {
        src_landmarks[i][0] = h_landmarks[i * 2] - x1;
        src_landmarks[i][1] = h_landmarks[i * 2 + 1] - y1;
        dst_landmarks[i][0] = ARCFACE_DST[i * 2];
        dst_landmarks[i][1] = ARCFACE_DST[i * 2 + 1];
    }

    std::array<float, 6> M;
    if (!AffineTransformEstimator::estimateAffinePartial2D(src_landmarks, dst_landmarks, M)) {
        cudaFree(d_image);
        cudaFree(d_roi);
        cudaFree(d_output);
        cudaFree(d_landmarks);
        cudaFree(d_M);

        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "Failed to compute affine transformation matrix"
        );
    }

    // Copy matrix to GPU
    cudaMemcpy(d_M, M.data(), 6 * sizeof(float), cudaMemcpyHostToDevice);

    // Affine warp
    affine_warp(
        d_roi, d_output, d_M,
        roi_height, roi_width,
        spec.size, spec.size,
        spec.mean, spec.std,
        spec.is_nchw,
        nullptr  // stream
    );

    // Copy result back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_roi);
    cudaFree(d_output);
    cudaFree(d_landmarks);
    cudaFree(d_M);

    return nullptr;
}

}}}  // namespace triton::backend::face_alignment

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    // Implementation would initialize the backend
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    const uint32_t request_count,
    TRITONBACKEND_Request** requests,
    TRITONBACKEND_Response** responses
) {
    // Dispatch to backend implementation
    return nullptr;
}

}  // extern "C"
