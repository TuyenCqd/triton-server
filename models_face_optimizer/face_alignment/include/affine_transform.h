#pragma once

#include <vector>
#include <array>
#include <cmath>

/**
 * Compute 2D affine transformation matrix from point correspondences
 * Using least squares method with pseudo-inverse
 */
class AffineTransformEstimator {
public:
    /**
     * Estimate affine transformation matrix from source and destination landmarks
     * 
     * @param src_points: Source landmarks (5x2)
     * @param dst_points: Destination landmarks (5x2) - typically ARCFACE_DST
     * @param M: Output 2x3 transformation matrix (stored as [a, b, c, d, e, f])
     *          where x' = a*x + b*y + c, y' = d*x + e*y + f
     * @return: true if successful, false if matrix is singular
     */
    static bool estimateAffinePartial2D(
        const std::array<std::array<float, 2>, 5>& src_points,
        const std::array<std::array<float, 2>, 5>& dst_points,
        std::array<float, 6>& M
    ) {
        // Build system of equations: A * x = b
        // where x = [a, b, c, d, e, f]
        // and for each point: u_i = a*x_i + b*y_i + c, v_i = d*x_i + e*y_i + f

        // Normal equations: A^T * A * x = A^T * b
        float ATA[36] = {0};  // 6x6 matrix
        float ATb[6] = {0};   // 6x1 vector

        // Build normal equations
        for (int i = 0; i < 5; i++) {
            float xi = src_points[i][0];
            float yi = src_points[i][1];
            float ui = dst_points[i][0];
            float vi = dst_points[i][1];

            // For first 3 unknowns (a, b, c): u_i = a*x_i + b*y_i + c
            // Row in A: [x_i, y_i, 1, 0, 0, 0]
            ATA[0] += xi * xi;      // A^T*A[0,0]
            ATA[1] += xi * yi;      // A^T*A[0,1]
            ATA[2] += xi;           // A^T*A[0,2]
            ATA[6] += xi * yi;      // A^T*A[1,0]
            ATA[7] += yi * yi;      // A^T*A[1,1]
            ATA[8] += yi;           // A^T*A[1,2]
            ATA[12] += xi;          // A^T*A[2,0]
            ATA[13] += yi;          // A^T*A[2,1]
            ATA[14] += 1.0f;        // A^T*A[2,2]

            ATb[0] += xi * ui;
            ATb[1] += yi * ui;
            ATb[2] += ui;

            // For second 3 unknowns (d, e, f): v_i = d*x_i + e*y_i + f
            // Row in A: [0, 0, 0, x_i, y_i, 1]
            ATA[18] += xi * xi;     // A^T*A[3,3]
            ATA[19] += xi * yi;     // A^T*A[3,4]
            ATA[20] += xi;          // A^T*A[3,5]
            ATA[24] += xi * yi;     // A^T*A[4,3]
            ATA[25] += yi * yi;     // A^T*A[4,4]
            ATA[26] += yi;          // A^T*A[4,5]
            ATA[30] += xi;          // A^T*A[5,3]
            ATA[31] += yi;          // A^T*A[5,4]
            ATA[32] += 1.0f;        // A^T*A[5,5]

            ATb[3] += xi * vi;
            ATb[4] += yi * vi;
            ATb[5] += vi;
        }

        // Solve using Gaussian elimination with partial pivoting
        if (!solveLinearSystem(ATA, ATb, M.data())) {
            return false;
        }

        return true;
    }

private:
    /**
     * Solve 6x6 linear system using Gaussian elimination
     */
    static bool solveLinearSystem(float A[36], float b[6], float x[6]) {
        const float EPS = 1e-10f;

        // Forward elimination with partial pivoting
        for (int col = 0; col < 6; col++) {
            // Find pivot
            int pivot_row = col;
            float max_val = fabs(A[col * 6 + col]);

            for (int row = col + 1; row < 6; row++) {
                float val = fabs(A[row * 6 + col]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = row;
                }
            }

            if (max_val < EPS) {
                return false;  // Singular matrix
            }

            // Swap rows
            if (pivot_row != col) {
                for (int j = col; j < 6; j++) {
                    std::swap(A[col * 6 + j], A[pivot_row * 6 + j]);
                }
                std::swap(b[col], b[pivot_row]);
            }

            // Eliminate column
            float pivot = A[col * 6 + col];
            for (int j = col; j < 6; j++) {
                A[col * 6 + j] /= pivot;
            }
            b[col] /= pivot;

            for (int row = col + 1; row < 6; row++) {
                float factor = A[row * 6 + col];
                for (int j = col; j < 6; j++) {
                    A[row * 6 + j] -= factor * A[col * 6 + j];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        for (int i = 5; i >= 0; i--) {
            x[i] = b[i];
            for (int j = i + 1; j < 6; j++) {
                x[i] -= A[i * 6 + j] * x[j];
            }
        }

        return true;
    }
};
