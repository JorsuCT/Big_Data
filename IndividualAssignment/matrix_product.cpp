#include "matrix_product.hpp"
#include <iostream>
#include <random>
#include <vector>

/**
 * Production function for matrix multiplication.
 * Uses std::vector<std::vector<double>> for matrix representation.
 */
namespace MatrixProd {

    /**
     * Initializes a square matrix of size N with random double values.
     * @param N The dimension of the square matrix.
     * @return The initialized Matrix.
     */
    Matrix initialize_matrix(int N) {
        Matrix M(N, std::vector<double>(N));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                M[i][j] = dis(gen);
            }
        }
        return M;
    }

    /**
     * Performs the standard matrix multiplication (C = A * B) using ijk loop order.
     * @param A The first matrix (N x N).
     * @param B The second matrix (N x N).
     * @param N The dimension.
     * @return The result matrix C (N x N).
     */
    Matrix multiply(const Matrix& A, const Matrix& B, int N) {
        // Initialize C with N rows, each containing N zeros
        Matrix C(N, std::vector<double>(N, 0.0));

        // Standard ijk loop order
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum_val = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum_val += A[i][k] * B[k][j];
                }
                C[i][j] = sum_val;
            }
        }
        return C;
    }
} // namespace MatrixProd
