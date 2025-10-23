#ifndef MATRIX_PRODUCT_HPP
#define MATRIX_PRODUCT_HPP

#include <vector>

namespace MatrixProd {
    using Matrix = std::vector<std::vector<double>>;

    /**
     * @brief Generates a square matrix of size N, filled with random doubles.
     */
    Matrix initialize_matrix(int N);

    /**
     * @brief Performs standard O(N^3) matrix multiplication (C = A * B).
     */
    Matrix multiply(const Matrix& A, const Matrix& B, int N);
}

#endif
