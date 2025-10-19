#ifndef MATRIX_PRODUCT_HPP
#define MATRIX_PRODUCT_HPP

#include <vector>

namespace MatrixProd {
    // Define the matrix type as a standard vector of vectors (allocated on the heap in main).
    using Matrix = std::vector<std::vector<double>>;

    /**
     * @brief Generates a square matrix of size N, filled with random doubles.
     */
    Matrix initialize_matrix(int N);

    /**
     * @brief Performs standard O(N^3) matrix multiplication (C = A * B).
     * Takes constant references to the matrices to prevent unnecessary copying.
     */
    Matrix multiply(const Matrix& A, const Matrix& B, int N);
}

#endif // MATRIX_PRODUCT_HPP
