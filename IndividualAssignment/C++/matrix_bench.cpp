#include "matrix_product.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace MatrixProd {
    using Matrix = std::vector<std::vector<double>>;
}

constexpr int NUM_REPETITIONS = 5;
const std::vector<int> MATRIX_SIZES = {256, 512};

void print_matrix_preview(const MatrixProd::Matrix& M, const std::string& name, int N) {
    constexpr int PREVIEW_SIZE = 8;
    std::cerr << "\n--- Preview of Result Matrix " << name << " (top-left "
              << std::min(N, PREVIEW_SIZE) << "x" << std::min(N, PREVIEW_SIZE) << ") ---" << std::endl;
    for (int i = 0; i < std::min(N, PREVIEW_SIZE); ++i) {
        for (int j = 0; j < std::min(N, PREVIEW_SIZE); ++j) {
            std::cerr << std::fixed << std::setprecision(4) << M[i][j] << "\t";
        }
        std::cerr << std::endl;
    }
}

void run_experiment(int N) {
    std::cerr << "\n--- Starting C++ Benchmark (N=" << N << ") ---" << std::endl;
    MatrixProd::Matrix A = MatrixProd::initialize_matrix(N);
    MatrixProd::Matrix B = MatrixProd::initialize_matrix(N);
    MatrixProd::Matrix C;
    std::vector<double> times;

    for (int run = 0; run < NUM_REPETITIONS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        C = MatrixProd::multiply(A, B, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        times.push_back(duration.count());
        std::cerr << "Run " << run + 1 << "/" << NUM_REPETITIONS
                  << ": " << std::fixed << std::setprecision(6)
                  << times.back() << " seconds" << std::endl;
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg_time = sum / NUM_REPETITIONS;
    std::cerr << "\nAverage Time (N=" << N << "): "
              << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;

    if (!C.empty()) {
        print_matrix_preview(C, "C", N);
    }
}

int main() {
    std::cerr << "Starting C++ Matrix Multiplication Benchmarks (O(N^3) algorithm)" << std::endl;
    for (int N : MATRIX_SIZES) {
        run_experiment(N);
    }
    std::cerr << "\nBenchmarks finished." << std::endl;
    return 0;
}
