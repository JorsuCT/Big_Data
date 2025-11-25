#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <immintrin.h>
#include <thread>
#include <mutex>
#include <vector>
#include <future>

using Matrix = std::vector<std::vector<double>>;

Matrix initialize_matrix(int N) {
    Matrix M(N, std::vector<double>(N));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[i][j] = dis(gen);
    return M;
}

Matrix transpose(const Matrix& M, int N) {
    Matrix T(N, std::vector<double>(N));
    for(int i=0; i<N; ++i)
        for(int j=0; j<N; ++j)
            T[i][j] = M[j][i];
    return T;
}

Matrix multiply_serial(const Matrix& A, const Matrix& B, int N) {
    Matrix C(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix multiply_openmp(const Matrix& A, const Matrix& B, int N) {
    Matrix C(N, std::vector<double>(N, 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

std::mutex progress_mutex;
int rows_completed = 0;

void worker_task(const Matrix& A, const Matrix& B, Matrix& C, int start_row, int end_row, int N) {
    for (int i = start_row; i < end_row; ++i) {
        for (int k = 0; k < N; ++k) {
            double val_a = A[i][k];
            for (int j = 0; j < N; ++j) {
                C[i][j] += val_a * B[k][j];
            }
        }
        
        if (i % 10 == 0) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            rows_completed++; 
        }
    }
}

Matrix multiply_manual_threads(const Matrix& A, const Matrix& B, int N) {
    Matrix C(N, std::vector<double>(N, 0.0));
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;
    
    int chunk_size = N / num_threads;
    rows_completed = 0;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? N : start + chunk_size;
        
        futures.push_back(std::async(std::launch::async, worker_task, 
                                     std::cref(A), std::cref(B), std::ref(C), start, end, N));
    }

    for (auto& f : futures) f.wait();
    return C;
}

Matrix multiply_simd_avx(const Matrix& A, const Matrix& B, int N) {
    Matrix C(N, std::vector<double>(N, 0.0));
    
    Matrix Bt = transpose(B, N); 

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256d sum_vec = _mm256_setzero_pd();
            double sum_arr[4];
            double final_sum = 0.0;

            int k = 0;
            for (; k <= N - 4; k += 4) {

                __m256d a_vals = _mm256_loadu_pd(&A[i][k]);
                __m256d b_vals = _mm256_loadu_pd(&Bt[j][k]);
                
                sum_vec = _mm256_fmadd_pd(a_vals, b_vals, sum_vec);
            }

            _mm256_storeu_pd(sum_arr, sum_vec);
            final_sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

            for (; k < N; ++k) {
                final_sum += A[i][k] * Bt[j][k];
            }
            C[i][j] = final_sum;
        }
    }
    return C;
}

void run_benchmark(int N) {
    std::cout << "\n=== Benchmarking Matrix Size N = " << N << " ===\n";
    
    auto A = initialize_matrix(N);
    auto B = initialize_matrix(N);
    int num_cores = std::thread::hardware_concurrency();

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C_serial = multiply_serial(A, B, N);
    auto end = std::chrono::high_resolution_clock::now();
    double time_serial = std::chrono::duration<double>(end - start).count();
    
    std::cout << std::left << std::setw(25) << "Method" 
              << std::setw(15) << "Time (s)" 
              << std::setw(15) << "Speedup" 
              << std::setw(15) << "Efficiency" << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    std::cout << std::left << std::setw(25) << "Serial (Baseline)" 
              << std::setw(15) << time_serial 
              << std::setw(15) << "1.00x" 
              << std::setw(15) << "-" << "\n";

    start = std::chrono::high_resolution_clock::now();
    Matrix C_omp = multiply_openmp(A, B, N);
    end = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double>(end - start).count();
    double speedup_omp = time_serial / time_omp;
    
    std::cout << std::left << std::setw(25) << "Parallel (OpenMP)" 
              << std::setw(15) << time_omp 
              << std::setw(15) << speedup_omp 
              << std::setw(15) << (speedup_omp / num_cores) << "\n";

    start = std::chrono::high_resolution_clock::now();
    Matrix C_thread = multiply_manual_threads(A, B, N);
    end = std::chrono::high_resolution_clock::now();
    double time_thread = std::chrono::duration<double>(end - start).count();
    double speedup_thread = time_serial / time_thread;

    std::cout << std::left << std::setw(25) << "Parallel (std::async)" 
              << std::setw(15) << time_thread 
              << std::setw(15) << speedup_thread 
              << std::setw(15) << (speedup_thread / num_cores) << "\n";

    start = std::chrono::high_resolution_clock::now();
    Matrix C_simd = multiply_simd_avx(A, B, N);
    end = std::chrono::high_resolution_clock::now();
    double time_simd = std::chrono::duration<double>(end - start).count();
    double speedup_simd = time_serial / time_simd;

    std::cout << std::left << std::setw(25) << "SIMD (AVX + OMP)" 
              << std::setw(15) << time_simd 
              << std::setw(15) << speedup_simd 
              << std::setw(15) << (speedup_simd / num_cores) << "\n";
}

int main() {
    std::cout << "CPUs detected: " << std::thread::hardware_concurrency() << "\n";
    std::vector<int> sizes = {256, 512, 1024};
    
    for (int N : sizes) {
        run_benchmark(N);
    }
    return 0;
}
