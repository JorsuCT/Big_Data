import numpy as np
import time
from matrix_multiplication import multiply_matrices

def run_experiment(N: int, num_runs: int) -> float:
    """
    Sets up matrices, runs the multiplication multiple times, and reports
    the average execution time.
    """

    print(f"--- Python Experiment N={N} ({num_runs} runs) ---")

    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)

    times = []

    for run in range(num_runs):
        start_time = time.perf_counter()

        C = multiply_matrices(A, B, N)
        end_time = time.perf_counter()

        times.append(end_time - start_time)
        print(f"Run {run+1}/{num_runs}: {times[-1]:.6f} seconds")

    avg_time = np.mean(times)

    print(f"\nAverage Execution Time (N={N}): {avg_time:.6f} seconds")
    return avg_time

if __name__ == "__main__":

    MATRIX_SIZES = [256, 512] 
    NUM_REPETITIONS = 5

    all_results = {}
    print("--- Python Matrix Multiplication Experiment ---")

    for N in MATRIX_SIZES:
        all_results[N] = run_experiment(N, NUM_REPETITIONS)

    print("\n--- Summary ---")
    for N, avg_time in all_results.items():
        print(f"Python (N={N}): {avg_time:.6f} s")
