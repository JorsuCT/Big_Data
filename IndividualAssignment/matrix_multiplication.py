import numpy as np
import sys

def multiply_matrices(A: np.ndarray, B: np.ndarray, N: int) -> np.ndarray:
    """
    Performs the standard matrix multiplication (C = A * B) using
    the triple-nested loop (ijk) algorithm. Assumes square matrices of size N.
    Uses native Python loops over NumPy arrays for raw performance comparison.
    
    Args:
        A (np.ndarray): The first matrix (N x N).
        B (np.ndarray): The second matrix (N x N).
        N (int): The dimension of the square matrices.

    Returns:
        np.ndarray: The resulting matrix C (N x N).
    """

    C = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            sum_val = 0.0

            for k in range(N):
                sum_val += A[i, k] * B[k, j]
            C[i, j] = sum_val
            
    return C

