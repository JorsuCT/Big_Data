package BD.individual.task1;

public class MatrixMultiplier {
    
    /**
     * @param A The first matrix (N x N).
     * @param B The second matrix (N x N).
     * @param C The result matrix (must be initialized to N x N).
     * @param N The dimension of the square matrices.
     */

    public static void multiply(double[][] A, double[][] B, double[][] C, int N) {

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                C[i][j] = 0.0; 

                for (int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}
