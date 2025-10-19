package BD.individual.task1;

import java.util.Random;

public class MatrixBench {

    public static void main(String[] args) {

        int[] matrixSizes = {256, 512};
        int numRuns = 5;

        System.out.println("--- Java Matrix Multiplication Experiment ---");

        for (int N : matrixSizes) {
            runExperiment(N, numRuns);
        }
    }

    private static void initializeMatrix(double[][] matrix, int N) {
        Random random = new Random();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }
    }

    private static void runExperiment(int N, int numRuns) {
        System.out.println(String.format("\n--- Experiment N=%d (%d runs) ---", N, numRuns));

        double[][] A = new double[N][N];
        double[][] B = new double[N][N];
        double[][] C = new double[N][N];
        initializeMatrix(A, N);
        initializeMatrix(B, N);

        double totalTime = 0.0;

        for (int i = 0; i < 3; i++) { MatrixMultiplier.multiply(A, B, C, N); }
        
        for (int run = 0; run < numRuns; run++) {
            long startTime = System.nanoTime();
            MatrixMultiplier.multiply(A, B, C, N);
            long endTime = System.nanoTime();
            
            double duration = (endTime - startTime) / 1e9;
            totalTime += duration;

            System.out.println(String.format("Run %d/%d: %.6f seconds", run + 1, numRuns, duration));
        }

        double avgTime = totalTime / numRuns;
        System.out.println(String.format("\nAverage Execution Time (N=%d): %.6f seconds", N, avgTime));
    }
}
