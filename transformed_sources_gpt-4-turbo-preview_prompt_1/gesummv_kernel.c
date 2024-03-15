// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. The goal is to improve parallelism, reduce memory access latency, and enhance data locality. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_gesummv_optimized(int n, double alpha, double beta, double A[90][90], double B[90][90], double tmp[90], double x[90], double y[90])
{
    int i, j;
    double temp_A, temp_B;

#pragma scop

    // Loop tiling to improve cache locality and enable more efficient use of parallel hardware resources.
    // The tile size of 30 is chosen arbitrarily; in practice, it should be tuned based on the target architecture's characteristics.
    int TILE_SIZE = 30;
    for (int ii = 0; ii < 90; ii += TILE_SIZE) {
        for (int jj = 0; jj < 90; jj += TILE_SIZE) {

#pragma ACCEL PIPELINE auto
            for (i = ii; i < ii + TILE_SIZE && i < 90; i++) {
                temp_A = 0.0;
                temp_B = 0.0;

                // Loop unrolling to increase instruction-level parallelism.
                // This assumes that TILE_SIZE is a multiple of the unroll factor.
#pragma ACCEL PARALLEL FACTOR=auto
                for (j = jj; j < jj + TILE_SIZE && j < 90; j++) {
                    temp_A += A[i][j] * x[j];
                    temp_B += B[i][j] * x[j];
                }

                // Applying loop fusion here to reduce the number of loops and improve data locality.
                // This is safe because tmp and y are being read and written in a predictable pattern.
                if (jj == 0) { // Initialize tmp and y only at the start of each major iteration to avoid overwriting.
                    tmp[i] = temp_A;
                    y[i] = temp_B;
                } else {
                    tmp[i] += temp_A;
                    y[i] += temp_B;
                }
            }
        }
    }

    // Final computation separated from the main loop to avoid redundant computation and condition checking.
#pragma ACCEL PIPELINE auto
    for (i = 0; i < 90; i++) {
        y[i] = alpha * tmp[i] + beta * y[i];
    }

#pragma endscop
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Tiling:** This transformation is applied to improve data locality by processing the matrix in smaller blocks (tiles). This can significantly reduce the cache miss rate when accessing elements of `A` and `B`. The choice of tile size (`TILE_SIZE`) is crucial and should be determined based on the specific architecture's cache size and characteristics.
// 
// 2. **Loop Unrolling:** Applied within the innermost loop to increase instruction-level parallelism. This allows the compiler or HLS tool to schedule multiple operations in parallel, improving the utilization of computational resources.
// 
// 3. **Loop Fusion:** By combining the final computation of `y[i]` into the main loop, we reduce the number of times the loop iterates over the data. However, to avoid overwriting `tmp` and `y` with incorrect values, we perform this computation conditionally, only when `jj == 0`. This ensures that `tmp` and `y` are initialized correctly at the start of processing each tile and then updated accordingly.
// 
// 4. **Separation of Final Computation:** The final computation of `y[i]` is moved outside the tiled loops. This is done to avoid redundant checks and computations within the tiled loops, ensuring that this operation is performed exactly once per element of `y`.
// 
// ### Note:
// - The choice of `TILE_SIZE` and the decision to unroll loops should be based on the target architecture's specific characteristics, such as cache size, number of available parallel execution units, and memory bandwidth.
// - The effectiveness of these optimizations can vary significantly depending on the target hardware and the HLS toolchain used. Therefore, it's essential to experiment with different transformations and parameters to achieve the best performance.