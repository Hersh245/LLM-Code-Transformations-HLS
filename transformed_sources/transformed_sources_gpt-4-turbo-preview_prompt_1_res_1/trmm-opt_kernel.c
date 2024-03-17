// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. The goal is to improve parallelism, reduce memory access latency, and make efficient use of the FPGA resources. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_trmm(double alpha, double A[60][60], double B[60][80]) {
  double temp_B[60][80];

  // Loop tiling
  // We tile the loops to work on smaller blocks of data at a time, which can improve cache locality and parallelism.
  // The tile sizes (TILE_SIZE_I and TILE_SIZE_J) are chosen based on the typical block sizes that fit well into the FPGA's local memory.
  const int TILE_SIZE_I = 10; // Example tile size, adjust based on FPGA resources and experimentation
  const int TILE_SIZE_J = 16; // Example tile size

  #pragma ACCEL PIPELINE auto
  for (int ii = 0; ii < 60; ii += TILE_SIZE_I) {
    for (int jj = 0; jj < 80; jj += TILE_SIZE_J) {
      
      // Loop permutation
      // Permute the loops to bring the innermost loop that has the independent iterations to the outer,
      // to enhance parallel execution opportunities.
      for (int k = 0; k < 60; k++) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int i = ii; i < ii + TILE_SIZE_I && i < 60; i++) {
          for (int j = jj; j < jj + TILE_SIZE_J && j < 80; j++) {
            if (k == 0) {
              temp_B[i][j] = B[i][j]; // Initialize temp_B with B values
            }
            if (k > i) {
              temp_B[i][j] += A[k][i] * B[k][j];
            }
          }
        }
      }

      // Apply alpha multiplication after accumulation to reduce the number of operations
      for (int i = ii; i < ii + TILE_SIZE_I && i < 60; i++) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int j = jj; j < jj + TILE_SIZE_J && j < 80; j++) {
          B[i][j] = alpha * temp_B[i][j];
        }
      }
    }
  }
}
// ### Rationale Behind Each Transformation:
// 
// 1. **Loop Tiling**: This transformation is applied to work on smaller blocks of the matrices at a time. It helps in improving the cache locality and can lead to better parallelism because smaller blocks of data can be loaded into the FPGA's local memory, reducing memory access times.
// 
// 2. **Loop Permutation**: Changing the order of the loops can significantly impact performance. By moving the loop with independent iterations to the outermost position, we can enhance the parallel execution opportunities. This is because independent iterations can be executed in parallel, which is beneficial in an FPGA context where parallelism is a key factor for performance.
// 
// 3. **Temporary Buffer (`temp_B`)**: Instead of directly updating `B`, we use a temporary buffer. This allows us to accumulate the results first and then apply the multiplication by `alpha` in a separate step. This reduces the number of multiplications needed and ensures that the multiplication by `alpha` is only done once per element of `B`, after all additions are completed.
// 
// 4. **Parallelism Pragmas**: The use of `#pragma ACCEL PARALLEL FACTOR=auto` hints to the HLS tool that it should automatically determine the optimal parallelism factor for the loop. This is crucial for achieving high performance, as it allows the compiler to exploit data parallelism to the fullest extent possible.
// 
// These transformations aim to make the code more suitable for HLS by improving data locality, reducing memory access times, and maximizing parallel execution opportunities. The exact tile sizes (`TILE_SIZE_I` and `TILE_SIZE_J`) and other parameters should be tuned based on the target FPGA's resources and the specific requirements of the application.