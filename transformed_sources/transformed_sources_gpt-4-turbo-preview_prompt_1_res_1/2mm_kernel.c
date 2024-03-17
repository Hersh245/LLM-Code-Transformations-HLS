// To optimize the given C code for High-Level Synthesis (HLS), we can apply several code transformations. These transformations aim to improve parallelism, data locality, and resource utilization, which are crucial for achieving high performance on hardware accelerators like FPGAs. Below is the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_2mm(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[40][50], double A[40][70], double B[70][50], double C[50][80], double D[40][80]) {
  int i, j, k;

  // Loop tiling for improved data locality and parallelism
  int TILE_SIZE_I = 10; // Example tile size, adjust based on target hardware constraints
  int TILE_SIZE_J = 10;
  int TILE_SIZE_K = 10;

  // Temporarily hold tile results to reduce memory accesses
  double temp_tile[TILE_SIZE_I][TILE_SIZE_J];

  // First tiled loop for matrix multiplication A*B
  for (int ii = 0; ii < 40; ii += TILE_SIZE_I) {
    for (int jj = 0; jj < 50; jj += TILE_SIZE_J) {
      for (int kk = 0; kk < 70; kk += TILE_SIZE_K) {

        // Initialize temporary tile storage
        for (i = 0; i < TILE_SIZE_I; i++) {
          for (j = 0; j < TILE_SIZE_J; j++) {
            if (kk == 0) { // Only reset tmp when processing the first K-tile
              temp_tile[i][j] = 0.0;
            }
          }
        }

        // Perform multiplication on tiles
        for (i = 0; i < TILE_SIZE_I; i++) {
          for (k = 0; k < TILE_SIZE_K; k++) {
            for (j = 0; j < TILE_SIZE_J; j++) {
              temp_tile[i][j] += alpha * A[ii + i][kk + k] * B[kk + k][jj + j];
            }
          }
        }

        // Store results back to tmp from temporary tile storage
        for (i = 0; i < TILE_SIZE_I; i++) {
          for (j = 0; j < TILE_SIZE_J; j++) {
            tmp[ii + i][jj + j] = temp_tile[i][j];
          }
        }
      }
    }
  }

  // Second loop for D := beta*D + tmp*C
  // Loop tiling not applied here for simplicity, but similar principles could be used
  for (i = 0; i < 40; i++) {
    for (j = 0; j < 80; j++) {
      D[i][j] *= beta;
      for (k = 0; k < 50; ++k) {
        D[i][j] += tmp[i][k] * C[k][j];
      }
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Tiling:** This transformation is applied to the first loop nest that performs the multiplication `A*B`. Loop tiling, also known as loop blocking, helps in improving data locality by processing the matrix in smaller blocks or tiles. This can significantly reduce the number of memory accesses and improve cache utilization, which is critical for performance on many architectures.
// 
// 2. **Temporary Tile Storage:** Within the tiled loops, a temporary storage (`temp_tile`) is used to hold the results of tile multiplication before writing them back to the `tmp` array. This reduces the frequency of memory accesses to the `tmp` array, further enhancing data locality and potentially reducing memory bandwidth requirements.
// 
// 3. **Loop Order Adjustment:** Inside the tiled multiplication, the loop order is adjusted to `i-k-j` from the original `i-j-k`. This adjustment is made to ensure that the innermost loop iterates over the `j` dimension, which is contiguous in memory for the `temp_tile` and `B` arrays, thus exploiting spatial locality.
// 
// 4. **Loop Fusion/Distribution:** The original code could be further optimized by fusing loops that have the same loop bounds and are independent, or by distributing loops to separate different concerns (e.g., initialization from computation). However, in this transformation, the focus was on loop tiling and adjusting loop orders for clarity.
// 
// These transformations are designed to optimize the code for HLS by improving parallelism and data locality. The exact tile sizes (`TILE_SIZE_I`, `TILE_SIZE_J`, `TILE_SIZE_K`) should be chosen based on the specific hardware constraints and the size of the available on-chip memory to achieve the best performance.