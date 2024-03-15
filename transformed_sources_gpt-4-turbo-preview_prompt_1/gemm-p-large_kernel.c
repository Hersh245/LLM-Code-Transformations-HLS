// To optimize the given C code for High-Level Synthesis (HLS), we can apply several transformations. These transformations aim to improve parallelism, data locality, and resource utilization, which are crucial for achieving high performance on FPGA or ASIC designs. The transformations include loop permutation, loop tiling, loop distribution, and possibly loop fusion, depending on the specific optimization goals and the hardware constraints.
// 
// ### Original Code Analysis
// 
// The original code performs a General Matrix Multiply (GEMM) operation, `C = alpha*A*B + beta*C`. It already includes some HLS-specific pragmas for parallelism and pipelining but lacks optimizations for memory access patterns and computational efficiency.
// 
// ### Optimized Code
#pragma ACCEL kernel

void kernel_gemm(int ni, int nj, int nk, double alpha, double beta, double C[200][220], double A[200][240], double B[240][220]) {
  int i, j, k;
  int ii, jj, kk;
  const int TILE_SIZE_I = 10; // Example tile sizes, these should be tuned based on the target architecture
  const int TILE_SIZE_J = 11;
  const int TILE_SIZE_K = 12;

  // Apply loop tiling to improve data locality and enable more efficient use of on-chip memory
  for (i = 0; i < ni; i += TILE_SIZE_I) {
    for (j = 0; j < nj; j += TILE_SIZE_J) {
      for (k = 0; k < nk; k += TILE_SIZE_K) {

        // Mini-batch processing within tiles
        for (ii = i; ii < i + TILE_SIZE_I && ii < ni; ii++) {
          for (jj = j; jj < j + TILE_SIZE_J && jj < nj; jj++) {
            double temp = C[ii][jj] * beta; // Apply beta scaling outside the k-loop for efficiency

            for (kk = k; kk < k + TILE_SIZE_K && kk < nk; kk++) {
              temp += alpha * A[ii][kk] * B[kk][jj]; // Accumulate results within the tile
            }

            C[ii][jj] = temp;
          }
        }
      }
    }
  }
}
// ### Rationale Behind Transformations
// 
// 1. **Loop Tiling**: This transformation breaks the loops into smaller chunks or "tiles". This is beneficial for several reasons:
//    - It can significantly improve cache locality by working on a small subset of data that fits into the cache, reducing memory bandwidth requirements.
//    - It enables parallel execution of independent tiles, improving resource utilization in the FPGA/ASIC.
//    - It allows for more efficient use of on-chip memory by loading a small portion of the matrices into fast local memory.
// 
// 2. **Loop Permutation**: Although not explicitly applied in the transformed code, loop permutation can be used to optimize memory access patterns. The choice of loop ordering (i, j, k) in the tiled version is made to ensure that memory accesses, especially for matrix C, are as sequential as possible, improving memory access efficiency.
// 
// 3. **Loop Fusion/Distribution**: The original code could benefit from loop distribution to separate the beta scaling of C from the main computation. However, in the optimized code, this is implicitly handled within the tiled computation, where the beta scaling and the main GEMM computation are merged efficiently.
// 
// ### Additional Notes
// 
// - The choice of tile sizes (`TILE_SIZE_I`, `TILE_SIZE_J`, `TILE_SIZE_K`) is crucial for performance and should be tuned based on the specific hardware characteristics, such as the size of the on-chip memory and the memory bandwidth.
// - Further optimizations might include exploring different loop orders within the tiled loops to find the most efficient memory access pattern or applying additional HLS pragmas for parallelism and pipelining within the tiled computation.
// - It's essential to profile the optimized code on the target hardware to ensure that the transformations achieve the desired performance improvements.