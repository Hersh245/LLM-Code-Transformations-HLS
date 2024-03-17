// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. These transformations aim to improve data locality, parallelism, and pipeline efficiency. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[60][80], double ey[60][80], double hz[60][80], double _fict_[40]) {
  int t, i, j;

  // Loop tiling is applied to improve data locality and enable more efficient use of on-chip memory.
  // Loop permutation is applied to ensure that the innermost loop has the best spatial locality.
  // Loop fusion is avoided here as it might reduce parallelism opportunities and increase the complexity of dependencies.
  
  #pragma ACCEL PIPELINE auto
  for (t = 0; t < 40; t++) {
    #pragma ACCEL PARALLEL FACTOR=auto
    for (j = 0; j < 80; j++) {
      ey[0][j] = _fict_[t];
    }

    // Applying loop tiling for the ey update
    int TILE_SIZE_i = 10; // Example tile size for i dimension
    int TILE_SIZE_j = 20; // Example tile size for j dimension
    for (i = 1; i < 60; i += TILE_SIZE_i) {
      for (j = 0; j < 80; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 60; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 80; ++tj) {
            ey[ti][tj] = ey[ti][tj] - 0.5 * (hz[ti][tj] - hz[ti - 1][tj]);
          }
        }
      }
    }

    // Applying loop tiling for the ex update
    for (i = 0; i < 60; i += TILE_SIZE_i) {
      for (j = 1; j < 80; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 60; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 80; ++tj) {
            ex[ti][tj] = ex[ti][tj] - 0.5 * (hz[ti][tj] - hz[ti][tj - 1]);
          }
        }
      }
    }

    // Applying loop tiling for the hz update
    for (i = 0; i < 59; i += TILE_SIZE_i) {
      for (j = 0; j < 79; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 59; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 79; ++tj) {
            hz[ti][tj] = hz[ti][tj] - 0.7 * (ex[ti][tj + 1] - ex[ti][tj] + ey[ti + 1][tj] - ey[ti][tj]);
          }
        }
      }
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Tiling:** This transformation is applied to the `ey`, `ex`, and `hz` update loops. It helps in improving data locality by working on smaller sub-blocks of the data array that fit into the faster, but smaller, on-chip memory. This reduces the number of memory accesses to the slower off-chip memory.
// 
// 2. **Loop Permutation:** Although not explicitly shown as a separate transformation, the loop ordering inside the tiled loops is chosen to ensure that the data is accessed in a way that is favorable for spatial locality, which is important for achieving high memory bandwidth utilization.
// 
// 3. **Parallelism:** The `#pragma ACCEL PARALLEL FACTOR=auto` directive is used to hint at automatic parallelization opportunities within the loops. This is crucial for exploiting the parallel compute resources available in FPGAs.
// 
// 4. **Pipeline:** The `#pragma ACCEL PIPELINE auto` directive outside the time loop (`t`) suggests that the compiler should attempt to pipeline this loop, allowing for different iterations of the loop to be executed in an overlapped manner, further increasing the throughput of the computation.
// 
// By applying these transformations, the code is optimized for HLS targeting FPGA architectures, which can significantly improve the performance and efficiency of the FDTD 2D kernel computation.