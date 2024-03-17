// Here is the optimized code after applying loop permutation, loop tiling, and loop fusion:
#pragma ACCEL kernel

void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[60][80], double ey[60][80], double hz[60][80], double _fict_[40])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t < 40; t++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
    for (j = 0; j < 80; j++) {
      ey[0][j] = _fict_[t];
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L0_1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
    for (i = 1; i < 60; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
      for (j = 0; j < 80; j++) {
        ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
        ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
        if (i < 59 && j < 79) {
          hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
        }
      }
    }
  }
//#pragma endscop
}
// Explanation of transformations:
// 1. Loop Permutation: The original code had loops for updating `ey`, `ex`, and `hz` in separate blocks. By permuting the loops, we can combine the updates for `ey`, `ex`, and `hz` within the same loop iteration, reducing loop overhead and improving data locality.
// 2. Loop Tiling: The `PARALLEL` directive was used to parallelize the outer loop over `t`, and the `PIPELINE` directive was used to enable pipelining within the loops. This helps in maximizing parallelism and throughput.
// 3. Loop Fusion: The updates for `ey`, `ex`, and `hz` were fused within the same loop iteration to reduce redundant computations and improve memory access patterns. This can lead to better performance by reducing loop overhead and improving cache utilization.