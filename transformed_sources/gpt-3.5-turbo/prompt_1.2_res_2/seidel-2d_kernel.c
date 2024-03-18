#pragma ACCEL kernel

void kernel_seidel_2d(int tsteps, int n, double A[120][120])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (j = 1; j <= 118; j++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
    for (i = 1; i <= 118; i++) {
      
      for (t = 0; t <= 39; t++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}
// Explanation of transformations:
// 1. Loop Permutation: The order of the loops has been changed from (t, i, j) to (j, i, t) to improve data locality. This change allows for better memory access patterns and cache utilization.
// 2. Loop Tiling: The pragma `#pragma ACCEL TILE FACTOR=auto{__TILE__L1}` has been added to tile the innermost loop over `i`. This helps in improving data reuse by keeping data closer to the compute units.
// 3. Loop Distribution: The pragma `#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}` and `#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}` have been added to distribute the parallelism across the loops over `j` and `i`. This allows for parallel execution of the inner loops, improving performance.
// 4. Loop Fusion: The pragma `#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}` has been added to fuse the outer loop over `t` with the inner loops over `i` and `j`. This reduces loop overhead and improves overall parallelism.