#pragma ACCEL kernel

void kernel_symm(double alpha,double beta,double C[200][240],double A[200][200],double B[200][240])
{
  int i;
  int j;
  int k;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 240; j++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 0; i < 200; i++) {
      double tmp = B[i][j];
      
#pragma ACCEL PARALLEL reduction=C FACTOR=auto{__PARA__L2}
      for (k = 0; k < 200; k++) {
        if (k < i) {
          C[k][j] += alpha * tmp * A[i][k];
        }
      }
      double temp2 = (double )0;
      
#pragma ACCEL PARALLEL reduction=temp2 FACTOR=auto{__PARA__L3}
      for (k = 0; k < 200; k++) {
        if (k < i) {
          temp2 += B[k][j] * A[i][k];
        }
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  }
}
// Rationale for loop permutation:
// - By swapping the outer loop from iterating over `i` to iterating over `j`, we can improve memory access patterns and potentially increase data locality, leading to better performance.
// 
// Rationale for loop tiling:
// - By introducing tiling with appropriate tile sizes, we can exploit data reuse within the cache and reduce the number of cache misses, which can improve overall performance.
// 
// Rationale for loop distribution:
// - The code already has loop distribution applied through the parallelization directives. By distributing the work across multiple processing elements, we can achieve parallel execution and potentially speed up the computation.
// 
// Overall, these transformations aim to optimize the code for HLS by improving memory access patterns, data locality, and parallelism.