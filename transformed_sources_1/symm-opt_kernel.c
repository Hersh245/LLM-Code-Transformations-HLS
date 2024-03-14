#pragma ACCEL kernel

void kernel_symm(double alpha,double beta,double C[60][80],double A[60][60],double B[60][80])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//SIDE = 'L'
//UPLO = 'L'
// =>  Form  C := alpha*A*B + beta*C
// A is MxM
// B is MxN
// C is MxN
//note that due to Fortran array layout, the code below more closely resembles upper triangular case in BLAS
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 0; i < 60; i++) {
      
      double tmp = B[i][j];
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (k = 0; k < 60; k++) {
        if (k < i) {
          C[k][j] += alpha * tmp * A[i][k];
        }
      }

      double temp2 = ((double )0);
#pragma ACCEL PARALLEL reduction=temp2 FACTOR=auto{__PARA__L3}
      for (k = 0; k < 60; k++) {
        if (k < i) {
          temp2 += B[k][j] * A[i][k];
        }
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  }
}
// Explanation of transformations:
// 1. Loop permutation: The outer loop has been changed to iterate over `j` instead of `i`. This can help improve data locality and cache utilization as the inner loop accesses elements of `B` and `C` in a contiguous manner.
// 2. Loop tiling: Tiling has been applied to both the outer and inner loops to break down the computation into smaller chunks that can fit into cache more efficiently. This can help reduce memory access latency and improve parallelism.
// 3. Loop distribution: The code has not been distributed into multiple loops as the original code already has nested loops that can be parallelized independently.
// 4. Loop fusion: Loop fusion has not been applied as the loops are already tightly nested and cannot be fused together without affecting the computation logic.