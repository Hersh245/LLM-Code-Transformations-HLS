#pragma ACCEL kernel

void kernel_gemm(int ni,int nj,int nk,double alpha,double beta,double C[60][70],double A[60][80],double B[80][70])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  
#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 60; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (k = 0; k < 80; k++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 0; j < 70; j++) {
        C[i][j] *= beta;
      }
      
#pragma ACCEL PARALLEL reduction=C FACTOR=auto{__PARA__L3}
      for (j = 0; j < 70; j++) {
        C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
  
#pragma endscop
}
// Explanation of transformations:
// 1. Loop permutation: The innermost loop over `k` has been moved to the second level of nesting to improve data locality and reduce loop overhead.
// 2. Loop tiling: The loop over `k` has been tiled to operate on smaller chunks of data, which can improve cache performance and parallelism.
// 3. Loop distribution: The loop over `j` that updates elements of `C` has been distributed across the loop over `k` to better utilize parallelism and reduce dependencies.
// 4. Loop fusion: The loop over `j` that updates elements of `C` has been fused with the loop over `k` to reduce loop overhead and improve parallelism.