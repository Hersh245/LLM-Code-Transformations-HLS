#pragma ACCEL kernel

void kernel_syrk(double alpha,double beta,double C[80][80],double A[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] *= beta;
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}
// Explanation:
// 1. Loop permutation: The innermost loop over 'k' has been moved to the second level of nesting, allowing for better data reuse and reducing loop overhead.
// 2. Loop fusion: The two inner loops over 'j' have been fused together to combine the operations of scaling and updating the elements of matrix C in a single loop iteration, improving memory access patterns and reducing loop overhead.