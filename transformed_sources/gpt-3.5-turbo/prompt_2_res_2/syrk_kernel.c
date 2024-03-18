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
    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }
  }

#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (i = 0; i < 80; i++) {
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}
// Explanation of transformations:
// 1. Loop permutation: The outer loop over `i` has been split into two separate loops. The first loop now only updates the lower triangular part of matrix `C` with the scaling factor `beta`, while the second loop performs the main computation involving matrix multiplication with `alpha`. This separation allows for better pipelining and parallelization.
// 2. Loop tiling: The tiling pragma has been applied to the second loop over `i` to partition the computation into smaller tiles. This helps to improve data locality and reduces memory access overhead.
// 3. Loop distribution: The computation involving matrix multiplication has been distributed across the two loops over `i` and `k`. This distribution allows for better parallelization and pipelining of the matrix multiplication operations.
// 4. Loop fusion: No loop fusion has been applied in this case as the loops are already optimized for parallelization and pipelining.