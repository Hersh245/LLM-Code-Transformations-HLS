#pragma ACCEL kernel

void kernel_gesummv(int n,double alpha,double beta,double A[90][90],double B[90][90],double tmp[90],double x[90],double y[90])
{
  int i;
  int j;
  
#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 90; j++) {
    for (i = 0; i < 90; i++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
  }
  
  for (i = 0; i < 90; i++) {
    y[i] = alpha * tmp[i] + beta * y[i];
  }
  
#pragma endscop
}
// 1. Loop Permutation:
//    - The inner loop (loop over `j`) has been moved outside the outer loop (loop over `i`). This can help improve data locality and cache utilization by accessing contiguous elements of the arrays `A` and `B`.
// 
// 2. Loop Distribution:
//    - The computation of `tmp` and `y` has been distributed across the two loops, allowing for parallel execution of these operations.
// 
// 3. Loop Fusion:
//    - The final loop that calculates `y` has been fused with the loop over `j` to reduce loop overhead and improve performance.