#pragma ACCEL kernel

void kernel_gesummv(double alpha,double beta,double A[250][250],double B[250][250],double tmp[250],double x[250],double y[250])
{
  int i;
  int j;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 250; j++) {
    for (i = 0; i < 250; i++) {
      tmp[i] += A[i][j] * x[j];
      y[i] += B[i][j] * x[j];
    }
  }
  
  for (i = 0; i < 250; i++) {
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}
// 1. Loop permutation: The inner and outer loops have been swapped to improve data locality and cache efficiency. This change can help in optimizing memory access patterns.
// 2. Loop distribution: The computation of `tmp` and `y` has been distributed across the two loops to allow for better parallelization. This can help in utilizing the available hardware resources more efficiently.
// 3. Loop fusion: The final loop that calculates the values of `y` has been separated from the loop that performs the matrix-vector multiplication. This separation can help in reducing loop dependencies and improving parallelism.