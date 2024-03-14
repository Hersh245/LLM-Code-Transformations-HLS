#pragma ACCEL kernel

void kernel_jacobi_1d(int tsteps, int n, double A[120], double B[120])
{
  int t;
  int i;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t < 40; t++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (i = 1; i < 120 - 1; i++) {
      B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
      A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop Fusion: The two separate loops that update arrays B and A have been fused into a single loop. This reduces the overhead of loop control and iteration, improving performance.
// 2. Loop Distribution: The loop fusion allows for better parallelization as both operations can now be done in a single iteration. This can lead to better resource utilization and faster execution.
// 3. Loop Permutation: The order of the nested loops has not been changed as it does not affect the performance significantly in this case.