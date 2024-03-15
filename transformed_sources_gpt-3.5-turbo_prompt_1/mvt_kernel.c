// Here is the optimized code after applying loop tiling and loop permutation:
#pragma ACCEL kernel

void kernel_mvt(double x1[120], double x2[120], double y_1[120], double y_2[120], double A[120][120])
{
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 120; i++) {
    
#pragma ACCEL PARALLEL reduction = x1 FACTOR=auto{__PARA__L2}
    for (j = 0; j < 120; j++) {
      x1[i] += A[i][j] * y_1[j];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 120; j++) {
    
#pragma ACCEL PARALLEL reduction = x2 FACTOR=auto{__PARA__L3}
    for (i = 0; i < 120; i++) {
      x2[i] += A[j][i] * y_2[j];
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop Permutation: In the original code, the loops were nested with the outer loop iterating over `i` and the inner loop iterating over `j`. By permuting the loops, we can improve data locality and potentially increase parallelism by allowing better memory access patterns.
// 2. Loop Tiling: By adding loop tiling directives, we can partition the loops into smaller tiles that can fit into cache memory more efficiently. This can reduce memory access latency and improve data reuse, leading to better performance.