// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count (TC), accumulated cycles (AC), and cycles per call (CPC).
// 
// 1. Loop Tiling:
//    - We can tile the loops to improve data locality and reduce the number of cache misses. This can help optimize memory access patterns and reduce the overall cycles per call (CPC).
// 
// 2. Loop Permutation:
//    - By permuting the loops, we can potentially improve parallelism and reduce dependencies between iterations. This can lead to better pipelining and lower accumulated cycles (AC).
// 
// Here is the transformed code with loop tiling and permutation applied:
#pragma ACCEL kernel

void kernel_gemver(int n,double alpha,double beta,double A[120][120],double u1[120],double v1[120],double u2[120],double v2[120],double w[120],double x[120],double y[120],double z[120])
{
  int i;
  int j;
  
#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 120; i++) {
    
#pragma ACCEL PARALLEL reduction=A FACTOR=auto{__PARA__L4}
    for (j = 0; j < 120; j++) {
      A[i][j] += u1[i] * v1[j] + u2[i] * v2[j];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 120; j++) {
    
#pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L5}
    for (i = 0; i < 120; i++) {
      x[i] += beta * A[j][i] * y[j];
    }
  }
  
#pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L2}
  for (i = 0; i < 120; i++) {
    x[i] +=  z[i];
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (i = 0; i < 120; i++) {
    
#pragma ACCEL PARALLEL reduction=w FACTOR=auto{__PARA__L6}
    for (j = 0; j < 120; j++) {
      w[i] += alpha * A[i][j] * x[j];
    }
  }
  
#pragma endscop
}
// By applying loop tiling and loop permutation, we aim to optimize memory access patterns, improve parallelism, and reduce dependencies between iterations. This should lead to better performance in terms of trip count (TC), accumulated cycles (AC), and cycles per call (CPC) for High-Level Synthesis (HLS).