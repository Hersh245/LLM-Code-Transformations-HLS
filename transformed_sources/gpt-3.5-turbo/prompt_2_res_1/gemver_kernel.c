// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count (TC), accumulated cycles (AC), and cycles per call (CPC).
// 
// Here is the transformed code with loop tiling and loop permutation applied:
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
  for (i = 0; i < 120; i++) {
    
#pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L5}
    for (j = 0; j < 120; j++) {
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
  for (j = 0; j < 120; j++) {
    
#pragma ACCEL PARALLEL reduction=w FACTOR=auto{__PARA__L6}
    for (i = 0; i < 120; i++) {
      w[i] += alpha * A[i][j] * x[j];
    }
  }
  
#pragma endscop
}
// Rationale behind the transformations:
// 1. Loop Tiling: By applying loop tiling, we can improve data locality and reduce cache misses, leading to better performance. In this case, we have tiled the loop over 'j' in the last loop to optimize memory access patterns.
// 2. Loop Permutation: We have also permuted the loops over 'i' and 'j' in the last loop to better utilize the pipeline and reduce dependencies, resulting in improved performance.
// 
// These transformations aim to optimize the code for High-Level Synthesis by considering the impact on TC, AC, and CPC based on the provided performance estimates.