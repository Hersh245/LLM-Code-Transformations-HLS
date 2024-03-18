#pragma ACCEL kernel

void kernel_gemver(int n,double alpha,double beta,double A[120][120],double u1[120],double v1[120],double u2[120],double v2[120],double w[120],double x[120],double y[120],double z[120])
{
  int i;
  int j;
  
#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (i = 0; i < 120; i++) {
    
#pragma ACCEL PARALLEL reduction=A FACTOR=auto{__PARA__L4}
    for (j = 0; j < 120; j++) {
      A[i][j] += u1[i] * v1[j] + u2[i] * v2[j];
    }
    
#pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L5}
    for (j = 0; j < 120; j++) {
      x[i] += beta * A[j][i] * y[j];
    }
    
    x[i] +=  z[i];
    
#pragma ACCEL PARALLEL reduction=w FACTOR=auto{__PARA__L6}
    for (j = 0; j < 120; j++) {
      w[i] += alpha * A[i][j] * x[j];
    }
  }
  
#pragma endscop
}
// Explanation:
// 1. Loop Fusion: The original code had separate loops for updating the values of A, x, and w. By fusing these loops together, we reduce loop overhead and improve data locality.
// 2. Loop Permutation: The order of operations within the loop was rearranged to first update A, then x, and finally w. This can help in better utilizing the pipeline and reducing dependencies between loop iterations.
// 3. Loop Distribution: The operations within the loop were distributed to different parallel sections to take advantage of parallelism and optimize resource utilization. This helps in improving performance by executing multiple operations concurrently.