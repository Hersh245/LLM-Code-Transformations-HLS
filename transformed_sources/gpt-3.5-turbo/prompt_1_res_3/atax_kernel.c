#pragma ACCEL kernel

void kernel_atax(int m,int n,double A[116][124],double x[124],double y[124],double tmp[116])
{
  int i;
  int j;
//#pragma scop
  for (i = 0; i < 124; i++) 
    y[i] = ((double )0);
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 124; j++) {
    tmp[j] = 0.0;
    
#pragma ACCEL PARALLEL reduction=tmp FACTOR=auto{__PARA__L0_0}
    for (i = 0; i < 116; i++) {
      tmp[j] += A[i][j] * x[j];
    }
  }
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 116; i++) {
    
#pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L1_0}
    for (j = 0; j < 124; j++) {
      y[j] += A[i][j] * tmp[j];
    }
  }
//#pragma endscop
}
// Explanation of transformations:
// 1. Loop interchange: The loops for `i` and `j` have been interchanged to improve data locality and access patterns. This can potentially reduce memory access latency and improve performance.
// 2. Loop distribution: The computation of `tmp` values has been separated into its own loop, which allows for better parallelism and pipelining opportunities. This can help in maximizing resource utilization and improving throughput.
// 3. Loop fusion: The loops for updating `y` values have been fused together to reduce loop overhead and improve parallelism. This can lead to better performance by reducing synchronization and communication overhead.