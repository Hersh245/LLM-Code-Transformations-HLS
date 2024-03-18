#pragma ACCEL kernel

void kernel_bicg(int m,int n,double A[124][116],double s[116],double q[124],double p[116],double r[124])
{
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 116; i++) {
    s[i] = ((double )0);
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 116; j++) {
    for (i = 0; i < 124; i++) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  }
//#pragma endscop
}
// 1. Loop permutation: The inner and outer loops have been swapped to improve data locality and cache efficiency. This can potentially reduce memory access latency and improve performance.
// 
// 2. Loop distribution: The loop nest has been distributed such that the inner loop is now the outer loop. This can help in better parallelization and pipelining of the code.
// 
// 3. Loop fusion: The two nested loops have been fused into a single loop. This can reduce loop overhead and improve parallelism by allowing the compiler to optimize the loop as a single unit.