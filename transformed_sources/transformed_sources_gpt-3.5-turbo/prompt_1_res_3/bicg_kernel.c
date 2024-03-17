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
// Explanation of transformations:
// 1. Loop permutation: The inner and outer loops have been swapped to improve data locality and cache efficiency. This allows for better utilization of memory hierarchy.
// 2. Loop tiling: No tiling transformation was applied in this case as the code was already optimized for HLS and further tiling may not provide significant benefits.
// 3. Loop distribution: The loop distribution transformation was not applied as the code structure did not lend itself well to this transformation.
// 4. Loop fusion: The loops were fused to combine the computation of `s[j]` and `q[i]` into a single loop iteration. This reduces loop overhead and improves performance.