#pragma ACCEL kernel

void kernel_bicg(int m, int n, double A[124][116], double s[116], double q[124], double p[116], double r[124])
{
  int i;
  int j;
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
#pragma ACCEL PIPELINE auto{__PIPE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (j = 0; j < 116; j++) {
    for (i = 0; i < 124; i++) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  }
}
// Explanation:
// 1. Loop Permutation: The inner loop (previously iterating over `j`) has been moved outside the outer loop (previously iterating over `i`). This helps in better data locality and improves parallelism.
// 2. Loop Tiling: The pragma `#pragma ACCEL TILE FACTOR=auto{__TILE__L1}` has been added to enable loop tiling, which partitions the loops into smaller tiles to improve cache utilization and reduce memory access latency.
// 3. Loop Distribution: The pragma `#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}` has been added to distribute the computation of `s[j]` and `q[i]` across multiple processing elements, improving parallelism and performance.
// 4. Loop Fusion: By combining the two inner loops into a single loop, we reduce the overhead of loop control and improve the efficiency of the computation.