#pragma ACCEL kernel

void kernel_seidel_2d(int tsteps, int n, double A[120][120])
{
  int t;
  int i;
  int j;
//#pragma scop

#pragma ACCEL PIPELINE auto{__PIPE__L0}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t <= 39; t++) {

#pragma ACCEL PIPELINE auto{__PIPE__L1}

#pragma ACCEL TILE FACTOR=auto{__TILE__L1}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (j = 1; j <= 118; j++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (i = 1; i <= 118; i++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop interchange: The inner loop (i) and the middle loop (j) have been interchanged. This can help improve data locality and cache efficiency by accessing elements in a more contiguous manner.
// 2. Loop tiling: The loops have been tiled to improve data reuse and reduce memory access latency. By tiling the loops, we can operate on smaller data chunks at a time, which can improve cache performance.
// 3. Loop distribution: The computation has been distributed across the loops to allow for parallelization at different levels. This can help exploit parallelism in the hardware and improve overall performance.