#pragma ACCEL kernel

void kernel_doitgen(int nr,int nq,int np,double A[25][20][30],double C4[30][30],double sum[30])
{
  int r;
  int q;
  int p;
  int s;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  for (p = 0; p < 30; p++) {
    sum[p] = 0.0;
    for (s = 0; s < 30; s++) {
      for (r = 0; r < 25; r++) {
        for (q = 0; q < 20; q++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
    }
    for (r = 0; r < 25; r++) {
      for (q = 0; q < 20; q++) {
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop Permutation: The loops have been reordered so that the loop over 'p' is the outermost loop. This allows for better data locality and reduces memory access overhead.
// 2. Loop Tiling: The original code already had loop tiling directives, but they were applied separately for each loop. Here, the tiling factor has been applied to all loops at once to further optimize memory access patterns and improve parallelism.
// 3. Loop Fusion: The loops over 'r' and 'q' have been fused into the loop over 's' to reduce loop overhead and improve cache utilization. This also helps in better exploiting parallelism in the hardware.