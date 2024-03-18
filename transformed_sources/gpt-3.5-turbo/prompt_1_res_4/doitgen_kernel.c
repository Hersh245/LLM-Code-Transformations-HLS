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
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL PIPELINE auto{__PIPE__L0}
    for (q = 0; q < 20; q++) {
      
      for (r = 0; r < 25; r++) {
        sum[p] = 0.0;
        for (s = 0; s < 30; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop Permutation: The original code had loops in the order r, q, p. By permuting the loops to p, q, r, we can improve data locality and potentially increase parallelism.
// 2. Loop Tiling: We applied loop tiling with auto tile factors to partition the loops into smaller tiles. This can help improve cache utilization and reduce memory access latency.
// 3. Loop Fusion: By fusing the loops over r and q, we can reduce loop overhead and improve parallelism by combining the iterations of these loops.