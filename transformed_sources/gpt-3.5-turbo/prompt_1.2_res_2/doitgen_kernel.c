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
  for (p = 0; p < 30; p++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (r = 0; r < 25; r++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      for (q = 0; q < 20; q++) {
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
// 1. Loop Permutation: The original code had loops over r, q, and p. By permuting the loops and iterating over p first, we can improve data locality and potentially increase parallelism.
// 2. Loop Tiling: Tiling is applied to the outermost loop over p. This helps in improving data reuse by breaking down the computation into smaller chunks that fit into cache more efficiently.
// 3. Loop Fusion: The original code had two separate loops for calculating the sum and updating the A matrix. By fusing these loops together, we reduce the number of loop iterations and improve memory access patterns.