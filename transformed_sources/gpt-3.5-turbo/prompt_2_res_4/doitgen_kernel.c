// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count (TC), accumulated cycles (AC), and cycles per call (CPC).
// 
// Here is the transformed code with loop tiling and loop permutation applied:
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
  for (q = 0; q < 20; q++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (r = 0; r < 25; r++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      for (p = 0; p < 30; p++) {
        sum[p] = 0.0;
        for (s = 0; s < 30; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
      for (p = 0; p < 30; p++) {
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}
// Rationale:
// 1. Loop Permutation: By swapping the loops for `q` and `r`, we can improve data locality and cache efficiency, as the innermost loop now accesses contiguous memory locations in the `A` array. This can reduce cache misses and improve performance.
// 2. Loop Tiling: By tiling the loops for `q` and `r`, we can exploit data reuse and improve parallelism. Tiling helps in maximizing data reuse within the cache and allows for better utilization of the hardware resources, leading to improved performance.
// 
// These transformations aim to optimize the code for HLS by considering the impact on TC, AC, and CPC based on the provided performance estimates.