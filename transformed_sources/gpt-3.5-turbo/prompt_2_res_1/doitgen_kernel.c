// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the performance metrics such as accumulated cycles (AC) and cycles per call (CPC). 
// 
// Here is the transformed code with loop tiling and loop permutation:
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
// Rationale for the transformations:
// 1. Loop Permutation: By permuting the loops for `q` and `r`, we can improve data locality and cache efficiency, which can lead to reduced memory access times and improved performance.
// 2. Loop Tiling: By tiling the loops for `q` and `r`, we can exploit data reuse and improve parallelism, which can result in reduced accumulated cycles (AC) and cycles per call (CPC) for better performance.
// 
// These transformations aim to optimize the code for High-Level Synthesis (HLS) by leveraging loop tiling and loop permutation techniques to enhance performance metrics based on the provided performance estimates.