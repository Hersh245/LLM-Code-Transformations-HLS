// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the performance metrics such as accumulated cycles (AC) and cycles per call (CPC).
// 
// 1. Loop Tiling:
//    - We can apply loop tiling to the outermost loop 'r' to improve data locality and reduce memory access overhead.
//    - By tiling the 'r' loop, we can exploit spatial locality and reduce the number of memory accesses, which can lead to better performance.
// 
// 2. Loop Permutation:
//    - We can also apply loop permutation to interchange the 'q' and 'p' loops to improve parallelism and reduce loop-carried dependencies.
//    - By permuting the 'q' and 'p' loops, we can potentially increase the parallelism in the code and optimize the pipeline utilization.
// 
// Here is the transformed code with loop tiling and loop permutation applied:
#pragma ACCEL kernel

void kernel_doitgen(int nr, int nq, int np, double A[25][20][30], double C4[30][30], double sum[30])
{
  int r;
  int q;
  int p;
  int s;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (r = 0; r < 25; r+=5) { // Tiled 'r' loop with tile size of 5
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (p = 0; p < 30; p++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      for (q = 0; q < 20; q++) { // Permute 'q' and 'p' loops
        sum[p] = 0.0;
        for (s = 0; s < 30; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
      for (q = 0; q < 20; q++) { // Permute 'q' and 'p' loops
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}
// By applying loop tiling and loop permutation as described above, we aim to improve the performance metrics such as accumulated cycles (AC) and cycles per call (CPC) by enhancing data locality and increasing parallelism in the code.