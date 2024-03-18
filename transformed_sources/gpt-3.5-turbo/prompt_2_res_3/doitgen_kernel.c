// Based on the performance estimate, we can see that the majority of the accumulated cycles are spent in the loop over variable 'r'. To optimize the code for HLS, we can apply loop tiling to this loop to improve data locality and reduce memory access overhead.
// 
// Here is the transformed code with loop tiling applied to the 'r' loop:
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
  for (r = 0; r < 25; r+=5) { // Tiling factor of 5
    for (q = 0; q < 20; q++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L1}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
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
// By applying loop tiling with a factor of 5 to the 'r' loop, we divide the iteration space into smaller tiles, which can improve cache utilization and reduce memory access latency. This transformation aims to optimize the code for HLS by reducing the accumulated cycles spent in the 'r' loop.