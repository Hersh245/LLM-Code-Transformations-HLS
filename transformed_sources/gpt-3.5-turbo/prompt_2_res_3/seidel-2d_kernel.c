// Based on the performance estimate, the bottleneck in the code is the loop at line 25 (loop i) with a trip count of 118. To optimize the code for HLS, we can apply loop tiling to improve data locality and reduce memory access overhead.
// 
// Here is the transformed code with loop tiling applied:
#pragma ACCEL kernel

void kernel_seidel_2d(int tsteps,int n,double A[120][120])
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
    for (i = 1; i <= 118; i+=4) { // Apply loop tiling with a tile size of 4
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 1; j <= 118; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
        A[i+1][j] = (A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1] + A[i + 2][j - 1] + A[i + 2][j] + A[i + 2][j + 1]) / 9.0;
        A[i+2][j] = (A[i+1][j - 1] + A[i+1][j] + A[i+1][j + 1] + A[i + 2][j - 1] + A[i + 2][j] + A[i + 2][j + 1] + A[i + 3][j - 1] + A[i + 3][j] + A[i + 3][j + 1]) / 9.0;
        A[i+3][j] = (A[i+2][j - 1] + A[i+2][j] + A[i+2][j + 1] + A[i + 3][j - 1] + A[i + 3][j] + A[i + 3][j + 1] + A[i + 4][j - 1] + A[i + 4][j] + A[i + 4][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}
// In this transformed code, we applied loop tiling with a tile size of 4 for the loop at line 25 (loop i). By breaking down the loop iterations into smaller tiles, we can improve data reuse and reduce memory access overhead, leading to better performance in terms of accumulated cycles and cycles per call.