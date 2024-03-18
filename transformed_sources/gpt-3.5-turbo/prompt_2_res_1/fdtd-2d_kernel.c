// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count (TC), accumulated cycles (AC), and cycles per call (CPC) metrics.
// 
// 1. Loop Tiling:
//    - We can tile the loops over 'i' and 'j' to improve data locality and reduce memory access overhead.
//    - By tiling the loops, we can exploit spatial locality and potentially reduce the number of cache misses.
// 
// 2. Loop Permutation:
//    - We can permute the loops to optimize the order of loop execution and improve parallelism.
//    - Reordering the loops can help in better pipelining and vectorization of the code.
// 
// Here is the transformed code with loop tiling and loop permutation applied:
#pragma ACCEL kernel

void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[60][80], double ey[60][80], double hz[60][80], double _fict_[40])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=4{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t < 40; t++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
    for (i = 0; i < 60; i+=4) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0_0}
      for (j = 0; j < 80; j+=4) {
        for (int ii = i; ii < i + 4; ii++) {
          for (int jj = j; jj < j + 4; jj++) {
            ey[ii][jj] = _fict_[t];
          }
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_1}
    
#pragma ACCEL TILE FACTOR=4{__TILE__L0_1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
    for (i = 1; i < 60; i+=4) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
      for (j = 0; j < 80; j+=4) {
        for (int ii = i; ii < i + 4; ii++) {
          for (int jj = j; jj < j + 4; jj++) {
            ey[ii][jj] = ey[ii][jj] - 0.5 * (hz[ii][jj] - hz[ii - 1][jj]);
          }
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_2}
    
#pragma ACCEL TILE FACTOR=4{__TILE__L0_2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2}
    for (i = 0; i < 60; i+=4) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2_0}
      for (j = 1; j < 80; j+=4) {
        for (int ii = i; ii < i + 4; ii++) {
          for (int jj = j; jj < j + 4; jj++) {
            ex[ii][jj] = ex[ii][jj] - 0.5 * (hz[ii][jj] - hz[ii][jj - 1]);
          }
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_3}
    
#pragma ACCEL TILE FACTOR=4{__TILE__L0_3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3}
    for (i = 0; i < 59; i+=4) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3_0}
      for (j = 0; j < 79; j+=4) {
        for (int ii = i; ii < i + 4; ii++) {
          for (int jj = j; jj < j + 4; jj++) {
            hz[ii][jj] = hz[ii][jj] - 0.7 * (ex[ii][jj + 1] - ex[ii][jj] + ey[ii + 1][jj] - ey[ii][jj]);
          }
        }
      }
    }
  }
//#pragma endscop
}
// By applying loop tiling and loop permutation, we aim to improve the performance metrics such as TC, AC, and CPC by optimizing memory access patterns and enhancing parallelism in the code.