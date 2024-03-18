// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the performance metrics such as trip count (TC), accumulated cycles (AC), and cycles per call (CPC).
// 
// 1. Loop Tiling:
//    - We can apply loop tiling to the outer loop (loop i) to improve data locality and reduce memory access overhead.
//    - By tiling the loop, we can process smaller chunks of data at a time, which can lead to better cache utilization and reduced memory access latency.
// 
// 2. Loop Permutation:
//    - We can also consider permuting the loops to optimize the data access patterns and improve parallelism.
//    - By permuting the loops, we can potentially expose more parallelism and reduce loop-carried dependencies.
// 
// Here is the transformed code with loop tiling and loop permutation applied:
#pragma ACCEL kernel

void kernel_syrk(double alpha, double beta, double C[80][80], double A[80][60])
{
  int i;
  int j;
  int k;

#pragma ACCEL PIPELINE auto{__PIPE__L0}

#pragma ACCEL TILE FACTOR=8{__TILE__L0}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i+=8) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=8{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 0; j < 80; j+=8) {
        for (int ii = i; ii < i+8; ii++) {
          for (int jj = j; jj < j+8; jj++) {
            if (jj <= ii) {
              C[ii][jj] *= beta;
            }
          }
        }

        for (int ii = i; ii < i+8; ii++) {
          for (int jj = j; jj < j+8; jj++) {
            if (jj <= ii) {
              C[ii][jj] += alpha * A[ii][k] * A[jj][k];
            }
          }
        }
      }
    }
  }
}
// By applying loop tiling and loop permutation, we aim to reduce the memory access overhead, improve data locality, and increase parallelism, leading to better performance in terms of TC, AC, and CPC. The transformed code is optimized for HLS to achieve better efficiency and resource utilization.