// Based on the provided performance estimates, we can optimize the code by applying loop tiling and loop permutation to improve the performance metrics such as accumulated cycles (AC) and cycles per call (CPC).
// 
// 1. Loop Tiling:
//    - By tiling the loops, we can improve data locality and reduce memory accesses, leading to better performance.
//    - We will tile the loops over the 'i' and 'j' dimensions.
// 
// 2. Loop Permutation:
//    - Reordering the loops can help in maximizing parallelism and reducing loop-carried dependencies.
//    - We will permute the loops to optimize the execution order.
// 
// Here is the transformed code with loop tiling and loop permutation:
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
    for (j = 0; j < 80; j+=8) {
      for (int ii = i; ii < i + 8; ii++) {
        for (int jj = j; jj < j + 8; jj++) {
          if (jj <= ii) {
            C[ii][jj] *= beta;
          }
        }
      }

#pragma ACCEL PIPELINE auto{__PIPE__L2}

#pragma ACCEL TILE FACTOR=8{__TILE__L2}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (k = 0; k < 60; k++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
        for (int ii = i; ii < i + 8; ii++) {
          for (int jj = j; jj < j + 8; jj++) {
            if (jj <= ii) {
              C[ii][jj] += alpha * A[ii][k] * A[jj][k];
            }
          }
        }
      }
    }
  }
}
// By applying loop tiling and loop permutation, we have optimized the code for better performance. The tiling factor of 8 was chosen based on the performance estimates to balance the trade-off between parallelism and data locality. The loop permutation ensures that the innermost loops are executed in a way that maximizes parallelism and minimizes loop-carried dependencies. These transformations should help reduce the accumulated cycles (AC) and cycles per call (CPC) for improved performance in High-Level Synthesis (HLS).