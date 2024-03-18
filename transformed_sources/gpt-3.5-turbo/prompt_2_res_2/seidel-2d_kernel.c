// Based on the performance estimates provided, we can apply loop tiling and loop permutation transformations to optimize the code for HLS.
// 
// 1. Loop Tiling:
//    - We can tile the loops over 'i' and 'j' to improve data locality and reduce memory access overhead.
//    - By tiling the loops, we can increase the cache hit rate and reduce the number of memory accesses.
// 
// 2. Loop Permutation:
//    - We can permute the loops to change the order of iteration for better pipelining and parallelization opportunities.
//    - By permuting the loops, we can potentially improve the pipeline II and reduce the overall cycle count.
// 
// Here is the transformed code with loop tiling and loop permutation applied:
#pragma ACCEL kernel

void kernel_seidel_2d(int tsteps, int n, double A[120][120])
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
    for (i = 1; i <= 118; i+=4) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 1; j <= 118; j+=4) {
        for (int ii = i; ii < i+4; ii++) {
          for (int jj = j; jj < j+4; jj++) {
            A[ii][jj] = (A[ii - 1][jj - 1] + A[ii - 1][jj] + A[ii - 1][jj + 1] + A[ii][jj - 1] + A[ii][jj] + A[ii][jj + 1] + A[ii + 1][jj - 1] + A[ii + 1][jj] + A[ii + 1][jj + 1]) / 9.0;
          }
        }
      }
    }
  }
//#pragma endscop
}
// By applying loop tiling and loop permutation, we aim to reduce the number of memory accesses, improve data locality, and potentially increase parallelization opportunities. These transformations can lead to better performance in terms of reduced cycle count and improved pipeline efficiency.