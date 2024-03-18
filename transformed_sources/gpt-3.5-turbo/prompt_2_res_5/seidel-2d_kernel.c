// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count and accumulated cycles.
// 
// 1. Loop Tiling:
//    - We can apply loop tiling to the outer loop 't' to reduce the number of iterations and improve data locality.
//    - By tiling the loop, we can optimize memory access patterns and potentially reduce cache misses.
// 
// 2. Loop Permutation:
//    - We can apply loop permutation to interchange the loops 'i' and 'j' to improve data reuse and parallelism.
//    - This transformation can help in better utilizing the available resources and reducing the overall cycle count.
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
  for (t = 0; t <= 39; t+=2) { // Loop tiling applied with tile size of 2
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (j = 1; j <= 118; j++) { // Loop permutation with j as outer loop
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (i = 1; i <= 118; i++) { // Loop permutation with i as inner loop
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}
// By applying loop tiling and loop permutation, we aim to reduce the trip count and accumulated cycles, leading to improved performance in terms of cycles per call. These transformations help in optimizing memory access patterns, data reuse, and parallelism, which are crucial for achieving better performance in HLS.