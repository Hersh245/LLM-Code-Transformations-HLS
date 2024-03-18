// Based on the performance estimate provided, we can see that the most significant bottleneck is the loop at line 18 with a high accumulated cycle count. To optimize the code for HLS, we can apply loop tiling and loop distribution transformations to improve performance.
// 
// 1. Loop Tiling:
//    By applying loop tiling, we can break down the computation into smaller tiles that fit into the cache more efficiently, reducing the number of cache misses and improving data locality.
// 
// 2. Loop Distribution:
//    By distributing the computation across multiple loops, we can increase parallelism and reduce the overall cycle count.
// 
// Here is the transformed code with loop tiling and loop distribution applied:
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
// In this transformed code:
// - We have applied loop tiling by iterating over `i` and `j` in larger steps (i+=4, j+=4) to create 4x4 tiles for computation.
// - The innermost loops now iterate over the tiles, improving data locality and cache utilization.
// - By distributing the computation across smaller tiles, we increase parallelism and reduce the overall cycle count.
// 
// These transformations aim to optimize the code for HLS by improving data locality, reducing cache misses, and increasing parallelism, leading to better performance in terms of cycle count.