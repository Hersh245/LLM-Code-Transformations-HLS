
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_seidel_2d_original(int tsteps,int n,double A[120][120])
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
    for (i = 1; i <= 118; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 1; j <= 118; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}

// To optimize the provided C code for High-Level Synthesis (HLS) using the performance estimates from the `merlin.rpt` file, we can apply several code transformations. The goal is to reduce the accumulated cycles (AC) and cycles per call (CPC) while maintaining or improving the trip count (TC) efficiency. Based on the performance estimates, the most time-consuming part is the nested loops, especially the outermost loop `t` and its nested loops `i` and `j`. We will focus on optimizing these loops.
// 
// ### Original Performance Bottlenecks:
// 
// 1. **Loop `t` has a high pipeline initiation interval (II) of 54**, which indicates that a new iteration can only start after 54 cycles, leading to inefficiency.
// 2. **Memory access for `A` (both read and write)** could be optimized to reduce the overhead.
// 
// ### Proposed Transformations:
// 
// 1. **Loop Tiling on `i` and `j` Loops**: This transformation can help in improving data locality and potentially reduce the memory access overhead for array `A`. By processing a smaller block of the array at a time, we can make better use of the cache, which can reduce the cycles spent on memory accesses.
// 
// 2. **Loop Permutation on `i` and `j` Loops**: This transformation can be considered to optimize memory access patterns, especially if loop tiling introduces non-contiguous memory accesses. However, due to the nature of the computation (which accesses neighboring elements), loop permutation might not yield significant benefits in this specific case but is worth exploring.
// 
// 3. **Maintain Original Pragmas**: The original pragmas for pipeline, tile, and parallel should be maintained to ensure that the compiler can still apply auto-optimizations based on these hints.
// 
// ### Transformed Code:
#pragma ACCEL kernel

void kernel_seidel_2d_transformed(int tsteps, int n, double A[120][120]) {
    int t, i, j, ii, jj;
    int TILE_SIZE = 10; // Example tile size, this can be tuned

    //#pragma scop

    #pragma ACCEL PIPELINE auto{__PIPE__L0}

    #pragma ACCEL TILE FACTOR=auto{__TILE__L0}

    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
    for (t = 0; t <= 39; t++) {

        #pragma ACCEL PIPELINE auto{__PIPE__L1}

        #pragma ACCEL TILE FACTOR=auto{__TILE__L1}

        #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
        for (ii = 1; ii <= 118; ii += TILE_SIZE) {
            for (jj = 1; jj <= 118; jj += TILE_SIZE) {
                for (i = ii; i < ii + TILE_SIZE && i <= 118; i++) {

                    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
                    for (j = jj; j < jj + TILE_SIZE && j <= 118; j++) {
                        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
                    }
                }
            }
        }
    }
    //#pragma endscop
}
// ### Rationale:
// 
// - **Loop Tiling**: By introducing `ii` and `jj` as the starting points for tiles within the `i` and `j` loops, we aim to improve cache utilization. The size of the tile (`TILE_SIZE`) is a parameter that can be tuned based on the specific architecture and cache sizes.
// - **Maintaining Original Pragmas**: The original pragmas are kept to ensure that the compiler's auto-optimization strategies are still applied. The transformed code structure should allow the compiler to more effectively apply these optimizations within the context of the tiled loops.
// 
// ### Conclusion:
// 
// This transformed code aims to address the identified performance bottlenecks by improving data locality through loop tiling. It's important to note that the actual performance gains would need to be verified through testing, as the optimal tile size and the effectiveness of these transformations can depend on the specific hardware architecture and compiler capabilities.

void init_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
    }
}


int compare_arrays(double *arr1, double *arr2, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(arr1[i] - arr2[i]) > 1e-6)
        {             // Using a tolerance to account for floating-point arithmetic differences
            return 0; // Arrays are not the same
        }
    }
    return 1; // Arrays are the same
}

int main()
{
    const int n = 120;
    const int tsteps = 10; // Example timesteps, adjust as needed
    double A_original[120][120];
    double A_transformed[120][120];

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize both matrices with the same values
    for (int i = 0; i < n; i++)
    {
        init_array(A_original[i], n);
        for (int j = 0; j < n; j++)
        {
            A_transformed[i][j] = A_original[i][j];
        }
    }

    // Execute both the original and transformed functions
    kernel_seidel_2d_original(tsteps, n, A_original);
    kernel_seidel_2d_transformed(tsteps, n, A_transformed);

    // Compare the resulting arrays
    if (compare_arrays((double *)A_original, (double *)A_transformed, n * n))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    