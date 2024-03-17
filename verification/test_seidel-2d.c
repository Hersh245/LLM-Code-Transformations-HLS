
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

// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. The goal is to improve parallelism, data locality, and resource utilization. Here's an approach using loop tiling and loop permutation:
// 
// 1. **Loop Tiling**: This technique breaks down the loops into smaller chunks or "tiles" to improve cache locality and enable parallel execution of these chunks. It's particularly effective for nested loops operating over large data sets.
// 
// 2. **Loop Permutation**: Changing the loop order can sometimes improve data locality and access patterns, which can be beneficial for performance, especially in memory-bound applications.
// 
// Given the original code, we'll apply loop tiling to the `i` and `j` loops to handle the data in smaller blocks. This can help in fitting the data used by the computation into faster levels of memory, reducing the latency of memory accesses. Loop permutation does not seem immediately beneficial in this context since the computation pattern and dependencies are symmetric with respect to `i` and `j`. However, in different contexts, changing the order of `i` and `j` could be considered if it aligns better with data access patterns.
// 
// Here's the transformed code with loop tiling applied:
#pragma ACCEL kernel

void kernel_seidel_2d_transformed(int tsteps, int n, double A[120][120]) {
  int t, i, j, ii, jj;
  const int TILE_SIZE = 10; // Example tile size, can be adjusted based on the target architecture and memory hierarchy

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
        
        #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
        for (i = ii; i < ii + TILE_SIZE && i <= 118; i++) {
          for (j = jj; j < jj + TILE_SIZE && j <= 118; j++) {
            A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
          }
        }
      }
    }
  }
  //#pragma endscop
}
// **Rationale Behind Transformations:**
// 
// - **Loop Tiling**: By applying loop tiling, we've broken down the computation into smaller blocks. This can significantly improve the cache hit rate as the working set of the data fits better into the cache. It also opens up opportunities for parallel execution of these smaller blocks.
// 
// - **Choice of TILE_SIZE**: The tile size of `10` is arbitrary and should be tuned based on the specific architecture's cache sizes and the characteristics of the memory hierarchy. The goal is to choose a size that balances the benefits of temporal and spatial locality with the overhead of managing smaller blocks of computation.
// 
// - **Parallelism and Pipelining**: The original pragmas for parallelism and pipelining are retained to hint at the compiler for HLS that these loops are candidates for parallel execution and pipelining optimizations. The effectiveness of these hints will depend on the capabilities of the HLS tool and the target architecture.
// 
// This transformed code aims to leverage data locality and enable more efficient use of computational resources through loop tiling, while maintaining the potential for parallel execution and pipelining as directed by the pragmas.

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
    