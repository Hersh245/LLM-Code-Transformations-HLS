
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_bicg_original(int m,int n,double A[124][116],double s[116],double q[124],double p[116],double r[124])
{
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 116; i++) {
    s[i] = ((double )0);
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 124; i++) {
    q[i] = 0.0;
    
#pragma ACCEL PARALLEL reduction FACTOR=auto{__PARA__L2}
    for (j = 0; j < 116; j++) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  }
//#pragma endscop
}

// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. The key is to improve data locality, parallelism, and reduce memory access latency. Here's the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_bicg_transformed(int m, int n, double A[124][116], double s[116], double q[124], double p[116], double r[124]) {
  int i, j;
  // Initialize s and q arrays to 0
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 116; i++) {
    s[i] = ((double)0);
  }
  
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 124; i++) {
    q[i] = 0.0;
  }

  // Loop tiling for better cache locality and parallelism
  int TILE_SIZE = 32; // Example tile size, this can be adjusted based on the target architecture
  #pragma ACCEL TILE FACTOR=TILE_SIZE{__TILE__L1}
  for (int ii = 0; ii < 124; ii += TILE_SIZE) {
    int upper_i = ii + TILE_SIZE > 124 ? 124 : ii + TILE_SIZE;
    for (int jj = 0; jj < 116; jj += TILE_SIZE) {
      int upper_j = jj + TILE_SIZE > 116 ? 116 : jj + TILE_SIZE;
      
      // Loop distribution to separate the operations on s and q
      // This allows for independent optimization and parallel execution
      #pragma ACCEL PIPELINE auto{__PIPE__L1}
      for (i = ii; i < upper_i; i++) {
        for (j = jj; j < upper_j; j++) {
          s[j] += r[i] * A[i][j];
        }
      }
      
      #pragma ACCEL PIPELINE auto{__PIPE__L2}
      for (i = ii; i < upper_i; i++) {
        for (j = jj; j < upper_j; j++) {
          q[i] += A[i][j] * p[j];
        }
      }
    }
  }
}
// ### Rationale Behind Each Transformation:
// 
// 1. **Loop Tiling**: We applied loop tiling to both the `i` and `j` loops. This transformation helps in improving data locality by processing the matrix `A` in smaller blocks that fit into the cache. This reduces cache misses and can significantly improve performance, especially for large matrices.
// 
// 2. **Loop Distribution**: The original loop had two distinct operations on `s[j]` and `q[i]` within the same loop body. By distributing the loops, we separate these two operations. This allows for more focused optimization on each loop and potentially enables parallel execution of these independent loops. It also makes it clearer to the HLS tool how to pipeline these operations efficiently.
// 
// 3. **Parallelism and Pipelining**: The original pragmas for parallelism and pipelining are retained and appropriately placed to hint at the HLS tool for potential parallel execution and pipelining opportunities. Adjusting the parallel factor (`FACTOR=auto`) allows the HLS tool to automatically determine the optimal factor based on the target architecture and available resources.
// 
// 4. **Initialization Loops**: The initialization loops for `s` and `q` arrays are kept separate and before the main computation loops to ensure that all necessary data is ready before any computation begins. This also allows these initializations to potentially be parallelized or pipelined independently from the main computation.
// 
// By applying these transformations, the code is better structured for HLS tools to exploit parallelism, improve data locality, and optimize memory access patterns, leading to potentially higher performance on the target FPGA or ASIC.

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
    const int M = 124;
    const int N = 116;
    double A[M][N], p[N], r[M];
    double s_original[N], q_original[M];
    double s_transformed[N], q_transformed[M];

    srand(time(NULL)); // Seed for random number generation

    // Initialize arrays
    for (int i = 0; i < M; i++)
    {
        init_array(A[i], N);
    }
    init_array(p, N);
    init_array(r, M);

    // Initialize output arrays with zeros
    init_array(s_original, N);
    init_array(q_original, M);
    init_array(s_transformed, N);
    init_array(q_transformed, M);

    // Run both versions of the kernel_bicg function
    kernel_bicg_original(M, N, A, s_original, q_original, p, r);
    kernel_bicg_transformed(M, N, A, s_transformed, q_transformed, p, r);

    // Compare the output arrays
    if (compare_arrays(s_original, s_transformed, N) && compare_arrays(q_original, q_transformed, M))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    