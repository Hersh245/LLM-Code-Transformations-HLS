
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#pragma ACCEL kernel

void kernel_syrk_original(double alpha,double beta,double C[80][80],double A[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}

// Based on the performance estimate provided, the most time-consuming part of the code is within the `loop i`, specifically the nested loops `loop j` inside `loop i` and `loop k` inside `loop i`. The `loop k` and its nested `loop j` are particularly expensive, consuming 97.2% of the execution time. To optimize this code for High-Level Synthesis (HLS), we can apply several loop transformations. Here's an optimized version of the code with explanations for each transformation:
#pragma ACCEL kernel

void kernel_syrk_transformed(double alpha, double beta, double C[80][80], double A[80][60]) {
  int i, j, k;

#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i++) {
    // Loop distribution applied here to separate the updates of C[i][j] *= beta
    // from the updates of C[i][j] += alpha * A[i][k] * A[j][k]
    // This allows for better pipelining and parallelization opportunities.
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (j = 0; j <= i; j++) {
      C[i][j] *= beta;
    }
    
    // Loop tiling applied to the computation of C[i][j] += alpha * A[i][k] * A[j][k]
    // to improve data locality and cache utilization.
    // Assuming a tile size of T for both j and k loops.
    int T = 20; // Example tile size, adjust based on the target architecture and cache sizes.
    for (int jj = 0; jj <= i; jj += T) {
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (k = 0; k < 60; k++) {
        for (int j = jj; j < ((jj + T) > (i + 1) ? (i + 1) : (jj + T)); j++) {
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Distribution**: The original code updates `C[i][j]` with `beta` and then immediately updates it again within the same `i` loop iteration based on `alpha`, `A[i][k]`, and `A[j][k]`. By distributing these loops, we separate concerns, allowing for more focused optimizations on each part. This also helps in reducing the dependency between different parts of the loop body, potentially improving parallelism.
// 
// 2. **Loop Tiling**: Applied to the computation of `C[i][j] += alpha * A[i][k] * A[j][k]` to improve data locality. By working on smaller sub-matrices (tiles) at a time, we can ensure that the data used by these computations is more likely to be in cache, reducing memory access times. The tile size `T` is a parameter that should be tuned based on the specific architecture's cache size and other characteristics for optimal performance.
// 
// 3. **Loop Permutation**: Although not explicitly applied in the transformed code above, loop permutation could be considered if it were identified that swapping the loop order could lead to better performance, for instance, by improving data access patterns. However, in this specific case, the primary focus was on loop distribution and tiling to address the identified bottlenecks.
// 
// By applying these transformations, we aim to reduce the overall accumulated cycles (AC) and cycles per call (CPC), improving the performance of the `kernel_syrk` function for High-Level Synthesis (HLS).

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
    double A[80][60], C_original[80][80], C_transformed[80][80];
    double alpha = 1.2, beta = 0.8;

    // Initialize matrices A, C_original, and C_transformed with random values
    init_array(&A[0][0], 80 * 60);
    init_array(&C_original[0][0], 80 * 80);
    // Copy C_original to C_transformed to ensure they start with the same values
    for (int i = 0; i < 80; i++)
    {
        for (int j = 0; j < 80; j++)
        {
            C_transformed[i][j] = C_original[i][j];
        }
    }

    // Run both kernel versions
    kernel_syrk_original(alpha, beta, C_original, A);
    kernel_syrk_transformed(alpha, beta, C_transformed, A);

    // Compare the output matrices
    if (compare_arrays(&C_original[0][0], &C_transformed[0][0], 80 * 80))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    