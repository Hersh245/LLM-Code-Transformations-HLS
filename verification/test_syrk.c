
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

// To optimize the provided C code for High-Level Synthesis (HLS) using code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion, we need to consider the memory access patterns, parallelism, and data reuse. The goal is to enhance data locality, reduce memory access latency, and increase the parallel execution capability of the code.
// 
// Here's the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_syrk_transformed(double alpha, double beta, double C[80][80], double A[80][60]) {
    int i, j, k;

    // Loop tiling is applied to enhance data locality and enable more efficient use of on-chip memory.
    // Tiling factors are chosen based on typical FPGA on-chip memory sizes and access patterns.
    // Here, we assume 20 as an arbitrary tiling factor for demonstration purposes.
    int TILE_SIZE_I = 20;
    int TILE_SIZE_J = 20;
    int TILE_SIZE_K = 20;

    // Loop distribution is applied to separate the two distinct operations on C[i][j].
    // This allows for better pipelining and parallelization opportunities.

    // First operation: C[i][j] *= beta;
    #pragma ACCEL PIPELINE auto
    for (int ii = 0; ii < 80; ii += TILE_SIZE_I) {
        for (int jj = 0; jj < 80; jj += TILE_SIZE_J) {
            #pragma ACCEL PARALLEL FACTOR=auto
            for (i = ii; i < ii + TILE_SIZE_I && i < 80; i++) {
                for (j = jj; j < jj + TILE_SIZE_J && j < 80; j++) {
                    if (j <= i) {
                        C[i][j] *= beta;
                    }
                }
            }
        }
    }

    // Second operation: C[i][j] += alpha * A[i][k] * A[j][k];
    #pragma ACCEL PIPELINE auto
    for (int ii = 0; ii < 80; ii += TILE_SIZE_I) {
        for (int jj = 0; jj < 80; jj += TILE_SIZE_J) {
            for (int kk = 0; kk < 60; kk += TILE_SIZE_K) {
                #pragma ACCEL PARALLEL FACTOR=auto
                for (i = ii; i < ii + TILE_SIZE_I && i < 80; i++) {
                    for (k = kk; k < kk + TILE_SIZE_K && k < 60; k++) {
                        for (j = jj; j < jj + TILE_SIZE_J && j < 80; j++) {
                            if (j <= i) {
                                C[i][j] += alpha * A[i][k] * A[j][k];
                            }
                        }
                    }
                }
            }
        }
    }
}
// ### Rationale Behind Each Transformation:
// 
// 1. **Loop Tiling**: This transformation is applied to both the `i-j` loops and the `i-k-j` loops. The main goal is to improve data locality by working on smaller, more manageable chunks of data that fit into the FPGA's on-chip memory (e.g., Block RAMs or BRAMs). This reduces the need for frequent off-chip memory accesses, which are slower and can become a bottleneck.
// 
// 2. **Loop Distribution**: The original code performs two distinct operations on `C[i][j]` within the same loop nest: scaling by `beta` and updating with the `alpha * A[i][k] * A[j][k]` term. By distributing these into separate loop nests, each operation can be optimized independently. This also allows for better exploitation of parallelism, as the compiler/HLS tool can more easily identify opportunities for pipelining and parallel execution within each operation.
// 
// 3. **Loop Permutation**: Although not explicitly applied in the transformed code above, loop permutation could be considered based on the memory access patterns and the hardware architecture. For instance, changing the loop order could further optimize memory access patterns or parallelism, depending on the specific details of the FPGA architecture and memory hierarchy.
// 
// 4. **Pragma Directives**: The `#pragma ACCEL` directives are retained and applied to the transformed code to hint to the HLS tool about potential optimization opportunities. These directives suggest where pipelining, parallel execution, and tiling could be beneficial, but the exact factors and strategies are left as "auto" for the HLS tool to decide based on its analysis of the code and the target architecture.
// 
// By applying these transformations, the code is better structured for parallel execution and efficient memory access, which are crucial for achieving high performance on FPGA-based systems.

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
    