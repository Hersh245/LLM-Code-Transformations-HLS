
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_stencil_stencil2d_original(int orig[8192],int sol[8192],int filter[9])
{
  int r;
  int c;
  int k1;
  int k2;
  int temp;
  int mul;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  stencil_label1:
  for (r = 0; r < 128 - 2; r++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    stencil_label2:
    for (c = 0; c < 64 - 2; c++) {
      temp = ((int )0);
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      stencil_label3:
      for (k1 = 0; k1 < 3; k1++) {
        stencil_label4:
        for (k2 = 0; k2 < 3; k2++) {
          mul = filter[k1 * 3 + k2] * orig[(r + k1) * 64 + c + k2];
          temp += mul;
        }
      }
      sol[r * 64 + c] = temp;
    }
  }
}

// Given the performance estimates and the structure of the code, we can apply several optimizations. The primary goal is to reduce the accumulated cycles (AC) and improve the cycles per call (CPC), focusing on the innermost loops which are the most computationally intensive parts of the code. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_stencil_stencil2d_transformed(int orig[8192], int sol[8192], int filter[9]) {
  int r, c, k1, k2, temp, mul;

#pragma ACCEL PIPELINE auto{__PIPE__L0}
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}

  // Loop permutation applied to switch the order of the k1 and k2 loops
  // This is done to improve memory access patterns and locality
  stencil_label1:
  for (r = 0; r < 128 - 2; r++) {

#pragma ACCEL PIPELINE auto{__PIPE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}

    stencil_label2:
    for (c = 0; c < 64 - 2; c++) {
      temp = ((int)0);

#pragma ACCEL PIPELINE auto{__PIPE__L2}
      stencil_label3:
      for (k2 = 0; k2 < 3; k2++) { // Loop permutation: Moved k2 loop outside
        stencil_label4:
        for (k1 = 0; k1 < 3; k1++) { // Loop permutation: k1 loop is now inside
          mul = filter[k1 * 3 + k2] * orig[(r + k1) * 64 + c + k2];
          temp += mul;
        }
      }
      sol[r * 64 + c] = temp;
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Permutation**: The original code iterates over `k1` and then `k2` within the innermost loops. By permuting these loops (switching their order), we can potentially improve the spatial locality of the memory accesses. This is because accessing `orig[(r + k1) * 64 + c + k2]` in the order of `k2` then `k1` might result in accessing memory locations that are closer to each other, improving cache utilization. This transformation is expected to reduce the number of cache misses and, consequently, the accumulated cycles for memory access.
// 
// 2. **Preservation of Pragmas**: All original pragmas have been preserved to maintain directives for pipeline, tiling, and parallelization optimizations provided by the High-Level Synthesis (HLS) tool. These pragmas are crucial for instructing the HLS tool on how to optimize the loops for parallel execution and efficient memory access patterns.
// 
// 3. **Other Considerations**: Loop tiling, loop distribution, and loop fusion were considered. However, without specific details on the memory hierarchy and the exact nature of the computational bottleneck from the provided performance estimates, applying these transformations without empirical testing could lead to suboptimal results. The primary focus was on improving memory access patterns through loop permutation, which is generally a safe and effective optimization for nested loops accessing multi-dimensional arrays.
// 
// By applying these transformations, we aim to reduce the accumulated cycles (AC) and improve the cycles per call (CPC), especially focusing on optimizing memory access patterns which are often a significant bottleneck in computational kernels like this one.

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
    int orig[8192], sol_original[8192], sol_transformed[8192], filter[9];
    double orig_double[8192], sol_original_double[8192], sol_transformed_double[8192], filter_double[9];

    // Initialize double arrays for comparison
    init_array(orig_double, 8192);
    init_array(filter_double, 9);
    init_array(sol_original_double, 8192);

    // Cast double arrays to int for stencil functions
    for (int i = 0; i < 8192; i++)
    {
        orig[i] = (int)(orig_double[i] * 100);
        sol_original[i] = (int)(sol_original_double[i] * 100);
        sol_transformed[i] = (int)(sol_original_double[i] * 100);
        // printf("%f  %d\t", orig[i], orig[i]);
    }
    for (int i = 0; i < 9; i++)
    {
        filter[i] = (int)(filter_double[i] * 100.0);
        // printf("%f  %d\t", filter[i], filter[i]);
    }

    // Execute the original and transformed stencil functions
    kernel_stencil_stencil2d_original(orig, sol_original, filter);
    kernel_stencil_stencil2d_transformed(orig, sol_transformed, filter);

    // Cast int arrays back to double for comparison
    for (int i = 0; i < 8192; i++)
    {
        sol_original_double[i] = (double)(sol_original[i]);
        sol_transformed_double[i] = (double)(sol_transformed[i]);
        // printf("%f  %f\t", sol_original_double[i], sol_transformed_double[i]);
    }

    // Compare the output arrays
    if (compare_arrays(sol_original_double, sol_transformed_double, 8192))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    