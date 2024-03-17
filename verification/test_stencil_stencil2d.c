
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

// To optimize the given C code for High-Level Synthesis (HLS) with the goal of improving parallelism, reducing latency, and increasing throughput, we can apply several code transformations. The transformations that seem most relevant to this code are loop unrolling and loop tiling. Loop unrolling can increase the degree of parallelism by allowing more operations to be executed in parallel. Loop tiling can help in improving data locality and reducing memory access times.
// 
// Given the structure of the original code, here's how it can be transformed:
// 
// 1. **Loop Unrolling**: This is particularly beneficial for the innermost loops where the bounds are small and known. Unrolling these loops can significantly reduce the loop control overhead and increase the opportunity for parallel execution of the multiplications and additions.
// 
// 2. **Loop Tiling**: Although the original code already includes tiling pragmas, we can explicitly define a tiling strategy for the outer loops to improve cache locality and possibly parallelize blocks of computation.
// 
// Here is the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_stencil_stencil2d_transformed(int orig[8192],int sol[8192],int filter[9])
{
  int r, c, k1, k2, temp, mul;

#pragma ACCEL PIPELINE auto{__PIPE__L0}
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}

  // Loop tiling for the outer loops is implicitly suggested by the original TILE pragmas.
  // Explicit tiling is not shown here but would involve dividing the loops into blocks.
  
  stencil_label1:
  for (r = 0; r < 126; r++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}

    stencil_label2:
    for (c = 0; c < 62; c++) {
      temp = 0;
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      stencil_label3:
      for (k1 = 0; k1 < 3; k1++) {
        // Unrolling the innermost loop manually
        // Assuming the compiler might not fully unroll this loop automatically
        mul = filter[k1 * 3] * orig[(r + k1) * 64 + c];
        temp += mul;
        
        mul = filter[k1 * 3 + 1] * orig[(r + k1) * 64 + c + 1];
        temp += mul;
        
        mul = filter[k1 * 3 + 2] * orig[(r + k1) * 64 + c + 2];
        temp += mul;
      }
      sol[r * 64 + c] = temp;
    }
  }
}
// ### Rationale Behind Transformations:
// 
// - **Loop Unrolling on the Innermost Loops**: This transformation was applied to the innermost loop (`k2` loop) to reduce loop overhead and increase parallelism. Since the loop bounds are small and fixed, unrolling these loops is a straightforward way to optimize the computation. This should allow the HLS tool to better schedule the operations and potentially parallelize the multiplications and additions.
// 
// - **Implicit Loop Tiling**: The original code includes pragmas for loop tiling (`#pragma ACCEL TILE FACTOR=auto{__TILE__L0}` and `#pragma ACCEL TILE FACTOR=auto{__TILE__L1}`). While the explicit tiling transformation is not shown in the transformed code, the rationale behind loop tiling is to break down the computation into smaller blocks that can fit into the faster levels of the memory hierarchy, thus improving data locality and potentially enabling parallel execution of these blocks. The `auto` factor suggests relying on the HLS tool to determine the optimal tiling factor, which is often the practical approach in complex applications.
// 
// - **Maintaining Other Pragmas**: The original pipeline and parallel factor pragmas are retained to hint at the HLS tool for automatic optimization opportunities. These pragmas suggest that the compiler should aim to pipeline the loops and parallelize them as much as possible, based on the available resources and the specified factors.
// 
// These transformations aim to strike a balance between manual optimization and relying on the HLS tool's automatic optimization capabilities, focusing on increasing parallelism and improving data locality.

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
    