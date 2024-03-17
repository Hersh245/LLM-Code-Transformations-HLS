
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_atax_original(int m,int n,double A[116][124],double x[124],double y[124],double tmp[116])
{
  int i;
  int j;
//#pragma scop
  for (i = 0; i < 124; i++) 
    y[i] = ((double )0);
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 116; i++) {
    tmp[i] = 0.0;
    
#pragma ACCEL PARALLEL reduction=tmp FACTOR=auto{__PARA__L0_0}
    for (j = 0; j < 124; j++) {
      tmp[i] += A[i][j] * x[j];
    }
    
#pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L0_1}
    for (j = 0; j < 124; j++) {
      y[j] += A[i][j] * tmp[i];
    }
  }
//#pragma endscop
}

// To optimize the given code for High-Level Synthesis (HLS) using code transformations such as loop permutation, loop tiling, loop distribution, or loop fusion, we need to consider the goals of reducing latency, increasing throughput, and improving data locality. Here's the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_atax_transformed(int m, int n, double A[116][124], double x[124], double y[124], double tmp[116]) {
  int i, j;
  // Initialize y to 0
  for (i = 0; i < 124; i++)
    y[i] = ((double)0);

  // Loop tiling to improve data locality and enable more efficient use of on-chip memory
  // Tile sizes are chosen based on typical FPGA on-chip memory sizes and access patterns
  // Assuming TILE_SIZE_I and TILE_SIZE_J are defined elsewhere according to the FPGA's characteristics
  #define TILE_SIZE_I 8
  #define TILE_SIZE_J 16

  #pragma ACCEL PIPELINE auto{__PIPE__L0}
  #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}

  for (int ii = 0; ii < 116; ii += TILE_SIZE_I) {
    for (int jj = 0; jj < 124; jj += TILE_SIZE_J) {
      // Loop tiling for tmp computation
      for (i = ii; i < ii + TILE_SIZE_I && i < 116; i++) {
        double sum_tmp = 0.0;

        #pragma ACCEL PARALLEL reduction=sum_tmp FACTOR=auto{__PARA__L0_0}
        for (j = jj; j < jj + TILE_SIZE_J && j < 124; j++) {
          sum_tmp += A[i][j] * x[j];
        }
        tmp[i] += sum_tmp;
      }

      // Loop tiling for y update
      for (i = ii; i < ii + TILE_SIZE_I && i < 116; i++) {
        #pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L0_1}
        for (j = jj; j < jj + TILE_SIZE_J && j < 124; j++) {
          y[j] += A[i][j] * tmp[i];
        }
      }
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Tiling (Blocking):** This transformation is applied to both the computation of `tmp` and the update of `y`. It helps in improving data locality by ensuring that the data used in computations is as close to the processor as possible, reducing memory access times. It also enables better utilization of on-chip memory, which is faster than accessing off-chip memory.
// 
// 2. **Loop Fusion:** The original code does not have an obvious opportunity for loop fusion without affecting the correctness of the program. Loop fusion is typically used to reduce the overhead of loop control and improve data locality by merging loops that have the same iteration space and are independent.
// 
// 3. **Loop Distribution:** This transformation was not explicitly applied here as the original code structure and the optimizations aimed at did not present a clear opportunity for it without affecting the correctness. Loop distribution splits a loop into multiple loops over the same iteration space but with different bodies, usually to separate concerns or to enable other optimizations like parallelization or vectorization on parts of the loop body.
// 
// 4. **Parallelism:** The use of `#pragma ACCEL PARALLEL` with reduction clauses and auto factor hints at the compiler to automatically parallelize these loops where possible, taking into account dependencies to ensure correctness. This aims to increase the throughput of the computation by exploiting data-level parallelism.
// 
// 5. **Pipeline:** The `#pragma ACCEL PIPELINE` directive is used to indicate that the loop iterations can be executed in a pipelined fashion, further increasing the throughput by overlapping the execution of different iterations.
// 
// The specific tile sizes (`TILE_SIZE_I` and `TILE_SIZE_J`) should be chosen based on the target FPGA's characteristics, such as the size of its on-chip memory and the optimal memory access patterns. These values are placeholders and should be defined elsewhere in the code or passed as parameters based on empirical performance tuning.

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
    // Seed the random number generator to get different results each run
    srand((unsigned)time(NULL));

    // Dimensions
    int m = 116, n = 124;

    // Declare and initialize arrays
    double A[116][124], x[124], y_original[124], y_transformed[124], tmp_original[116], tmp_transformed[116];

    // Initialize input arrays with random values
    for (int i = 0; i < m; i++)
    {
        init_array(A[i], n);
    }
    init_array(x, n);

    // Initialize output arrays with random values
    init_array(y_original, n);
    init_array(y_transformed, n);
    init_array(tmp_original, m);
    init_array(tmp_transformed, m);

    // Call the original function
    kernel_atax_original(m, n, A, x, y_original, tmp_original);
    // Call the transformed function
    kernel_atax_transformed(m, n, A, x, y_transformed, tmp_transformed);

    // Compare the outputs
    int y_compare = compare_arrays(y_original, y_transformed, n);
    int tmp_compare = compare_arrays(tmp_original, tmp_transformed, m);

    if (y_compare && tmp_compare)
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    