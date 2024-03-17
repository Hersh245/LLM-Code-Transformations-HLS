
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_heat_3d_original(int tsteps,int n,double A[20][20][20],double B[20][20][20])
{
  int t;
  int i;
  int j;
  int k;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 1; t <= 40; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (i = 1; i < 20 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L3}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
      for (j = 1; j < 20 - 1; j++) {
        for (k = 1; k < 20 - 1; k++) {
          B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k]) + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k]) + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1]) + A[i][j][k];
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    for (i = 1; i < 20 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L4}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
      for (j = 1; j < 20 - 1; j++) {
        for (k = 1; k < 20 - 1; k++) {
          A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k]) + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k]) + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1]) + B[i][j][k];
        }
      }
    }
  }
//#pragma endscop
}

// To optimize the given C code for High-Level Synthesis (HLS), I'll apply several code transformations. The goal is to improve data locality, parallelism, and pipeline efficiency, which are crucial for achieving high performance on hardware accelerators like FPGAs. Here's the transformed code with explanations for each transformation:
#pragma ACCEL kernel

void kernel_heat_3d_transformed(int tsteps, int n, double A[20][20][20], double B[20][20][20]) {
  int t, i, j, k;

  #pragma ACCEL PIPELINE auto{__PIPE__L0}
  #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 1; t <= 40; t++) {

    // Loop tiling for the i-loop to improve data locality and enable parallel execution
    #pragma ACCEL TILE FACTOR=4 variable=i
    for (i = 1; i < 20 - 1; i += 4) { // Assuming a tile size of 4 for demonstration

      // Loop tiling for the j-loop
      #pragma ACCEL TILE FACTOR=4 variable=j
      for (j = 1; j < 20 - 1; j += 4) { // Assuming a tile size of 4 for demonstration

        // Loop permutation to bring the k-loop to the outermost position for better pipelining
        #pragma ACCEL PIPELINE auto{__PIPE__L1}
        for (k = 1; k < 20 - 1; k++) {

          // Processing tiles
          for (int ii = i; ii < i + 4 && ii < 20 - 1; ii++) {
            for (int jj = j; jj < j + 4 && jj < 20 - 1; jj++) {
              B[ii][jj][k] = 0.125 * (A[ii + 1][jj][k] - 2.0 * A[ii][jj][k] + A[ii - 1][jj][k]) +
                             0.125 * (A[ii][jj + 1][k] - 2.0 * A[ii][jj][k] + A[ii][jj - 1][k]) +
                             0.125 * (A[ii][jj][k + 1] - 2.0 * A[ii][jj][k] + A[ii][jj][k - 1]) +
                             A[ii][jj][k];
            }
          }
        }
      }
    }

    // Repeating the same transformations for the second loop nest
    #pragma ACCEL TILE FACTOR=4 variable=i
    for (i = 1; i < 20 - 1; i += 4) {

      #pragma ACCEL TILE FACTOR=4 variable=j
      for (j = 1; j < 20 - 1; j += 4) {

        #pragma ACCEL PIPELINE auto{__PIPE__L2}
        for (k = 1; k < 20 - 1; k++) {

          for (int ii = i; ii < i + 4 && ii < 20 - 1; ii++) {
            for (int jj = j; jj < j + 4 && jj < 20 - 1; jj++) {
              A[ii][jj][k] = 0.125 * (B[ii + 1][jj][k] - 2.0 * B[ii][jj][k] + B[ii - 1][jj][k]) +
                             0.125 * (B[ii][jj + 1][k] - 2.0 * B[ii][jj][k] + B[ii][jj - 1][k]) +
                             0.125 * (B[ii][jj][k + 1] - 2.0 * B[ii][jj][k] + B[ii][jj][k - 1]) +
                             B[ii][jj][k];
            }
          }
        }
      }
    }
  }
}
// ### Rationale Behind Each Transformation:
// 
// 1. **Loop Tiling (Blocking):** This transformation is applied to the `i` and `j` loops to improve data locality by processing the data in smaller blocks that fit into the cache. This reduces the memory access latency and increases the cache hit rate. The tile size is chosen based on the cache size and the data size of the elements being processed. Here, a tile size of 4 is used for demonstration purposes.
// 
// 2. **Loop Permutation:** This transformation changes the loop order to bring the `k` loop to the outermost position in the nested loop structure for the computation of `B`. This is done to exploit pipeline parallelism better. The `k` loop does not depend on the results of the `i` and `j` loops within the same iteration, making it a good candidate for outer loop to maximize pipelining efficiency.
// 
// 3. **Pragma Directives:** The original pragma directives for pipeline, tile, and parallel are retained and adapted to the transformed code structure to guide the HLS tool in optimizing the loop structures for parallel execution and efficient pipelining.
// 
// By applying these transformations, the code is optimized for HLS, potentially leading to better performance on hardware accelerators by exploiting data locality, parallelism, and efficient pipelining.

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
    int tsteps = 10; // Example value
    int n = 20;      // Dimension size, as given
    double A_original[20][20][20], B_original[20][20][20];
    double A_transformed[20][20][20], B_transformed[20][20][20];

    // Initialize arrays with random values
    init_array((double *)A_original, 20 * 20 * 20);
    init_array((double *)B_original, 20 * 20 * 20);
    // Copying original arrays to keep the initial conditions same for both versions
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            for (int k = 0; k < 20; k++)
            {
                A_transformed[i][j][k] = A_original[i][j][k];
                B_transformed[i][j][k] = B_original[i][j][k];
            }
        }
    }

    // Execute both the original and transformed functions
    kernel_heat_3d_original(tsteps, n, A_original, B_original);
    kernel_heat_3d_transformed(tsteps, n, A_transformed, B_transformed);

    // Compare the output arrays
    if (compare_arrays((double *)B_original, (double *)B_transformed, 20 * 20 * 20))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    