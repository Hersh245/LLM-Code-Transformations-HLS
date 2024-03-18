
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#pragma ACCEL kernel

void kernel_covariance_original(int m,int n,double float_n,double data[100][80],double cov[80][80],double mean[80])
{
  int i;
  int j;
  int k;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;
    
#pragma ACCEL PARALLEL reduction=mean FACTOR=auto{__PARA__L3}
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 100; i++) {
    
#pragma ACCEL PARALLEL reduction=data FACTOR=auto{__PARA__L4}
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (i = 0; i < 80; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    for (j = i; j < 80; j++) {
      cov[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction=cov FACTOR=auto{__PARA__L6}
      for (k = 0; k < 100; k++) {
        cov[i][j] += data[k][i] * data[k][j];
      }
      cov[i][j] /= float_n - 1.0;
      cov[j][i] = cov[i][j];
    }
  }
//#pragma endscop
}

// Given the performance estimates from the Merlin report, it's clear that the majority of the execution time is spent in the nested loops starting at lines 44, 47, and 51, which are responsible for calculating the covariance matrix. To optimize this code for High-Level Synthesis (HLS), we will apply several loop transformations focusing on these loops, while also considering the entire function for any potential improvements. The transformations will aim to reduce the accumulated cycles (AC) and cycles per call (CPC), thus improving the overall performance.
// 
// ### Optimized Code with Transformations
#pragma ACCEL kernel

void kernel_covariance_transformed(int m, int n, double float_n, double data[100][80], double cov[80][80], double mean[80]) {
  int i, j, k;

  // Original pragma directives are kept as requested
  #pragma ACCEL PIPELINE auto{__PIPE__L0}
  #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;

    #pragma ACCEL PARALLEL reduction=mean FACTOR=auto{__PARA__L3}
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }

  #pragma ACCEL PIPELINE auto{__PIPE__L1}
  #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 100; i++) {

    #pragma ACCEL PARALLEL reduction=data FACTOR=auto{__PARA__L4}
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
    }
  }

  // Loop permutation applied to improve data locality and cache usage
  #pragma ACCEL PIPELINE auto{__PIPE__L2}
  #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (j = 0; j < 80; j++) { // Outer loop changed to 'j'
    for (i = j; i < 80; i++) { // Inner loop changed to 'i'

      #pragma ACCEL PIPELINE auto{__PIPE__L5}
      cov[i][j] = 0.0;
      for (k = 0; k < 100; k++) {

        #pragma ACCEL PARALLEL reduction=cov FACTOR=auto{__PARA__L6}
        cov[i][j] += data[k][i] * data[k][j];
      }
      cov[i][j] /= float_n - 1.0;
      if (i != j) {
        cov[j][i] = cov[i][j]; // This line ensures symmetry is preserved
      }
    }
  }
}
// ### Rationale Behind Transformations
// 
// 1. **Loop Permutation**: The original code calculates the covariance matrix by iterating over rows and then columns. By permuting these loops (swapping the order of the `i` and `j` loops in the covariance calculation part), we aim to improve data locality. This is because accessing data elements in a row-major order (which is how C arrays are stored in memory) can be more cache-friendly, potentially reducing cache misses and thus improving performance.
// 
// 2. **Maintaining Pragmas**: All original pragmas were kept as requested. These pragmas are crucial for directing the HLS tool to apply specific optimizations like pipelining, tiling, and parallel processing. The `auto` factor allows the HLS tool to automatically determine the best factor for each directive based on the code structure and target architecture.
// 
// 3. **Symmetry Preservation**: After permuting the loops, it's important to ensure that the covariance matrix remains symmetric. This is achieved by explicitly setting `cov[j][i] = cov[i][j]` only when `i != j`. This avoids redundant computations and ensures the correctness of the output.
// 
// By applying these transformations, we aim to significantly reduce the accumulated cycles (AC) and cycles per call (CPC), especially in the computation-heavy part of the code that calculates the covariance matrix. These optimizations should lead to a more efficient implementation when synthesized for hardware with HLS.

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
    int m = 100, n = 80;
    double float_n = (double)n;
    double data_orig[100][80], data_trans[100][80], cov_original[80][80], cov_transformed[80][80], mean_original[80], mean_transformed[80];

    // Seed the random number generator for reproducibility
    srand(42);

    // Initialize data array with random numbers
    for (int i = 0; i < m; i++)
    {
        init_array(data_orig[i], n);
    }

    memcpy(data_trans, data_orig, sizeof(data_orig));

    // Randomly initialize output arrays
    init_array((double *)cov_original, 80 * 80);
    init_array((double *)cov_transformed, 80 * 80);
    init_array(mean_original, 80);
    init_array(mean_transformed, 80);

    // // Call both the original and the transformed function
    kernel_covariance_original(m, n, float_n, data_orig, cov_original, mean_original);
    kernel_covariance_transformed(m, n, float_n, data_trans, cov_transformed, mean_transformed);

    // Compare the covariance matrices
    if (compare_arrays((double *)cov_original, (double *)cov_transformed, 80 * 80) &&
        compare_arrays(mean_original, mean_transformed, 80))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    