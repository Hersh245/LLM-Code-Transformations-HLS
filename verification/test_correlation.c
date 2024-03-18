
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <math.h>
#pragma ACCEL kernel

void kernel_correlation_original(double float_n,double data[100][80],double corr[80][80],double mean[80],double stddev[80])
{
  int i;
  int j;
  int k;
  double eps = 0.1;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 80; j++) {
    stddev[j] = 0.0;
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (i = 0; i < 100; i++) {
      stddev[j] += pow(data[i][j] - mean[j],(double )2);
    }
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
/* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
    stddev[j] = (stddev[j] <= eps?1.0 : stddev[j]);
  }
/* Center and reduce the column vectors. */
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (i = 0; i < 100; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }
/* Calculate the m * m correlation matrix. */
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (i = 0; i < 80 - 1; i++) {
    corr[i][i] = 1.0;
    
#pragma ACCEL PIPELINE auto{__PIPE__L7}
    for (j = i + 1; j < 80; j++) {
      corr[i][j] = 0.0;
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7_0}
      for (k = 0; k < 100; k++) {
        corr[i][j] += data[k][i] * data[k][j];
      }
      corr[j][i] = corr[i][j];
    }
  }
  corr[80 - 1][80 - 1] = 1.0;
}

// Given the performance estimates and the provided code snippet, it's clear that the majority of the execution time is spent in the nested loops for calculating the correlation matrix (`loop i` at `cnn-krnl.cpp:68`). This section of the code accounts for approximately 96.5% of the total accumulated cycles. Therefore, optimizations should primarily focus on this part of the code.
// 
// However, it's important to consider the entire code for possible optimizations that might reduce the overall execution time. Here's the transformed code with explanations for each transformation applied:
#include <math.h>
#pragma ACCEL kernel

void kernel_correlation_transformed(double float_n, double data[100][80], double corr[80][80], double mean[80], double stddev[80]) {
  int i, j, k;
  double eps = 0.1;

  // No change here, as the original loop order seems optimal for memory access patterns.
#pragma ACCEL PIPELINE auto{__PIPE__L0}
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }

  // No change here as well, since the loop is already optimized for the given task.
#pragma ACCEL PIPELINE auto{__PIPE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 80; j++) {
    stddev[j] = 0.0;
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (i = 0; i < 100; i++) {
      stddev[j] += pow(data[i][j] - mean[j], (double)2);
    }
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
    stddev[j] = (stddev[j] <= eps ? 1.0 : stddev[j]);
  }

  // No change here, preprocessing data is necessary before the correlation calculation.
#pragma ACCEL PIPELINE auto{__PIPE__L2}
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (i = 0; i < 100; i++) {
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }

  // Loop interchange: This transformation is applied to improve data locality and cache usage.
  // The original code iterates over 'i' and 'j' in a way that might not be optimal for accessing 'data'.
  // By interchanging the loops, we can potentially improve the access pattern to 'data' and 'corr'.
#pragma ACCEL PIPELINE auto{__PIPE__L3}
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (j = 0; j < 80; j++) {
    corr[j][j] = 1.0;
    for (i = 0; i < j; i++) {
      corr[i][j] = 0.0;
#pragma ACCEL PIPELINE auto{__PIPE__L7}
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7_0}
      for (k = 0; k < 100; k++) {
        corr[i][j] += data[k][i] * data[k][j];
      }
      corr[j][i] = corr[i][j];
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Interchange in the Correlation Calculation:** The performance report indicates that the nested loops for calculating the correlation matrix are the most time-consuming part of the code. By interchanging the loops `i` and `j`, we aim to improve the spatial locality of the data accesses, which can lead to better cache utilization and potentially reduce the number of cache misses. This is because accessing `data[k][i]` and `data[k][j]` in the innermost loop with `j` as the outer loop might allow for more efficient use of loaded data when `k` is the same for consecutive iterations.
// 
// 2. **Preservation of Pragmas:** All original pragmas have been preserved and applied to the transformed loops accordingly. These pragmas are crucial for instructing the HLS tool on how to parallelize and optimize the loops for hardware implementation.
// 
// 3. **No Loop Tiling, Distribution, or Fusion Applied:** Given the structure of the loops and the operations within them, loop tiling, distribution, or fusion do not appear to be beneficial without further information on the hardware architecture or without causing significant changes to the algorithm that could impact correctness. The focus was on improving data locality through loop interchange.
// 
// This transformation aims to optimize the given code for HLS by focusing on the most time-consuming parts and attempting to improve data locality and cache utilization without altering the algorithm's correctness.

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
    srand(time(NULL)); // Seed the random number generator for reproducibility

    double data_orig[100][80], data_trans[100][80], corr_orig[80][80], corr_trans[80][80], mean[80], stddev[80];
    double float_n = 100.0; // Example value, adjust as necessary

    // Initialize data, mean, and stddev arrays with random values
    init_array(&data_orig[0][0], 100 * 80); // You need to pass the address of the first element
    init_array(mean, 80);
    init_array(stddev, 80);

    // Randomly initialize corr arrays to ensure they start with the same values
    init_array(&corr_orig[0][0], 80 * 80);
    memcpy(corr_trans, corr_orig, sizeof(corr_orig)); // Ensure both corr arrays start identically
    memcpy(data_trans, data_orig, sizeof(data_orig));

    // Call the original and transformed functions
    kernel_correlation_original(float_n, data_orig, corr_orig, mean, stddev);
    kernel_correlation_transformed(float_n, data_trans, corr_trans, mean, stddev);

    // Compare the corr arrays to check if both functions produce the same output
    if (compare_arrays(&corr_orig[0][0], &corr_trans[0][0], 80 * 80))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    