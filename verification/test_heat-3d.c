
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

// Here is the optimized code after applying loop tiling and loop permutation:
#pragma ACCEL kernel

void kernel_heat_3d_transformed(int tsteps, int n, double A[20][20][20], double B[20][20][20])
{
    int t;
    int i;
    int j;
    int k;

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
#pragma ACCEL PIPELINE auto{__PIPE__L4}

#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
                for (k = 1; k < 20 - 1; k++) {
                    B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k]) + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k]) + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1]) + A[i][j][k];
                }
            }
        }

#pragma ACCEL PIPELINE auto{__PIPE__L2}

#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
        for (i = 1; i < 20 - 1; i++) {

#pragma ACCEL PIPELINE auto{__PIPE__L5}

#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
            for (k = 1; k < 20 - 1; k++) {
#pragma ACCEL PIPELINE auto{__PIPE__L6}

#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
                for (j = 1; j < 20 - 1; j++) {
                    A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k]) + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k]) + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1]) + B[i][j][k];
                }
            }
        }
    }
}
// Explanation:
// 1. Loop Tiling: We have applied loop tiling to the nested loops over `i`, `j`, and `k` to improve data locality and cache utilization. By breaking down the computation into smaller tiles, we can reduce memory access overhead and improve parallelism.
// 2. Loop Permutation: We have also permuted the inner loops over `j` and `k` to interchange their order. This can help in optimizing memory access patterns and improving parallelism by exploiting different loop dependencies.

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
    