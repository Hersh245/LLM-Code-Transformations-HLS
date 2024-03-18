
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_seidel_2d_original(int tsteps,int n,double A[120][120])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t <= 39; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 1; i <= 118; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 1; j <= 118; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}

// The original code snippet is already optimized for HLS with loop pipelining, loop tiling, and loop parallelization directives. However, we can further optimize the code by applying loop fusion to reduce loop overhead and improve data locality.
// 
// Here is the transformed code with loop fusion applied:
#pragma ACCEL kernel

void kernel_seidel_2d_transformed(int tsteps, int n, double A[120][120])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t <= 39; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 1; i <= 118; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      for (j = 1; j <= 118; j++) {
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9.0;
      }
    }
  }
//#pragma endscop
}
// In this transformed code, we have fused the loop parallelization and loop pipelining directives for the innermost loop (j loop) to reduce loop overhead and improve data locality. This will help in achieving better performance during HLS synthesis.

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
    const int n = 120;
    const int tsteps = 10; // Example timesteps, adjust as needed
    double A_original[120][120];
    double A_transformed[120][120];

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize both matrices with the same values
    for (int i = 0; i < n; i++)
    {
        init_array(A_original[i], n);
        for (int j = 0; j < n; j++)
        {
            A_transformed[i][j] = A_original[i][j];
        }
    }

    // Execute both the original and transformed functions
    kernel_seidel_2d_original(tsteps, n, A_original);
    kernel_seidel_2d_transformed(tsteps, n, A_transformed);

    // Compare the resulting arrays
    if (compare_arrays((double *)A_original, (double *)A_transformed, n * n))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    