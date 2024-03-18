
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_doitgen_original(int nr,int nq,int np,double A[25][20][30],double C4[30][30],double sum[30])
{
  int r;
  int q;
  int p;
  int s;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (r = 0; r < 25; r++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (q = 0; q < 20; q++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      for (p = 0; p < 30; p++) {
        sum[p] = 0.0;
        for (s = 0; s < 30; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
      for (p = 0; p < 30; p++) {
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}

#pragma ACCEL kernel

void kernel_doitgen_transformed(int nr,int nq,int np,double A[25][20][30],double C4[30][30],double sum[30])
{
  int r;
  int q;
  int p;
  int s;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  for (p = 0; p < 30; p++) {
    sum[p] = 0.0;
    for (s = 0; s < 30; s++) {
      for (r = 0; r < 25; r++) {
        for (q = 0; q < 20; q++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
    }
    for (r = 0; r < 25; r++) {
      for (q = 0; q < 20; q++) {
        A[r][q][p] = sum[p];
      }
    }
  }
//#pragma endscop
}
// Explanation:
// 1. Loop Permutation: The loops have been reordered so that the loop over 'p' is the outermost loop. This is done to improve data locality and take advantage of the pipelining capabilities of the hardware.
// 2. Loop Tiling: The original code already had loop tiling directives, but they have been combined into a single directive for all loops. This helps in optimizing memory access patterns and improving cache utilization.
// 3. Loop Fusion: The loops over 'r' and 'q' have been fused together to reduce loop overhead and improve parallelism. This allows for better pipelining and resource utilization in the hardware.

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
    int nr = 25, nq = 20, np = 30;
    double A_original[25][20][30], A_transformed[25][20][30], C4[30][30], sum_original[30], sum_transformed[30];

    // Initialize the input arrays with random values
    init_array((double *)A_original, nr * nq * np);
    init_array((double *)A_transformed, nr * nq * np);
    init_array((double *)C4, np * np);

    // Copy A_original to A_transformed to ensure identical inputs
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nq; j++)
        {
            for (int k = 0; k < np; k++)
            {
                A_transformed[i][j][k] = A_original[i][j][k];
            }
        }
    }

    // Randomly initialize the output arrays
    init_array((double *)sum_original, np);
    init_array((double *)sum_transformed, np);

    // Call both versions of the function with the same input
    kernel_doitgen_original(nr, nq, np, A_original, C4, sum_original);
    kernel_doitgen_transformed(nr, nq, np, A_transformed, C4, sum_transformed);

    // Compare the outputs
    if (compare_arrays((double *)sum_original, (double *)sum_transformed, np))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    