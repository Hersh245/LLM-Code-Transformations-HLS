
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

// Given the performance estimates and the structure of the code, it seems that the most time-consuming part is the nested loops. To optimize this code for High-Level Synthesis (HLS), we can apply several transformations. The primary goals are to reduce the loop latency and improve data access patterns for better utilization of the hardware resources. Here's how the code can be transformed:
// 
// 1. **Loop Fusion**: The two innermost loops over `p` can be merged. This reduces the overhead of loop control and potentially improves the locality of reference, which can lead to better performance.
// 
// 2. **Loop Interchange**: The loop over `s` is the innermost loop, which iterates over the second dimension of `A` and `C4`. Interchanging the loop `p` with the loop `s` might improve spatial locality since `C4` is accessed row-wise in the innermost loop.
// 
// 3. **Loop Tiling**: Although the original code has pragmas suggesting automatic tiling, explicitly specifying tiling factors based on the performance report and hardware characteristics might yield better performance. However, without specific details on the hardware and without running experiments, it's challenging to choose the optimal tiling factors. For demonstration, I'll assume generic tiling factors.
// 
// 4. **Maintain Original Pragmas**: The original pragmas for pipeline and tile should be maintained to ensure that the compiler's HLS directives are followed.
// 
// Here's the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_doitgen_transformed(int nr, int nq, int np, double A[25][20][30], double C4[30][30], double sum[30]) {
    int r, q, p, s;

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
                #pragma ACCEL PIPELINE
                for (s = 0; s < 30; s++) {
                    sum[p] += A[r][q][s] * C4[s][p];
                }
                A[r][q][p] = sum[p];
            }
        }
    }
}
// **Rationale Behind Transformations:**
// 
// - **Loop Fusion**: The two loops over `p` were merged to reduce the loop overhead and potentially improve data locality.
// - **Loop Interchange**: This transformation was considered but not applied in the provided code due to the complexity of determining its benefit without specific hardware details. However, it's a potential optimization point.
// - **Loop Tiling**: Explicit tiling was not added due to the lack of specific hardware details. The original pragmas suggest automatic tiling, which is left as is.
// - **Maintain Original Pragmas**: All original pragmas were maintained to ensure the HLS compiler can apply its optimizations based on the directives.
// 
// This transformed code aims to optimize the given code snippet based on the provided performance estimates and general optimization techniques for HLS. Further optimization might require detailed hardware specifications and iterative profiling to fine-tune the transformations.

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
    