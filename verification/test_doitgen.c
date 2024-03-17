
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

// To optimize the provided C code for High-Level Synthesis (HLS), we can apply several code transformations. The goal is to improve parallelism, data locality, and resource utilization, which are crucial for achieving high performance on FPGA or ASIC designs. Below is the transformed code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_doitgen_transformed(int nr, int nq, int np, double A[25][20][30], double C4[30][30], double sum[30]) {
  int r, q, p, s;

  // Loop interchange: Moving the 'p' loop outside to improve data locality and enable better pipelining
  // This transformation helps in accessing continuous memory locations, improving memory access patterns.
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  for (p = 0; p < 30; p++) {
    for (r = 0; r < 25; r++) {
#pragma ACCEL PIPELINE auto{__PIPE__L1}
      for (q = 0; q < 20; q++) {
        sum[p] = 0.0;
        // Loop fusion: Fusing the computation and update of A[r][q][p] into a single loop
        // This reduces the loop overhead and improves the efficiency of the pipeline.
        for (s = 0; s < 30; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
        A[r][q][p] = sum[p];
      }
    }
  }
}
// ### Transformations Applied:
// 
// 1. **Loop Interchange**: The original code iterates over `r`, `q`, and then `p` in the outermost loops. By moving the `p` loop to the outermost position, we improve data locality since the inner loops (`r` and `q`) access continuous memory locations in `A[r][q][s]` and `C4[s][p]`. This transformation is beneficial for memory-bound operations as it minimizes cache misses and takes advantage of spatial locality in caches.
// 
// 2. **Loop Fusion**: The original code computes the `sum[p]` in one loop and then updates `A[r][q][p]` in another separate loop. By fusing these operations into a single loop, we reduce the overhead of loop control and improve the efficiency of pipeline execution. This transformation is particularly effective in this context because it allows for the computation and the update to be performed in a single pass, reducing the need for additional iterations over the data.
// 
// ### Rationale:
// 
// - **Improving Data Locality**: By reordering the loops and fusing computations, the transformed code accesses data in a more sequential manner, which is beneficial for cache utilization and reducing memory access times.
// 
// - **Enhancing Parallelism and Pipelining**: The use of `#pragma ACCEL PIPELINE` with the transformed loop structure allows for better exploitation of parallelism. The compiler can more effectively pipeline operations within and across iterations, leading to reduced execution times.
// 
// - **Reducing Loop Overhead**: Loop fusion minimizes the number of loop iterations and the associated control overhead, making the execution more efficient.
// 
// These transformations are aimed at optimizing the code for HLS, where the goal is to generate hardware that efficiently utilizes resources (such as registers and memory bandwidth) while maximizing parallel execution capabilities.

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
    