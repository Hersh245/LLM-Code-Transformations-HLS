
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_adi_original(int tsteps,int n,double u[60][60],double v[60][60],double p[60][60],double q[60][60])
{
  int t;
  int i;
  int j;
  double DX;
  double DY;
  double DT;
  double B1;
  double B2;
  double mul1;
  double mul2;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
//#pragma scop
  DX = 1.0 / ((double )60);
  DY = 1.0 / ((double )60);
  DT = 1.0 / ((double )40);
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 1; t <= 40; t++) {
//Column Sweep
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 1; i < 59; i++) {
      v[0][i] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = v[0][i];
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
      for (j = 1; j < 59; j++) {
        p[i][j] = -c / (a * p[i][j - 1] + b);
        q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
      }
      v[60 - 1][i] = 1.0;
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
      for (j = 0; j <= 57; j++) {
        int _in_j_0 = 58 + -1 * j;
        v[_in_j_0][i] = p[i][_in_j_0] * v[_in_j_0 + 1][i] + q[i][_in_j_0];
      }
      j = 1 + -1;
    }
//Row Sweep
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (i = 1; i < 59; i++) {
      u[i][0] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = u[i][0];
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (j = 1; j < 59; j++) {
        p[i][j] = -f / (d * p[i][j - 1] + e);
        q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
      }
      u[i][60 - 1] = 1.0;
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (j = 0; j <= 57; j++) {
        int _in_j = 58 + -1 * j;
        u[i][_in_j] = p[i][_in_j] * u[i][_in_j + 1] + q[i][_in_j];
      }
      j = 1 + -1;
    }
  }
//#pragma endscop
}

// To optimize the given C code for High-Level Synthesis (HLS), we can apply several code transformations. The main goals of these transformations are to improve data locality, parallelism, and pipeline efficiency. Here's how we can approach it:
// 
// 1. **Loop Tiling**: This transformation helps to improve cache locality by processing a small block of data at a time. However, due to the nature of the algorithm (Alternating Direction Implicit method), and the already small problem size (60x60), tiling might not offer significant benefits without further knowledge of the target architecture's memory hierarchy. For HLS, focusing on parallelism and pipelining is more crucial.
// 
// 2. **Loop Unrolling (Parallelism)**: This transformation is applied to enhance parallel execution. The `#pragma ACCEL PARALLEL` directive suggests that the compiler attempts to automatically unroll loops to exploit parallelism. Specifying a factor or allowing the compiler to choose ("auto") can significantly impact the performance, depending on the hardware capabilities.
// 
// 3. **Loop Pipelining**: The `#pragma ACCEL PIPELINE` directive is used to enable loop pipelining, a technique that allows iterations of a loop to overlap, thus potentially executing in parallel. This is crucial for HLS as it can significantly reduce the latency of loop execution.
// 
// Given the structure of the provided code, direct loop tiling might not be straightforward without altering the algorithm's correctness due to dependencies in the computation of `p` and `q`. However, ensuring that the compiler is aware of opportunities for parallelism and pipelining is key.
// 
// Below is the optimized code with explanations for each transformation:
#pragma ACCEL kernel

void kernel_adi_transformed(int tsteps, int n, double u[60][60], double v[60][60], double p[60][60], double q[60][60]) {
  int t, i, j;
  double DX = 1.0 / 60;
  double DY = 1.0 / 60;
  double DT = 1.0 / 40;
  double B1 = 2.0;
  double B2 = 1.0;
  double mul1 = B1 * DT / (DX * DX);
  double mul2 = B2 * DT / (DY * DY);
  double a = -mul1 / 2.0;
  double b = 1.0 + mul1;
  double c = a;
  double d = -mul2 / 2.0;
  double e = 1.0 + mul2;
  double f = d;

  // No changes above since setup calculations do not benefit from HLS optimizations
  
  // Main time-stepping loop
  for (t = 1; t <= 40; t++) {
    // Column Sweep
    
    // The PIPELINE directive is crucial for HLS, allowing loop iterations to overlap
    #pragma ACCEL PIPELINE auto
    for (i = 1; i < 59; i++) {
      v[0][i] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = v[0][i];
      
      // PARALLEL directive suggests potential parallel execution of the loop iterations
      #pragma ACCEL PARALLEL auto
      for (j = 1; j < 59; j++) {
        p[i][j] = -c / (a * p[i][j - 1] + b);
        q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
      }
      v[59][i] = 1.0;
      
      // Reverse loop for dependency resolution, parallelism here might be limited by dependencies
      #pragma ACCEL PARALLEL auto
      for (j = 58; j >= 1; j--) {
        v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
      }
    }
    
    // Row Sweep
    
    // Similar optimizations as for the column sweep
    #pragma ACCEL PIPELINE auto
    for (i = 1; i < 59; i++) {
      u[i][0] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = u[i][0];
      
      #pragma ACCEL PARALLEL auto
      for (j = 1; j < 59; j++) {
        p[i][j] = -f / (d * p[i][j - 1] + e);
        q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
      }
      u[i][59] = 1.0;
      
      #pragma ACCEL PARALLEL auto
      for (j = 58; j >= 1; j--) {
        u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
      }
    }
  }
}
// **Rationale Behind Transformations:**
// 
// - **Loop Pipelining and Parallelism**: These are the primary transformations applied. They instruct the HLS tool to look for opportunities to execute loop iterations in parallel or in an overlapped manner, which is particularly beneficial on FPGAs where operations can be highly parallelized.
//   
// - **Loop Tiling**: Not applied due to the small problem size and potential dependency issues. Tiling is more beneficial when working with larger datasets or when memory bandwidth is a bottleneck.
// 
// - **Loop Unrolling (Implicitly with Parallelism)**: The `#pragma ACCEL PARALLEL` directive can lead to loop unrolling by the HLS tool, depending on the specified factor or the tool's automatic decision-making.
// 
// This approach focuses on maximizing the hardware's capability to execute operations in parallel, which is a key advantage of FPGAs and similar architectures targeted by HLS.

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
    int n = 60, tsteps = 100;
    double u_original[60][60], v_original[60][60], p_original[60][60], q_original[60][60];
    double u_transformed[60][60], v_transformed[60][60], p_transformed[60][60], q_transformed[60][60];

    // Initialize the arrays with random values
    init_array(&u_original[0][0], n * n);
    init_array(&v_original[0][0], n * n);
    init_array(&p_original[0][0], n * n);
    init_array(&q_original[0][0], n * n);

    // Copy the initialized values to the arrays for the transformed function
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            u_transformed[i][j] = u_original[i][j];
            v_transformed[i][j] = v_original[i][j];
            p_transformed[i][j] = p_original[i][j];
            q_transformed[i][j] = q_original[i][j];
        }
    }

    // Run both versions of the kernel_adi function
    kernel_adi_original(tsteps, n, u_original, v_original, p_original, q_original);
    kernel_adi_transformed(tsteps, n, u_transformed, v_transformed, p_transformed, q_transformed);

    // Compare the output arrays
    if (compare_arrays(&u_original[0][0], &u_transformed[0][0], n * n) &&
        compare_arrays(&v_original[0][0], &v_transformed[0][0], n * n) &&
        compare_arrays(&p_original[0][0], &p_transformed[0][0], n * n) &&
        compare_arrays(&q_original[0][0], &q_transformed[0][0], n * n))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    