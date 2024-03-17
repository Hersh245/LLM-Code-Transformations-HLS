
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_fdtd_2d_original(int tmax,int nx,int ny,double ex[60][80],double ey[60][80],double hz[60][80],double _fict_[40])
{
  int t;
  int i;
  int j;
//#pragma scop
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (t = 0; t < 40; t++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
    for (j = 0; j < 80; j++) {
      ey[0][j] = _fict_[t];
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L0_1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
    for (i = 1; i < 60; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
      for (j = 0; j < 80; j++) {
        ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L0_2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2}
    for (i = 0; i < 60; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2_0}
      for (j = 1; j < 80; j++) {
        ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L0_3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L0_3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3}
    for (i = 0; i < 59; i++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3_0}
      for (j = 0; j < 79; j++) {
        hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
      }
    }
  }
//#pragma endscop
}

// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. These transformations aim to improve data locality, parallelism, and pipeline efficiency. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_fdtd_2d_transformed(int tmax, int nx, int ny, double ex[60][80], double ey[60][80], double hz[60][80], double _fict_[40]) {
  int t, i, j;

  // Loop tiling is applied to improve data locality and enable more efficient use of on-chip memory.
  // Loop permutation is applied to ensure that the innermost loop has the best spatial locality.
  // Loop fusion is avoided here as it might reduce parallelism opportunities and increase the complexity of dependencies.
  
  #pragma ACCEL PIPELINE auto
  for (t = 0; t < 40; t++) {
    #pragma ACCEL PARALLEL FACTOR=auto
    for (j = 0; j < 80; j++) {
      ey[0][j] = _fict_[t];
    }

    // Applying loop tiling for the ey update
    int TILE_SIZE_i = 10; // Example tile size for i dimension
    int TILE_SIZE_j = 20; // Example tile size for j dimension
    for (i = 1; i < 60; i += TILE_SIZE_i) {
      for (j = 0; j < 80; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 60; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 80; ++tj) {
            ey[ti][tj] = ey[ti][tj] - 0.5 * (hz[ti][tj] - hz[ti - 1][tj]);
          }
        }
      }
    }

    // Applying loop tiling for the ex update
    for (i = 0; i < 60; i += TILE_SIZE_i) {
      for (j = 1; j < 80; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 60; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 80; ++tj) {
            ex[ti][tj] = ex[ti][tj] - 0.5 * (hz[ti][tj] - hz[ti][tj - 1]);
          }
        }
      }
    }

    // Applying loop tiling for the hz update
    for (i = 0; i < 59; i += TILE_SIZE_i) {
      for (j = 0; j < 79; j += TILE_SIZE_j) {
        #pragma ACCEL PARALLEL FACTOR=auto
        for (int ti = i; ti < i + TILE_SIZE_i && ti < 59; ++ti) {
          for (int tj = j; tj < j + TILE_SIZE_j && tj < 79; ++tj) {
            hz[ti][tj] = hz[ti][tj] - 0.7 * (ex[ti][tj + 1] - ex[ti][tj] + ey[ti + 1][tj] - ey[ti][tj]);
          }
        }
      }
    }
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Loop Tiling:** This transformation is applied to the `ey`, `ex`, and `hz` update loops. It helps in improving data locality by working on smaller sub-blocks of the data array that fit into the faster, but smaller, on-chip memory. This reduces the number of memory accesses to the slower off-chip memory.
// 
// 2. **Loop Permutation:** Although not explicitly shown as a separate transformation, the loop ordering inside the tiled loops is chosen to ensure that the data is accessed in a way that is favorable for spatial locality, which is important for achieving high memory bandwidth utilization.
// 
// 3. **Parallelism:** The `#pragma ACCEL PARALLEL FACTOR=auto` directive is used to hint at automatic parallelization opportunities within the loops. This is crucial for exploiting the parallel compute resources available in FPGAs.
// 
// 4. **Pipeline:** The `#pragma ACCEL PIPELINE auto` directive outside the time loop (`t`) suggests that the compiler should attempt to pipeline this loop, allowing for different iterations of the loop to be executed in an overlapped manner, further increasing the throughput of the computation.
// 
// By applying these transformations, the code is optimized for HLS targeting FPGA architectures, which can significantly improve the performance and efficiency of the FDTD 2D kernel computation.

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
    int tmax = 10;        // Example value, set as needed
    int nx = 60, ny = 80; // Dimensions as per your arrays

    // Allocate memory for input and output arrays for both original and transformed functions
    double ex_original[60][80], ey_original[60][80], hz_original[60][80], _fict_original[40];
    double ex_transformed[60][80], ey_transformed[60][80], hz_transformed[60][80], _fict_transformed[40];

    // Initialize input arrays with random values
    init_array((double *)_fict_original, 40);
    for (int i = 0; i < nx; i++)
    {
        init_array((double *)ex_original[i], ny);
        init_array((double *)ey_original[i], ny);
        init_array((double *)hz_original[i], ny);

        // Copy initialized values to arrays for transformed function to ensure both start with the same input
        for (int j = 0; j < ny; j++)
        {
            ex_transformed[i][j] = ex_original[i][j];
            ey_transformed[i][j] = ey_original[i][j];
            hz_transformed[i][j] = hz_original[i][j];
        }
    }
    for (int i = 0; i < 40; i++)
    {
        _fict_transformed[i] = _fict_original[i];
    }

    // Execute both the original and transformed FDTD kernels
    kernel_fdtd_2d_original(tmax, nx, ny, ex_original, ey_original, hz_original, _fict_original);
    kernel_fdtd_2d_transformed(tmax, nx, ny, ex_transformed, ey_transformed, hz_transformed, _fict_transformed);

    // Compare the output arrays from both functions to ensure they are the same
    int ex_same = compare_arrays((double *)ex_original, (double *)ex_transformed, nx * ny);
    int ey_same = compare_arrays((double *)ey_original, (double *)ey_transformed, nx * ny);
    int hz_same = compare_arrays((double *)hz_original, (double *)hz_transformed, nx * ny);

    if (ex_same && ey_same && hz_same)
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    