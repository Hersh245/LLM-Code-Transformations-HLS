
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#pragma ACCEL kernel

void kernel_md_original(double force_x[256],double force_y[256],double force_z[256],double position_x[256],double position_y[256],double position_z[256],int NL[4096])
{
  double delx;
  double dely;
  double delz;
  double r2inv;
  double r6inv;
  double potential;
  double force;
  double j_x;
  double j_y;
  double j_z;
  double i_x;
  double i_y;
  double i_z;
  double fx;
  double fy;
  double fz;
  int i;
  int j;
  int jidx;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  loop_i:
  for (i = 0; i < 256; i++) {
    i_x = position_x[i];
    i_y = position_y[i];
    i_z = position_z[i];
    fx = ((double )0);
    fy = ((double )0);
    fz = ((double )0);
    loop_j:
    for (j = 0; j < 16; j++) {
// Get neighbor
      jidx = NL[i * 16 + j];
// Look up x,y,z positions
      j_x = position_x[jidx];
      j_y = position_y[jidx];
      j_z = position_z[jidx];
// Calc distance
      delx = i_x - j_x;
      dely = i_y - j_y;
      delz = i_z - j_z;
      r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
// Assume no cutoff and aways account for all nodes in area
      r6inv = r2inv * r2inv * r2inv;
      potential = r6inv * (1.5 * r6inv - 2.0);
// Sum changes in force
      force = r2inv * potential;
      fx += delx * force;
      fy += dely * force;
      fz += delz * force;
    }
//Update forces after all neighbors accounted for.
    force_x[i] = fx;
    force_y[i] = fy;
    force_z[i] = fz;
//printf("dF=%lf,%lf,%lf\n", fx, fy, fz);
  }
}

// Given the performance estimate provided, the primary target for optimization is the nested loops `loop_i` and `loop_j`, which account for the majority of the accumulated cycles (99.8% of the total cycles). The inner loop `loop_j` has a pipeline initiation interval (II) of 5, indicating that a new iteration of the loop starts every 5 cycles. To optimize this code for High-Level Synthesis (HLS), we can consider applying loop transformations that aim to reduce the initiation interval, increase data locality, and improve parallelism. 
// 
// Here's a transformed version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_md_transformed(double force_x[256], double force_y[256], double force_z[256], double position_x[256], double position_y[256], double position_z[256], int NL[4096]) {
  double delx, dely, delz;
  double r2inv, r6inv, potential, force;
  double j_x, j_y, j_z;
  double i_x, i_y, i_z;
  double fx[256], fy[256], fz[256];
  int i, j, jidx;

  // Initialize accumulators to zero
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 256; i++) {
    fx[i] = 0.0;
    fy[i] = 0.0;
    fz[i] = 0.0;
  }

  // Split the original loop into two separate loops to enhance parallelism and data locality
  #pragma ACCEL PIPELINE auto{__PIPE__L0}
  loop_i:
  for (i = 0; i < 256; i++) {
    i_x = position_x[i];
    i_y = position_y[i];
    i_z = position_z[i];

    #pragma ACCEL PIPELINE auto{__PIPE__L0}
    loop_j:
    for (j = 0; j < 16; j++) {
      jidx = NL[i * 16 + j];
      j_x = position_x[jidx];
      j_y = position_y[jidx];
      j_z = position_z[jidx];

      delx = i_x - j_x;
      dely = i_y - j_y;
      delz = i_z - j_z;
      r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
      r6inv = r2inv * r2inv * r2inv;
      potential = r6inv * (1.5 * r6inv - 2.0);
      force = r2inv * potential;

      // Accumulate forces directly into temporary arrays to avoid dependency issues
      fx[i] += delx * force;
      fy[i] += dely * force;
      fz[i] += delz * force;
    }
  }

  // Final loop to update the forces after all neighbors are accounted for
  #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 256; i++) {
    force_x[i] = fx[i];
    force_y[i] = fy[i];
    force_z[i] = fz[i];
  }
}
// ### Rationale Behind Transformations:
// 
// 1. **Separation of Accumulation and Initialization**: By separating the initialization of the force accumulators (`fx`, `fy`, `fz`) from the main computation loop, we allow for better pipelining and parallel execution of the initialization phase. This can lead to a reduction in the initiation interval for the main computation loop.
// 
// 2. **Loop Pipelining**: The `#pragma ACCEL PIPELINE` directive is applied to both the `loop_i` and `loop_j` loops to encourage the HLS tool to pipeline these loops. Pipelining can significantly reduce the initiation interval (II) of a loop, allowing for faster execution.
// 
// 3. **Accumulation into Temporary Arrays**: Instead of directly updating the `force_x`, `force_y`, and `force_z` arrays within the inner loop, we accumulate the forces into temporary arrays (`fx`, `fy`, `fz`). This transformation reduces dependencies between loop iterations, potentially allowing for more aggressive pipelining and parallelization by the HLS tool.
// 
// 4. **Final Update in a Separate Loop**: After all computations are done, a final loop updates the original force arrays from the temporary accumulators. This separation ensures that the computation and memory update phases are distinct, which can help in optimizing memory access patterns and further exploiting parallelism.
// 
// By applying these transformations, the goal is to reduce the overall accumulated cycles (AC) and cycles per call (CPC) for the `md_kernel` function, leading to improved performance in the synthesized hardware.

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
    // Initialize seed for reproducibility
    srand(42);

    // Declare arrays
    double force_x_original[256], force_y_original[256], force_z_original[256];
    double position_x[256], position_y[256], position_z[256];
    int NL[4096];

    double force_x_transformed[256], force_y_transformed[256], force_z_transformed[256];

    // Initialize arrays
    init_array(force_x_original, 256);
    init_array(force_y_original, 256);
    init_array(force_z_original, 256);
    init_array(position_x, 256);
    init_array(position_y, 256);
    init_array(position_z, 256);
    for (int i = 0; i < 4096; i++)
    {
        NL[i] = rand() % 256; // Assuming NL is some kind of index array with values in the range [0, 255]
    }

    // Copy initialized forces to transformed arrays to ensure both functions start with the same input
    for (int i = 0; i < 256; i++)
    {
        force_x_transformed[i] = force_x_original[i];
        force_y_transformed[i] = force_y_original[i];
        force_z_transformed[i] = force_z_original[i];
    }

    // Call the original function
    kernel_md_original(force_x_original, force_y_original, force_z_original, position_x, position_y, position_z, NL);

    // Call the transformed function
    kernel_md_transformed(force_x_transformed, force_y_transformed, force_z_transformed, position_x, position_y, position_z, NL);

    // Compare output arrays
    if (compare_arrays(force_x_original, force_x_transformed, 256) &&
        compare_arrays(force_y_original, force_y_transformed, 256) &&
        compare_arrays(force_z_original, force_z_transformed, 256))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    