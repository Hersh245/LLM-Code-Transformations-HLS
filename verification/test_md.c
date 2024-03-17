
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

// To optimize the given code for High-Level Synthesis (HLS), we can apply several transformations. The goal is to improve parallelism, data locality, and reduce the latency and resource utilization of the generated hardware. Here's an optimized version of the code with explanations for each transformation applied:
#pragma ACCEL kernel

void kernel_md_transformed(double force_x[256], double force_y[256], double force_z[256], double position_x[256], double position_y[256], double position_z[256], int NL[4096]) {
    double delx, dely, delz, r2inv, r6inv, potential, force;
    double fx[256], fy[256], fz[256]; // Temporary arrays to hold force calculations
    int i, j, jidx;

    // Initialize temporary force arrays to zero
    #pragma ACCEL PARALLEL FACTOR=auto
    for (i = 0; i < 256; i++) {
        fx[i] = 0;
        fy[i] = 0;
        fz[i] = 0;
    }

    // Main computation loop
    #pragma ACCEL PIPELINE auto
    loop_i: for (i = 0; i < 256; i++) {
        double i_x = position_x[i];
        double i_y = position_y[i];
        double i_z = position_z[i];

        loop_j: for (j = 0; j < 16; j++) {
            jidx = NL[i * 16 + j];
            double j_x = position_x[jidx];
            double j_y = position_y[jidx];
            double j_z = position_z[jidx];

            delx = i_x - j_x;
            dely = i_y - j_y;
            delz = i_z - j_z;
            r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
            r6inv = r2inv * r2inv * r2inv;
            potential = r6inv * (1.5 * r6inv - 2.0);
            force = r2inv * potential;
            fx[i] += delx * force;
            fy[i] += dely * force;
            fz[i] += delz * force;
        }
    }

    // Update forces after all neighbors accounted for
    #pragma ACCEL PARALLEL FACTOR=auto
    for (i = 0; i < 256; i++) {
        force_x[i] = fx[i];
        force_y[i] = fy[i];
        force_z[i] = fz[i];
    }
}
// ### Rationale Behind Transformations:
// 
// 1. **Temporary Arrays for Force Calculations**: Instead of directly updating the `force_x`, `force_y`, and `force_z` arrays inside the inner loop, temporary arrays `fx`, `fy`, and `fz` are used. This transformation reduces dependencies between loop iterations, allowing for better parallelization and pipelining.
// 
// 2. **Separate Initialization Loop**: The initialization of the temporary force arrays (`fx`, `fy`, `fz`) is moved to a separate loop. This allows the compiler to parallelize the initialization independently from the main computation, improving data locality and enabling more efficient pipelining.
// 
// 3. **Pragma Directives**: The `#pragma ACCEL PIPELINE auto` and `#pragma ACCEL PARALLEL FACTOR=auto` directives are strategically placed to hint the HLS tool to automatically optimize the pipelining of the main loop and parallelize both the initialization and final update loops. This aims to maximize the throughput of the design by overlapping operations and fully utilizing available resources.
// 
// 4. **Loop Fusion/Distribution**: The original code is essentially kept as is in terms of loop structure, but by introducing temporary arrays and separating initialization and final update steps, we've implicitly applied principles of loop distribution. This ensures that each part of the code is optimized for its specific pattern of data access and computation.
// 
// These transformations are designed to enhance the performance of the code when synthesized to hardware with HLS, focusing on improving parallelism, reducing latency, and optimizing resource utilization.

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
    