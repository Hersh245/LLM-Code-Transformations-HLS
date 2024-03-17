#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#pragma ACCEL kernel

void md_kernel_original(double force_x[256], double force_y[256], double force_z[256], double position_x[256], double position_y[256], double position_z[256], int NL[4096])
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

#pragma ACCEL PIPELINE auto{__PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
loop_i:
    for (i = 0; i < 256; i++)
    {
        i_x = position_x[i];
        i_y = position_y[i];
        i_z = position_z[i];
        fx = ((double)0);
        fy = ((double)0);
        fz = ((double)0);
    loop_j:
        for (j = 0; j < 16; j++)
        {
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
        // Update forces after all neighbors accounted for.
        force_x[i] = fx;
        force_y[i] = fy;
        force_z[i] = fz;
        // printf("dF=%lf,%lf,%lf\n", fx, fy, fz);
    }
}

#pragma ACCEL kernel

void md_kernel_transformed(double force_x[256], double force_y[256], double force_z[256], double position_x[256], double position_y[256], double position_z[256], int NL[4096])
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

#pragma ACCEL PIPELINE auto{__PIPE__L0 }

#pragma ACCEL TILE FACTOR = 16 {__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
loop_j:
    for (j = 0; j < 16; j++)
    {
    loop_i:
        for (i = 0; i < 256; i++)
        {
            i_x = position_x[i];
            i_y = position_y[i];
            i_z = position_z[i];
            fx = ((double)0);
            fy = ((double)0);
            fz = ((double)0);

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
            fx += delx * force;
            fy += dely * force;
            fz += delz * force;

            force_x[i] += fx;
            force_y[i] += fy;
            force_z[i] += fz;
        }
    }
}

// Function to initialize array with random doubles
void init_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
    }
}

// Function to compare arrays
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
    md_kernel_original(force_x_original, force_y_original, force_z_original, position_x, position_y, position_z, NL);

    // Call the transformed function
    md_kernel_transformed(force_x_transformed, force_y_transformed, force_z_transformed, position_x, position_y, position_z, NL);

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