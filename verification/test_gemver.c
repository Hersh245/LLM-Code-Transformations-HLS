#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#pragma ACCEL kernel

void kernel_gemver_original(int n, double alpha, double beta, double A[120][120], double u1[120], double v1[120], double u2[120], double v2[120], double w[120], double x[120], double y[120], double z[120])
{
    int i;
    int j;

#pragma scop

#pragma ACCEL PIPELINE auto{ __PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (i = 0; i < 120; i++)
    {

#pragma ACCEL PARALLEL reduction = A FACTOR = auto{ __PARA__L4 }
        for (j = 0; j < 120; j++)
        {
            A[i][j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L1 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L1 }
    for (i = 0; i < 120; i++)
    {

#pragma ACCEL PARALLEL reduction = x FACTOR = auto{ __PARA__L5 }
        for (j = 0; j < 120; j++)
        {
            x[i] += beta * A[j][i] * y[j];
        }
    }

#pragma ACCEL PARALLEL reduction = x FACTOR = auto{ __PARA__L2 }
    for (i = 0; i < 120; i++)
    {
        x[i] += z[i];
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L3 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L3 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L3 }
    for (i = 0; i < 120; i++)
    {

#pragma ACCEL PARALLEL reduction = w FACTOR = auto{ __PARA__L6 }
        for (j = 0; j < 120; j++)
        {
            w[i] += alpha * A[i][j] * x[j];
        }
    }

#pragma endscop
}

#pragma ACCEL kernel

void kernel_gemver_transformed(int n, double alpha, double beta, double A[120][120], double u1[120], double v1[120], double u2[120], double v2[120], double w[120], double x[120], double y[120], double z[120])
{
    int i;
    int j;

#pragma scop

#pragma ACCEL PIPELINE auto{ __PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (i = 0; i < 120; i++)
    {

#pragma ACCEL PARALLEL reduction = A FACTOR = auto{ __PARA__L4 }
        for (j = 0; j < 120; j++)
        {
            A[i][j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L1 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L1 }
    for (i = 0; i < 120; i++)
    {

#pragma ACCEL PARALLEL reduction = x FACTOR = auto{ __PARA__L5 }
        for (j = 0; j < 120; j++)
        {
            x[i] += beta * A[j][i] * y[j];
        }
    }

#pragma ACCEL PARALLEL reduction = x FACTOR = auto{ __PARA__L2 }
    for (i = 0; i < 120; i++)
    {
        x[i] += z[i];
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L3 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L3 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L3 }
    for (j = 0; j < 120; j++)
    {

#pragma ACCEL PARALLEL reduction = w FACTOR = auto{ __PARA__L6 }
        for (i = 0; i < 120; i++)
        {
            w[i] += alpha * A[i][j] * x[j];
        }
    }

#pragma endscop
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
    int n = 120;
    double alpha = 0.23, beta = 0.45;

    double A_original[120][120], A_transformed[120][120];
    double u1[120], v1[120], u2[120], v2[120];
    double w[120], x[120], y[120], z[120];
    double w_original[120], x_original[120], y_original[120], z_original[120];
    double w_transformed[120], x_transformed[120], y_transformed[120], z_transformed[120];

    // Initialize matrices and vectors
    for (int i = 0; i < n; i++)
    {
        init_array(A_original[i], n);
        init_array(A_transformed[i], n); // Ensure both have the same initial values
        for (int j = 0; j < n; j++)
        {
            A_transformed[i][j] = A_original[i][j]; // Copy values to ensure equality
        }
    }

    init_array(u1, n);
    init_array(v1, n);
    init_array(u2, n);
    init_array(v2, n);
    init_array(w, n);
    init_array(x, n);
    init_array(y, n);
    init_array(z, n);

    // Copy initial values of output arrays
    for (int i = 0; i < n; i++)
    {
        w_original[i] = w[i];
        x_original[i] = x[i];
        y_original[i] = y[i];
        z_original[i] = z[i];

        w_transformed[i] = w[i];
        x_transformed[i] = x[i];
        y_transformed[i] = y[i];
        z_transformed[i] = z[i];
    }

    // Call the original function
    kernel_gemver_original(n, alpha, beta, A_original, u1, v1, u2, v2, w_original, x_original, y_original, z_original);

    // Call the transformed function
    kernel_gemver_transformed(n, alpha, beta, A_transformed, u1, v1, u2, v2, w_transformed, x_transformed, y_transformed, z_transformed);

    // Compare output arrays
    if (compare_arrays(w_original, w_transformed, n) && compare_arrays(x_original, x_transformed, n) &&
        compare_arrays(y_original, y_transformed, n) && compare_arrays(z_original, z_transformed, n))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}