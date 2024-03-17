#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Original version of kernel_atax
#pragma ACCEL kernel

void kernel_atax_original(int m, int n, double A[116][124], double x[124], double y[124], double tmp[116])
{
    int i;
    int j;
    // #pragma scop
    for (i = 0; i < 124; i++)
        y[i] = ((double)0);

#pragma ACCEL PIPELINE auto{ __PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (i = 0; i < 116; i++)
    {
        tmp[i] = 0.0;

#pragma ACCEL PARALLEL reduction = tmp FACTOR = auto{ __PARA__L0_0 }
        for (j = 0; j < 124; j++)
        {
            tmp[i] += A[i][j] * x[j];
        }

#pragma ACCEL PARALLEL reduction = y FACTOR = auto{ __PARA__L0_1 }
        for (j = 0; j < 124; j++)
        {
            y[j] += A[i][j] * tmp[i];
        }
    }
    // #pragma endscop
}

// Transformed version of kernel_atax
#pragma ACCEL kernel

void kernel_atax_transformed(int m, int n, double A[116][124], double x[124], double y[124], double tmp[116])
{
    int i;
    int j;
    // #pragma scop
    for (i = 0; i < 124; i++)
        y[i] = ((double)0);

#pragma ACCEL PIPELINE auto{ __PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (j = 0; j < 124; j++)
    {

#pragma ACCEL PARALLEL reduction = tmp FACTOR = auto{ __PARA__L0_0 }
        for (i = 0; i < 116; i++)
        {
            tmp[i] += A[i][j] * x[j];
        }

#pragma ACCEL PARALLEL reduction = y FACTOR = auto{ __PARA__L0_1 }
        for (i = 0; i < 116; i++)
        {
            y[j] += A[i][j] * tmp[i];
        }
    }
    // #pragma endscop
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
    // Seed the random number generator to get different results each run
    srand((unsigned)time(NULL));

    // Dimensions
    int m = 116, n = 124;

    // Declare and initialize arrays
    double A[116][124], x[124], y_original[124], y_transformed[124], tmp_original[116], tmp_transformed[116];

    // Initialize input arrays with random values
    for (int i = 0; i < m; i++)
    {
        init_array(A[i], n);
    }
    init_array(x, n);

    // Initialize output arrays with random values
    init_array(y_original, n);
    init_array(y_transformed, n);
    init_array(tmp_original, m);
    init_array(tmp_transformed, m);

    // Call the original function
    kernel_atax_original(m, n, A, x, y_original, tmp_original);
    // Call the transformed function
    kernel_atax_transformed(m, n, A, x, y_transformed, tmp_transformed);

    // Compare the outputs
    int y_compare = compare_arrays(y_original, y_transformed, n);
    int tmp_compare = compare_arrays(tmp_original, tmp_transformed, m);

    if (y_compare && tmp_compare)
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}