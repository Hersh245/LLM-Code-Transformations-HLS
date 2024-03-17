#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Original version of kernel_covariance
#pragma ACCEL kernel

void kernel_covariance_original(int m, int n, double float_n, double data[100][80], double cov[80][80], double mean[80])
{
    int i;
    int j;
    int k;
    // #pragma scop

#pragma ACCEL PIPELINE auto{ __PIPE__L0 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (j = 0; j < 80; j++)
    {
        mean[j] = 0.0;

#pragma ACCEL PARALLEL reduction = mean FACTOR = auto{ __PARA__L3 }
        for (i = 0; i < 100; i++)
        {
            mean[j] += data[i][j];
        }
        mean[j] /= float_n;
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L1 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L1 }
    for (i = 0; i < 100; i++)
    {

#pragma ACCEL PARALLEL reduction = data FACTOR = auto{ __PARA__L4 }
        for (j = 0; j < 80; j++)
        {
            data[i][j] -= mean[j];
        }
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L2 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L2 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L2 }
    for (i = 0; i < 80; i++)
    {

#pragma ACCEL PIPELINE auto{ __PIPE__L5 }
        for (j = i; j < 80; j++)
        {
            cov[i][j] = 0.0;

#pragma ACCEL PARALLEL reduction = cov FACTOR = auto{ __PARA__L6 }
            for (k = 0; k < 100; k++)
            {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= float_n - 1.0;
            cov[j][i] = cov[i][j];
        }
    }
    // #pragma endscop
}

// Transformed version of kernel_covariance
#pragma ACCEL kernel

void kernel_covariance_transformed(int m, int n, double float_n, double data[100][80], double cov[80][80], double mean[80])
{
    int i;
    int j;
    int k;

    // Calculate mean
#pragma ACCEL PIPELINE auto{ __PIPE__L0 }
#pragma ACCEL TILE FACTOR = auto{__TILE__L0 }
#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L0 }
    for (j = 0; j < 80; j++)
    {
        mean[j] = 0.0;

#pragma ACCEL PARALLEL reduction = mean FACTOR = auto{ __PARA__L3 }
        for (i = 0; i < 100; i++)
        {
            mean[j] += data[i][j];
        }
        mean[j] /= float_n;
    }

    // Center the data
#pragma ACCEL PIPELINE auto{ __PIPE__L1 }
#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }
#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L1 }
    for (i = 0; i < 100; i++)
    {

#pragma ACCEL PARALLEL reduction = data FACTOR = auto{ __PARA__L4 }
        for (j = 0; j < 80; j++)
        {
            data[i][j] -= mean[j];
        }
    }

    // Calculate covariance
#pragma ACCEL PIPELINE auto{ __PIPE__L2 }
#pragma ACCEL TILE FACTOR = auto{__TILE__L2 }
#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L2 }
    for (i = 0; i < 80; i++)
    {

#pragma ACCEL PIPELINE auto{ __PIPE__L5 }
        for (j = i; j < 80; j++)
        {
            cov[i][j] = 0.0;

#pragma ACCEL PARALLEL reduction = cov FACTOR = auto{ __PARA__L6 }
            for (k = 0; k < 100; k++)
            {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= float_n - 1.0;
            cov[j][i] = cov[i][j];
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
    int m = 100, n = 80;
    double float_n = (double)n;
    double data_orig[100][80], data_trans[100][80], cov_original[80][80], cov_transformed[80][80], mean_original[80], mean_transformed[80];

    // Seed the random number generator for reproducibility
    srand(42);

    // Initialize data array with random numbers
    for (int i = 0; i < m; i++)
    {
        init_array(data_orig[i], n);
    }

    memcpy(data_trans, data_orig, sizeof(data_orig));

    // Randomly initialize output arrays
    init_array((double *)cov_original, 80 * 80);
    init_array((double *)cov_transformed, 80 * 80);
    init_array(mean_original, 80);
    init_array(mean_transformed, 80);

    // // Call both the original and the transformed function
    kernel_covariance_original(m, n, float_n, data_orig, cov_original, mean_original);
    kernel_covariance_transformed(m, n, float_n, data_trans, cov_transformed, mean_transformed);

    // Compare the covariance matrices
    if (compare_arrays((double *)cov_original, (double *)cov_transformed, 80 * 80) &&
        compare_arrays(mean_original, mean_transformed, 80))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}