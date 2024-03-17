#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define M 124
#define N 116

// Original version of kernel_bicg
void kernel_bicg_original(int m, int n, double A[124][116], double s[116], double q[124], double p[116], double r[124])
{
    int i;
    int j;
    // #pragma scop

#pragma ACCEL PARALLEL FACTOR = auto{ __PARA__L0 }
    for (i = 0; i < 116; i++)
    {
        s[i] = ((double)0);
    }

#pragma ACCEL PIPELINE auto{ __PIPE__L1 }

#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }

#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L1 }
    for (i = 0; i < 124; i++)
    {
        q[i] = 0.0;

#pragma ACCEL PARALLEL reduction FACTOR = auto{ __PARA__L2 }
        for (j = 0; j < 116; j++)
        {
            s[j] += r[i] * A[i][j];
            q[i] += A[i][j] * p[j];
        }
    }
    // #pragma endscop
}

// Transformed version of kernel_bicg
void kernel_bicg_transformed(int m, int n, double A[124][116], double s[116], double q[124], double p[116], double r[124])
{
    int i;
    int j;

#pragma ACCEL PARALLEL FACTOR = auto{ __PARA__L1 }
#pragma ACCEL PIPELINE auto{__PIPE__L1 }
#pragma ACCEL TILE FACTOR = auto{__TILE__L1 }
#pragma ACCEL PARALLEL FACTOR = auto{__PARA__L2 }
    for (j = 0; j < 116; j++)
    {
        for (i = 0; i < 124; i++)
        {
            s[j] += r[i] * A[i][j];
            q[i] += A[i][j] * p[j];
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
    double A[M][N], p[N], r[M];
    double s_original[N], q_original[M];
    double s_transformed[N], q_transformed[M];

    srand(time(NULL)); // Seed for random number generation

    // Initialize arrays
    for (int i = 0; i < M; i++)
    {
        init_array(A[i], N);
    }
    init_array(p, N);
    init_array(r, M);

    // Initialize output arrays with zeros
    init_array(s_original, N);
    init_array(q_original, M);
    init_array(s_transformed, N);
    init_array(q_transformed, M);

    // Run both versions of the kernel_bicg function
    kernel_bicg_original(M, N, A, s_original, q_original, p, r);
    kernel_bicg_transformed(M, N, A, s_transformed, q_transformed, p, r);

    // Compare the output arrays
    if (compare_arrays(s_original, s_transformed, N) && compare_arrays(q_original, q_transformed, M))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}