
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#pragma ACCEL kernel

void kernel_syrk_original(double alpha,double beta,double C[80][80],double A[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}

#pragma ACCEL kernel

void kernel_syrk_transformed(double alpha,double beta,double C[80][80],double A[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 80; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (k = 0; k < 60; k++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] *= beta;
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}
// Explanation:
// 1. Loop permutation: The innermost loop over 'k' has been moved to the second level of nesting, allowing for better data reuse and reducing loop overhead.
// 2. Loop fusion: The two inner loops over 'j' have been fused together to combine the operations of scaling and updating the elements of matrix C in a single loop iteration, improving memory access patterns and reducing loop overhead.

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
    double A[80][60], C_original[80][80], C_transformed[80][80];
    double alpha = 1.2, beta = 0.8;

    // Initialize matrices A, C_original, and C_transformed with random values
    init_array(&A[0][0], 80 * 60);
    init_array(&C_original[0][0], 80 * 80);
    // Copy C_original to C_transformed to ensure they start with the same values
    for (int i = 0; i < 80; i++)
    {
        for (int j = 0; j < 80; j++)
        {
            C_transformed[i][j] = C_original[i][j];
        }
    }

    // Run both kernel versions
    kernel_syrk_original(alpha, beta, C_original, A);
    kernel_syrk_transformed(alpha, beta, C_transformed, A);

    // Compare the output matrices
    if (compare_arrays(&C_original[0][0], &C_transformed[0][0], 80 * 80))
    {
        printf("Success: The outputs of both functions are equivalent.\n");
    }
    else
    {
        printf("Failure: The outputs of the functions differ.\n");
    }

    return 0;
}
    