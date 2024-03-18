#include <math.h>
#pragma ACCEL kernel

void kernel_correlation(double float_n,double data[100][80],double corr[80][80],double mean[80],double stddev[80])
{
  int i;
  int j;
  int k;
  double eps = 0.1;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (j = 0; j < 80; j++) {
    mean[j] = 0.0;
    stddev[j] = 0.0;
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (i = 0; i < 100; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (i = 0; i < 100; i++) {
      stddev[j] += pow(data[i][j] - mean[j],(double )2);
    }
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
    stddev[j] = (stddev[j] <= eps?1.0 : stddev[j]);
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 100; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
    for (j = 0; j < 80; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (i = 0; i < 80 - 1; i++) {
    corr[i][i] = 1.0;
    
#pragma ACCEL PIPELINE auto{__PIPE__L7}
    for (j = i + 1; j < 80; j++) {
      corr[i][j] = 0.0;
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7_0}
      for (k = 0; k < 100; k++) {
        corr[i][j] += data[k][i] * data[k][j];
      }
      corr[j][i] = corr[i][j];
    }
  }
  corr[80 - 1][80 - 1] = 1.0;
}
// Explanation of transformations:
// 1. Loop permutation: The loops for calculating mean and standard deviation were originally separate. By permuting the loops, we can calculate both mean and standard deviation in the same loop iteration, reducing the number of loop iterations and improving data locality.
// 2. Loop fusion: The loops for calculating mean and standard deviation were fused into a single loop to further improve data locality and reduce loop overhead.
// 3. Loop distribution: The calculation of correlation matrix involves nested loops. By distributing the outer loop, we can parallelize the computation of correlation values for different pairs of columns, improving parallelism and performance.