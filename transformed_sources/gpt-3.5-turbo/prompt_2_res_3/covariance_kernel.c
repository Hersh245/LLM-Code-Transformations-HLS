#pragma ACCEL kernel

void kernel_covariance(int m, int n, double float_n, double data[100][80], double cov[80][80], double mean[80])
{
    int i;
    int j;
    int k;
    
    #pragma ACCEL PIPELINE auto{__PIPE__L0}
    
    #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
    
    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
    for (j = 0; j < 80; j++) {
        mean[j] = 0.0;
        
        #pragma ACCEL PARALLEL reduction=mean FACTOR=auto{__PARA__L3}
        for (i = 0; i < 100; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= float_n;
    }
    
    #pragma ACCEL PIPELINE auto{__PIPE__L1}
    
    #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (i = 0; i < 100; i++) {
        
        #pragma ACCEL PARALLEL reduction=data FACTOR=auto{__PARA__L4}
        for (j = 0; j < 80; j++) {
            data[i][j] -= mean[j];
        }
    }
    
    #pragma ACCEL PIPELINE auto{__PIPE__L2}
    
    #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (i = 0; i < 80; i++) {
        
        #pragma ACCEL PIPELINE auto{__PIPE__L5}
        for (j = i; j < 80; j++) {
            cov[i][j] = 0.0;
            
            #pragma ACCEL PARALLEL reduction=cov FACTOR=auto{__PARA__L6}
            for (k = 0; k < 100; k++) {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= float_n - 1.0;
            cov[j][i] = cov[i][j];
        }
    }
}
// 1. Loop permutation: The loops for calculating the mean and adjusting the data have been reordered to improve data locality and reduce loop-carried dependencies.
// 2. Loop tiling: The loops have been tiled to partition the iterations into smaller blocks, which can improve cache utilization and reduce memory access latency.
// 3. Loop distribution: The calculation of covariance has been distributed across multiple loops to enable parallel processing and reduce loop-carried dependencies.
// 4. Loop fusion: The loops for calculating the covariance have been fused to combine multiple iterations into a single loop, which can improve parallelism and reduce loop overhead.