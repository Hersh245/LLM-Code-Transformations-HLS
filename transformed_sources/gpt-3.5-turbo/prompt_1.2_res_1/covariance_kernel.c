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
// Explanation of transformations:
// 
// 1. Loop Permutation:
//    - No loop permutation was applied as the original loop order was already optimized for the given computations.
// 
// 2. Loop Tiling:
//    - Tiling was applied to the loops to improve data locality and facilitate parallel processing by breaking down the iteration space into smaller tiles. This can help reduce memory access latency and improve cache utilization.
// 
// 3. Loop Distribution:
//    - Distribution was not explicitly applied in this code as the loops were already distributed across different computations in an optimized manner.
// 
// 4. Loop Fusion:
//    - No loop fusion was applied as the loops were already performing distinct computations and fusing them could potentially introduce dependencies and hinder parallelism.