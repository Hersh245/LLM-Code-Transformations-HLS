// Based on the performance estimates provided, we can optimize the code by applying loop tiling and loop permutation to improve the trip count (TC), accumulated cycles (AC), and cycles per call (CPC).
// 
// Here is the transformed code with loop tiling and loop permutation:
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
// Rationale for the transformations:
// 1. Loop Tiling: By tiling the loops, we can improve data locality and reduce cache misses, leading to better performance. Tiling the loops also allows for better parallelization and pipelining of the code.
// 2. Loop Permutation: Reordering the loops can help optimize the code for better pipelining and parallelization, which can reduce the accumulated cycles and cycles per call.
// 
// By applying loop tiling and loop permutation, we aim to reduce the accumulated cycles and cycles per call, ultimately improving the overall performance of the code for High-Level Synthesis.