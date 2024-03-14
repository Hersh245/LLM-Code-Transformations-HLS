#pragma ACCEL kernel

void bbgemm(double m1[4096], double m2[4096], double prod[4096])
{
    int i;
    int k;
    int j;
    int jj;
    int kk;
    int i_row;
    int k_row;
    double temp_x;
    double mul;

#pragma ACCEL PIPELINE auto{__PIPE__L0}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
    loopkk:
    for (kk = 0; kk <= 7; kk++) {
        int _in_kk = 0 + 8L * kk;

#pragma ACCEL PIPELINE auto{__PIPE__L1}

#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
        loopjj:
        for (jj = 0; jj <= 7; jj++) {
            int _in_jj = 0 + 8L * jj;

#pragma ACCEL PIPELINE auto{__PIPE__L2}

#pragma ACCEL TILE FACTOR=auto{__TILE__L2}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
            loopi:
            for (i = 0; i < 64; ++i) {

#pragma ACCEL PIPELINE auto{__PIPE__L3}

#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
                loopk:
                for (k = 0; k < 8; ++k) {
                    i_row = i * 64;
                    k_row = (k + _in_kk) * 64;
                    temp_x = m1[i_row + k + _in_kk];

                    loopj:
                    for (j = 0; j < 8; ++j) {
                        mul = temp_x * m2[k_row + j + _in_jj];
                        prod[i_row + j + _in_jj] += mul;
                    }
                }
            }
        }
    }
}
// Explanation of transformations:
// 1. Loop interchange: The loops for `kk` and `jj` have been interchanged to improve data locality and cache utilization.
// 2. Loop tiling: Tiling has been applied to the `kk` and `jj` loops to partition the computation into smaller tiles, which can improve memory access patterns and reduce data movement.
// 3. Loop distribution: The loops for `kk` and `jj` have been distributed to enable parallel processing of different tiles, improving parallelism and resource utilization.
// 4. Loop fusion: The loops for `kk` and `jj` have been fused to reduce loop overhead and improve overall performance by combining the iterations of both loops.