#pragma ACCEL kernel

void kernel_fdtd_2d(int tmax, int nx, int ny, double ex[60][80], double ey[60][80], double hz[60][80], double _fict_[40])
{
    int t;
    int i;
    int j;
//#pragma scop

#pragma ACCEL PIPELINE auto{__PIPE__L0}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
    for (t = 0; t < 40; t++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
        for (j = 0; j < 80; j++) {
            ey[0][j] = _fict_[t];
        }

#pragma ACCEL PIPELINE auto{__PIPE__L0_1}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0_1}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
        for (i = 1; i < 60; i++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
            for (j = 0; j < 80; j++) {
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);
            }
        }

#pragma ACCEL PIPELINE auto{__PIPE__L0_2}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0_2}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2}
        for (i = 0; i < 60; i++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2_0}
            for (j = 1; j < 80; j++) {
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);
            }
        }

#pragma ACCEL PIPELINE auto{__PIPE__L0_3}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0_3}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3}
        for (i = 0; i < 59; i++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3_0}
            for (j = 0; j < 79; j++) {
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
            }
        }
    }
//#pragma endscop
}
// Explanation of transformations:
// 1. Loop Permutation: The original code already has the outer loop over 't' and inner loops over 'j' and 'i'. No further loop permutation is needed.
// 2. Loop Tiling: Tiling is applied to the outermost loop over 't' to improve data locality and facilitate parallel processing of tiles.
// 3. Loop Distribution: The code is already distributed across 'j' and 'i' loops for different computations, so no additional distribution is required.
// 4. Loop Fusion: Since the loops are already fused in a way that maximizes parallelism and minimizes dependencies, no further fusion is needed.