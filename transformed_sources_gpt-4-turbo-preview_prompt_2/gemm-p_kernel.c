// To optimize the given code for High-Level Synthesis (HLS), we can apply several code transformations. The goal is to improve parallelism, data locality, and resource utilization. Here's a step-by-step transformation of the code:
// 
// 1. **Loop Tiling**: This transformation breaks the loops into smaller chunks, improving cache locality and enabling more efficient use of on-chip memory.
// 2. **Loop Permutation**: Changing the loop order can improve data access patterns, enhancing memory bandwidth utilization.
// 3. **Loop Fusion**: Combining loops that iterate over the same range can reduce loop overhead and improve data locality.
// 4. **Loop Distribution**: Separating independent operations within a loop can enable more parallelism.
// 
// Given the original code, we'll focus on applying Loop Tiling and Loop Permutation, as these are the most relevant for the given matrix multiplication and the pragmas suggest that tiling and parallelism are desired optimizations.
#pragma ACCEL kernel

void kernel_gemm(int ni, int nj, int nk, double alpha, double beta, double C[60][70], double A[60][80], double B[80][70]) {
    int i, j, k;
    int ii, jj, kk;
    const int TILE_I = 10; // Tile size for i loop
    const int TILE_J = 14; // Tile size for j loop
    const int TILE_K = 16; // Tile size for k loop

#pragma scop

    // Applying loop tiling to i, j, and k loops
    for (ii = 0; ii < 60; ii += TILE_I) {
        for (jj = 0; jj < 70; jj += TILE_J) {
            for (kk = 0; kk < 80; kk += TILE_K) {

#pragma ACCEL PIPELINE auto{__PIPE__L0}

#pragma ACCEL TILE FACTOR=auto{__TILE__L0}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
                for (i = ii; i < ((ii + TILE_I) < 60 ? (ii + TILE_I) : 60); i++) {

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
                    for (j = jj; j < ((jj + TILE_J) < 70 ? (jj + TILE_J) : 70); j++) {
                        if (kk == 0) { // This ensures we only scale C once per full K iteration
                            C[i][j] *= beta;
                        }
                    }

#pragma ACCEL PIPELINE auto{__PIPE__L2}

#pragma ACCEL TILE FACTOR=auto{__TILE__L2}

#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
                    for (k = kk; k < ((kk + TILE_K) < 80 ? (kk + TILE_K) : 80); k++) {

#pragma ACCEL PARALLEL reduction=C FACTOR=auto{__PARA__L3}
                        for (j = jj; j < ((jj + TILE_J) < 70 ? (jj + TILE_J) : 70); j++) {
                            C[i][j] += alpha * A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }

#pragma endscop
}
// ### Rationale Behind Transformations:
// 
// - **Loop Tiling**: Applied to `i`, `j`, and `k` loops to break down the computation into smaller chunks that fit better in the cache, reducing memory access times and improving parallelism opportunities.
// - **Loop Permutation**: The loop order was adjusted to `ii-jj-kk-i-j-k` to align with the tiling and ensure that the most inner loops (`j` and `k` for the computation) benefit from spatial locality in accessing `A`, `B`, and `C`.
// - **Conditional Scaling of `C`**: To avoid redundant scaling of `C[i][j]` by `beta` in every iteration of `k`, it's done conditionally when `kk == 0`.
// 
// These transformations aim to enhance data locality, exploit fine-grained parallelism, and ensure efficient use of the FPGA resources through HLS pragmas.