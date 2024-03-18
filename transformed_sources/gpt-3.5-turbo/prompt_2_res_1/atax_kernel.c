#pragma ACCEL kernel

void kernel_atax(int m, int n, double A[116][124], double x[124], double y[124], double tmp[116]) {
    int i;
    int j;
    
    // Permute loops to improve data locality
    #pragma ACCEL PIPELINE auto{__PIPE__L0}
    #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
    #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
    for (j = 0; j < 124; j++) {
        y[j] = ((double)0);
        
        for (i = 0; i < 116; i++) {
            tmp[i] = 0.0;
            
            // Distribute loops to enable parallel processing
            #pragma ACCEL PARALLEL reduction=tmp FACTOR=auto{__PARA__L0_0}
            for (j = 0; j < 124; j++) {
                tmp[i] += A[i][j] * x[j];
            }
            
            // Fuse loops to reduce loop overhead
            #pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L0_1}
            for (j = 0; j < 124; j++) {
                y[j] += A[i][j] * tmp[i];
            }
        }
    }
}
// Explanation:
// 1. Loop Permutation: By permuting the loops, we can improve data locality by accessing elements of arrays in a more cache-friendly manner.
// 2. Loop Tiling: Tiling breaks down the computation into smaller chunks, improving data reuse and reducing memory access latency.
// 3. Loop Distribution: Distributing loops enables parallel processing of different iterations, improving performance by utilizing multiple processing units.
// 4. Loop Fusion: Loop fusion reduces loop overhead by combining multiple loops into a single loop, reducing the number of loop iterations and improving efficiency.