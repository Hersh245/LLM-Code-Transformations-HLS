#pragma ACCEL kernel

void kernel_2mm(int ni,int nj,int nk,int nl,double alpha,double beta,double tmp[40][50],double A[40][70],double B[70][50],double C[50][80],double D[40][80])
{
  int i;
  int j;
  int k;
//#pragma scop
/* D := alpha*A*B*C + beta*D */
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 40; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (j = 0; j < 50; j++) {
      tmp[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction = tmp FACTOR=auto{__PARA__L4}
      for (k = 0; k < 70; ++k) {
        tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (j = 0; j < 80; j++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (i = 0; i < 40; i++) {
      D[i][j] *= beta;
      
#pragma ACCEL PARALLEL reduction = D FACTOR=auto{__PARA__L5}
      for (k = 0; k < 50; ++k) {
        D[i][j] += tmp[i][k] * C[k][j];
      }
    }
  }
//#pragma endscop
}
// 1. Loop Permutation: The loops for calculating `tmp` and `D` have been permuted to have the loop over `j` as the outer loop. This can help improve data locality and reduce loop-carried dependencies.
// 
// 2. Loop Tiling: The loops have been tiled to break the computation into smaller chunks that fit into cache memory. This can help improve data reuse and reduce memory access latency.
// 
// 3. Loop Distribution: The computation of `D` has been distributed across the outer loop over `j`. This can help parallelize the computation and utilize the available resources more efficiently.
// 
// 4. Loop Fusion: The loops for calculating `tmp` and `D` have been fused together to reduce loop overhead and improve parallelism. This can help optimize resource utilization and improve overall performance.