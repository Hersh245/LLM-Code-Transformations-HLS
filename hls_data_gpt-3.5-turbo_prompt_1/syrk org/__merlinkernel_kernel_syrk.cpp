#define __constant
#define __kernel
#define __global
#include "memcpy_256_1d.h"
#include "memcpy_512_1d.h"
#define SIZE_1 80
#include "memcpy_512_2d.h"
#undef SIZE_1
#define SIZE_1 60
#include "memcpy_256_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"







// Original: #pragma ACCEL kernel
static int kernel_syrk_dummy_pos;
extern "C" { 

void kernel_syrk(double alpha,double beta,class ap_uint< 512 > C[800],class ap_uint< 256 > A[1200])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=1200 bundle=merlin_gmem_kernel_syrk_256_0
  
#pragma HLS INTERFACE m_axi port=C offset=slave depth=800 bundle=merlin_gmem_kernel_syrk_512_C
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=C bundle=control
  
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
  
#pragma HLS INTERFACE s_axilite port=beta bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma HLS DATA_PACK VARIABLE=C
  
#pragma ACCEL interface variable=A depth=80,60 max_depth=80,60
  
#pragma ACCEL interface variable=C depth=80,80 max_depth=80,80
  double A_5_1_buf[80][60];
  
#pragma HLS array_partition variable=A_5_1_buf cyclic factor=4 dim=2
  double A_5_0_buf[80][60];
  
#pragma HLS array_partition variable=A_5_0_buf cyclic factor=4 dim=2
  double C_buf[80][80];
  
#pragma HLS array_partition variable=C_buf cyclic factor=8 dim=2
  int i;
  int j;
  int k;
// Existing HLS partition: #pragma HLS array_partition variable=C_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(C_buf,0,0,(class ap_uint< 512 > *)C,(0 * 8),sizeof(double ) * ((unsigned long )6400L),6400L);
{
    memcpy_wide_bus_read_double_2d_60_256(A_5_0_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )4800L),4800L);
// Existing HLS partition: #pragma HLS array_partition variable=A_5_1_buf cyclic factor = 4 dim=2
    memcpy_wide_bus_read_double_2d_60_256(A_5_1_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )4800L),4800L);
  }
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  merlinL3:
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  for (i = 0; i < 80; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL2:
    for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=C_buf array inter false
      
#pragma HLS pipeline
      if (j <= i) {
        C_buf[i][j] *= beta;
      }
    }
    merlinL1:
    for (k = 0; k < 60; k++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
      merlinL0:
      for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=C_buf array inter false
        
#pragma HLS pipeline
        if (j <= i) {
          C_buf[i][j] += alpha * A_5_1_buf[i][k] * A_5_0_buf[j][k];
        }
      }
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=C_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)C,C_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )6400L),6400L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=A_5_0_buf cyclic factor = 4 dim=2
