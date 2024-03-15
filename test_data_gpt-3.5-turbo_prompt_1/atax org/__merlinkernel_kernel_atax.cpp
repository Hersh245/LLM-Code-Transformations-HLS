#define __constant
#define __kernel
#define __global
#include "memcpy_256_1d.h"
#include "memcpy_512_1d.h"
#define SIZE_1 124
#include "memcpy_256_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"






// Original: #pragma ACCEL kernel
static int kernel_atax_dummy_pos;
extern "C" { 

void kernel_atax(class ap_uint< 256 > A[3596],class ap_uint< 512 > x[16],class ap_uint< 512 > y[16],class ap_uint< 512 > tmp[15])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=3596 bundle=merlin_gmem_kernel_atax_256_0
  
#pragma HLS INTERFACE m_axi port=tmp offset=slave depth=15 bundle=merlin_gmem_kernel_atax_512_tmp
  
#pragma HLS INTERFACE m_axi port=x offset=slave depth=16 bundle=merlin_gmem_kernel_atax_512_0
  
#pragma HLS INTERFACE m_axi port=y offset=slave depth=16 bundle=merlin_gmem_kernel_atax_512_y
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=tmp bundle=control
  
#pragma HLS INTERFACE s_axilite port=x bundle=control
  
#pragma HLS INTERFACE s_axilite port=y bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=tmp
  
#pragma HLS DATA_PACK VARIABLE=y
  
#pragma HLS DATA_PACK VARIABLE=x
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=tmp depth=116 max_depth=116
  
#pragma ACCEL interface variable=y depth=124 max_depth=124
  
#pragma ACCEL interface variable=x depth=124 max_depth=124
  
#pragma ACCEL interface variable=A depth=116,124 max_depth=116,124
  double x_4_0_buf[124];
  
#pragma HLS array_partition variable=x_4_0_buf cyclic factor=8 dim=1
  double A_4_0_buf[116][124];
  
#pragma HLS array_partition variable=A_4_0_buf cyclic factor=4 dim=2
  double A_5_0_buf[116][124];
  
#pragma HLS array_partition variable=A_5_0_buf cyclic factor=4 dim=2
  double tmp_buf[116];
  
#pragma HLS array_partition variable=tmp_buf cyclic factor=8 dim=1
  double y_buf[124];
// Existing HLS partition: #pragma HLS array_partition variable=y_buf cyclic factor = 8 dim=1
  
#pragma HLS array_partition variable=y_buf cyclic factor=8 dim=1
  memcpy_wide_bus_read_double_512(&y_buf[0],(class ap_uint< 512 > *)y,(0 * 8),sizeof(double ) * ((unsigned long )124),124L);
  int i;
  int j;
//#pragma scop
  merlinL3:
//#pragma scop
  for (i = 0; i < 124; i++) 
// Original: #pragma ACCEL PIPELINE AUTO
{
    
#pragma HLS dependence variable=y_buf array inter false
    
#pragma HLS pipeline
    y_buf[i] = ((double )0);
  }
// Existing HLS partition: #pragma HLS array_partition variable=tmp_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&tmp_buf[0],(class ap_uint< 512 > *)tmp,(0 * 8),sizeof(double ) * ((unsigned long )116),116L);
{
    memcpy_wide_bus_read_double_2d_124_256(A_5_0_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14384L),14384L);
// Existing HLS partition: #pragma HLS array_partition variable=A_4_0_buf cyclic factor = 4 dim=2
    memcpy_wide_bus_read_double_2d_124_256(A_4_0_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14384L),14384L);
  }
{
    memcpy_wide_bus_read_double_512(&x_4_0_buf[0],(class ap_uint< 512 > *)x,(0 * 8),sizeof(double ) * ((unsigned long )124),124L);
  }
  merlinL2:
  for (i = 0; i < 116; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    tmp_buf[i] = 0.0;
    merlinL1:
    for (j = 0; j < 124; j++) 
// Original: #pragma ACCEL PARALLEL reduction=tmp FACTOR=auto{__PARA__L0_0}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=tmp
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      tmp_buf[i] += A_4_0_buf[i][j] * x_4_0_buf[j];
    }
    merlinL0:
    for (j = 0; j < 124; j++) 
// Original: #pragma ACCEL PARALLEL reduction=y FACTOR=auto{__PARA__L0_1}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=y
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=y_buf array inter false
      
#pragma HLS pipeline
      y_buf[j] += A_5_0_buf[i][j] * tmp_buf[i];
    }
  }
//#pragma endscop
// Existing HLS partition: #pragma HLS array_partition variable=tmp_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)tmp,&tmp_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )116),116L);
// Existing HLS partition: #pragma HLS array_partition variable=y_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)y,&y_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )124),124L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=A_5_0_buf cyclic factor = 4 dim=2
// Existing HLS partition: #pragma HLS array_partition variable=x_4_0_buf cyclic factor = 8 dim=1
