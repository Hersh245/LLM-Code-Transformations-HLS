#define __constant
#define __kernel
#define __global
#include "memcpy_256_1d.h"
#include "memcpy_512_1d.h"
#define SIZE_1 116
#include "memcpy_256_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"






// Original: #pragma ACCEL kernel
static int kernel_bicg_dummy_pos;
extern "C" { 

void kernel_bicg(class ap_uint< 256 > A[3596],class ap_uint< 512 > s[15],class ap_uint< 512 > q[16],class ap_uint< 512 > p[15],class ap_uint< 512 > r[16])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=3596 bundle=merlin_gmem_kernel_bicg_256_0
  
#pragma HLS INTERFACE m_axi port=p offset=slave depth=15 bundle=merlin_gmem_kernel_bicg_512_0
  
#pragma HLS INTERFACE m_axi port=q offset=slave depth=16 bundle=merlin_gmem_kernel_bicg_512_q
  
#pragma HLS INTERFACE m_axi port=r offset=slave depth=16 bundle=merlin_gmem_kernel_bicg_512_1
  
#pragma HLS INTERFACE m_axi port=s offset=slave depth=15 bundle=merlin_gmem_kernel_bicg_512_s
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=p bundle=control
  
#pragma HLS INTERFACE s_axilite port=q bundle=control
  
#pragma HLS INTERFACE s_axilite port=r bundle=control
  
#pragma HLS INTERFACE s_axilite port=s bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=r
  
#pragma HLS DATA_PACK VARIABLE=p
  
#pragma HLS DATA_PACK VARIABLE=q
  
#pragma HLS DATA_PACK VARIABLE=s
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=r depth=124 max_depth=124
  
#pragma ACCEL interface variable=p depth=116 max_depth=116
  
#pragma ACCEL interface variable=q depth=124 max_depth=124
  
#pragma ACCEL interface variable=s depth=116 max_depth=116
  
#pragma ACCEL interface variable=A depth=124,116 max_depth=124,116
  double p_4_0_buf[116];
  
#pragma HLS array_partition variable=p_4_0_buf cyclic factor=8 dim=1
  double A_4_1_buf[124][116];
  
#pragma HLS array_partition variable=A_4_1_buf cyclic factor=4 dim=2
  double A_4_0_buf[124][116];
  
#pragma HLS array_partition variable=A_4_0_buf cyclic factor=4 dim=2
  double r_4_0_buf[124];
  
#pragma HLS array_partition variable=r_4_0_buf cyclic factor=8 dim=1
  double q_buf[124];
  
#pragma HLS array_partition variable=q_buf cyclic factor=8 dim=1
  double s_buf[116];
// Existing HLS partition: #pragma HLS array_partition variable=s_buf cyclic factor = 8 dim=1
  
#pragma HLS array_partition variable=s_buf cyclic factor=8 dim=1
  memcpy_wide_bus_read_double_512(&s_buf[0],(class ap_uint< 512 > *)s,(0 * 8),sizeof(double ) * ((unsigned long )116),116L);
  int i;
  int j;
//#pragma scop
  merlinL2:
//#pragma scop
  for (i = 0; i < 116; i++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE AUTO
{
    
#pragma HLS dependence variable=s_buf array inter false
    
#pragma HLS pipeline
    s_buf[i] = ((double )0);
  }
// Existing HLS partition: #pragma HLS array_partition variable=q_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&q_buf[0],(class ap_uint< 512 > *)q,(0 * 8),sizeof(double ) * ((unsigned long )124),124L);
{
    memcpy_wide_bus_read_double_512(&r_4_0_buf[0],(class ap_uint< 512 > *)r,(0 * 8),sizeof(double ) * ((unsigned long )124),124L);
  }
{
    memcpy_wide_bus_read_double_2d_116_256(A_4_0_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14384L),14384L);
// Existing HLS partition: #pragma HLS array_partition variable=A_4_1_buf cyclic factor = 4 dim=2
    memcpy_wide_bus_read_double_2d_116_256(A_4_1_buf,0,0,(class ap_uint< 256 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14384L),14384L);
  }
{
    memcpy_wide_bus_read_double_512(&p_4_0_buf[0],(class ap_uint< 512 > *)p,(0 * 8),sizeof(double ) * ((unsigned long )116),116L);
  }
  merlinL1:
  for (i = 0; i < 124; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
    q_buf[i] = 0.0;
    merlinL0:
    for (j = 0; j < 116; j++) 
// Original: #pragma ACCEL PARALLEL reduction FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=s_buf array inter false
      
#pragma HLS pipeline
      s_buf[j] += r_4_0_buf[i] * A_4_1_buf[i][j];
      q_buf[i] += A_4_0_buf[i][j] * p_4_0_buf[j];
    }
  }
//#pragma endscop
// Existing HLS partition: #pragma HLS array_partition variable=q_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)q,&q_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )124),124L);
// Existing HLS partition: #pragma HLS array_partition variable=s_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)s,&s_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )116),116L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=A_4_0_buf cyclic factor = 4 dim=2
// Existing HLS partition: #pragma HLS array_partition variable=p_4_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=r_4_0_buf cyclic factor = 8 dim=1
