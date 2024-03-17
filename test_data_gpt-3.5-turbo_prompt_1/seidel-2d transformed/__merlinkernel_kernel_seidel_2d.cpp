#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#include <string.h> 

#include "merlin_type_define.h"




// Original: #pragma ACCEL kernel
static int kernel_seidel_2d_dummy_pos;
extern "C" { 

void kernel_seidel_2d(class ap_uint< 512 > A[1800])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=1800 bundle=merlin_gmem_kernel_seidel_2d_512_A
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=A depth=120,120 max_depth=120,120
  double A_buf[14400];
  
#pragma HLS array_partition variable=A_buf cyclic factor=8 dim=1
  int t;
  int i;
  int j;
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&A_buf[0],(class ap_uint< 512 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14400L),14400L);
//#pragma scop
  merlinL2:
//#pragma scop
  for (t = 0; t <= 39; t++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL1:
    for (i = 1; i <= 118; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
      merlinL0:
      for (j = 1; j <= 118; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS pipeline
        A_buf[((long )i) * 120L + ((long )j)] = (A_buf[- 121L + ((long )i) * 120L + ((long )j)] + A_buf[- 120L + ((long )i) * 120L + ((long )j)] + A_buf[- 119L + ((long )i) * 120L + ((long )j)] + A_buf[- 1L + ((long )i) * 120L + ((long )j)] + A_buf[((long )i) * 120L + ((long )j)] + A_buf[1L + ((long )i) * 120L + ((long )j)] + A_buf[119L + ((long )i) * 120L + ((long )j)] + A_buf[120L + ((long )i) * 120L + ((long )j)] + A_buf[121L + ((long )i) * 120L + ((long )j)]) / 9.0;
      }
    }
  }
//#pragma endscop
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)A,&A_buf[121L - ((long )0)],(8 * 121L),sizeof(double ) * ((unsigned long )14158L),14158L);
}
}
