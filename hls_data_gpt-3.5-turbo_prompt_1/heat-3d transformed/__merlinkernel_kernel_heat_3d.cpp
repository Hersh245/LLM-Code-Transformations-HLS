#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#include <string.h> 

#include "merlin_type_define.h"




// Original: #pragma ACCEL kernel
static int kernel_heat_3d_dummy_pos;
extern "C" { 

void kernel_heat_3d(class ap_uint< 512 > A[1000],class ap_uint< 512 > B[1000])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=1000 bundle=merlin_gmem_kernel_heat_3d_512_A
  
#pragma HLS INTERFACE m_axi port=B offset=slave depth=1000 bundle=merlin_gmem_kernel_heat_3d_512_B
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=B bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=B
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=B depth=20,20,20 max_depth=20,20,20
  
#pragma ACCEL interface variable=A depth=20,20,20 max_depth=20,20,20
  double A_buf[7958];
  
#pragma HLS array_partition variable=A_buf cyclic factor=8 dim=1
  double B_buf[7958];
  
#pragma HLS array_partition variable=B_buf cyclic factor=8 dim=1
  int t;
  int i;
  int j;
  int k;
// Existing HLS partition: #pragma HLS array_partition variable=B_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&B_buf[0],(class ap_uint< 512 > *)B,(21L * 8),sizeof(double ) * ((unsigned long )7958L),7958L);
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&A_buf[0],(class ap_uint< 512 > *)A,(21L * 8),sizeof(double ) * ((unsigned long )7958L),7958L);
  merlinL6:
  for (t = 1; t <= 40; t++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL5:
    for (i = 1; i < 20 - 1; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
      merlinL4:
      for (j = 1; j < 20 - 1; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
{
        merlinL3:
        for (k = 1; k < 20 - 1; k++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
          
#pragma HLS dependence variable=B_buf array inter false
          
#pragma HLS pipeline
          B_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] = 0.125 * (A_buf[379L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * A_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + A_buf[- 421L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + 0.125 * (A_buf[- 1L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * A_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + A_buf[- 41L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + 0.125 * (A_buf[- 20L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * A_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + A_buf[- 22L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + A_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)];
        }
      }
    }
    merlinL2:
    for (i = 1; i < 20 - 1; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L4}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L4}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L4}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L4}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L4}
{
      merlinL1:
      for (j = 1; j < 20 - 1; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L5}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L5}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
{
        merlinL0:
        for (k = 1; k < 20 - 1; k++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L6}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L6}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L6}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L6}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L6}
{
          
#pragma HLS dependence variable=A_buf array inter false
          
#pragma HLS pipeline
          A_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] = 0.125 * (B_buf[379L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * B_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + B_buf[- 421L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + 0.125 * (B_buf[- 1L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * B_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + B_buf[- 41L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + 0.125 * (B_buf[- 20L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] - 2.0 * B_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)] + B_buf[- 22L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)]) + B_buf[- 21L + ((long )i) * 400L + ((long )j) * 20L + ((long )k)];
        }
      }
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)A,&A_buf[421L - 21L],(8 * 421L),sizeof(double ) * ((unsigned long )7158L),7158L);
// Existing HLS partition: #pragma HLS array_partition variable=B_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)B,&B_buf[421L - 21L],(8 * 421L),sizeof(double ) * ((unsigned long )7158L),7158L);
}
}
