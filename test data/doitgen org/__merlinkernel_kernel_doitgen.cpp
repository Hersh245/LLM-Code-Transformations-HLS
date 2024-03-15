#define __constant
#define __kernel
#define __global
#include "memcpy_128_1d.h"
#include "memcpy_512_1d.h"
#define SIZE_1 20
#define SIZE_2 30
#include "memcpy_128_3d.h"
#undef SIZE_1
#undef SIZE_2
#define SIZE_1 30
#include "memcpy_128_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"







// Original: #pragma ACCEL kernel
static int kernel_doitgen_dummy_pos;
extern "C" { 

void kernel_doitgen(class ap_uint< 128 > A[7500],class ap_uint< 128 > C4[450],class ap_uint< 512 > sum[4])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=7500 bundle=merlin_gmem_kernel_doitgen_128_A
  
#pragma HLS INTERFACE m_axi port=C4 offset=slave depth=450 bundle=merlin_gmem_kernel_doitgen_128_0
  
#pragma HLS INTERFACE m_axi port=sum offset=slave depth=4 bundle=merlin_gmem_kernel_doitgen_512_sum
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=C4 bundle=control
  
#pragma HLS INTERFACE s_axilite port=sum bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=sum
  
#pragma HLS DATA_PACK VARIABLE=C4
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=sum depth=30 max_depth=30
  
#pragma ACCEL interface variable=C4 depth=30,30 max_depth=30,30
  
#pragma ACCEL interface variable=A depth=25,20,30 max_depth=25,20,30
  double C4_6_0_buf[30][30];
  
#pragma HLS array_partition variable=C4_6_0_buf cyclic factor=2 dim=2
  
#pragma HLS array_partition variable=C4_6_0_buf complete dim=1
  double A_buf[25][20][30];
  
#pragma HLS array_partition variable=A_buf complete dim=3
  double sum_buf[30];
  
#pragma HLS array_partition variable=sum_buf complete dim=1
  int r;
  int q;
  int p;
  int s;
// Existing HLS partition: #pragma HLS array_partition variable=sum_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&sum_buf[0],(class ap_uint< 512 > *)sum,(0 * 8),sizeof(double ) * ((unsigned long )30),30L);
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 2 dim=3
  memcpy_wide_bus_read_double_3d_20_30_128(A_buf,0,0,0,(class ap_uint< 128 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )15000L),15000L);
{
    memcpy_wide_bus_read_double_2d_30_128(C4_6_0_buf,0,0,(class ap_uint< 128 > *)C4,(0 * 8),sizeof(double ) * ((unsigned long )900L),900L);
  }
//#pragma scop
  merlinL6:
//#pragma scop
  for (r = 0; r < 25; r++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL5:
    for (q = 0; q < 20; q++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
      merlinL4:
      for (p = 0; p < 30; p++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
        
#pragma HLS dependence variable=sum_buf array inter false
        
#pragma HLS pipeline
        double sum_tmp = 0.0;
        double _sum_tmp_rdc[30];
        
#pragma HLS array_partition variable=_sum_tmp_rdc complete dim=1
{
          int s_0;
          merlinL3:
          for (s_0 = 0; s_0 <= 29; ++s_0) 
// Original: #pragma ACCEL PARALLEL COMPLETE
// Original: #pragma ACCEL ARRAY_PARTITION OFF
{
            
#pragma HLS unroll
            _sum_tmp_rdc[s_0] = ((double )0);
          }
        }
        merlinL2:
        for (s = 0; s < 30; s++) 
// Original: #pragma ACCEL PARALLEL reduction=sum_tmp FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=sum_tmp
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PARALLEL COMPLETE
{
          
#pragma HLS unroll
          _sum_tmp_rdc[s] += A_buf[r][q][s] * C4_6_0_buf[s][p];
        }
{
          int s_1;
          merlinL1:
          for (s_1 = 0; s_1 <= 29; ++s_1) 
// Original: #pragma ACCEL ARRAY_PARTITION OFF
// Original: #pragma ACCEL PARALLEL COMPLETE
{
            
#pragma HLS unroll
            sum_tmp += _sum_tmp_rdc[s_1];
          }
        }
        sum_buf[p] = sum_tmp;
      }
      merlinL0:
      for (p = 0; p < 30; p++) 
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        
#pragma HLS unroll
        A_buf[r][q][p] = sum_buf[p];
      }
    }
  }
//#pragma endscop
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 2 dim=3
  memcpy_wide_bus_write_double_3d_20_30_128((class ap_uint< 128 > *)A,A_buf,0,0,0,(8 * 0),sizeof(double ) * ((unsigned long )15000L),15000L);
// Existing HLS partition: #pragma HLS array_partition variable=sum_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)sum,&sum_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )30),30L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=C4_6_0_buf cyclic factor = 2 dim=2
