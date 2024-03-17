#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#define SIZE_1 80
#include "memcpy_512_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"





// Original: #pragma ACCEL kernel
static int kernel_fdtd_2d_dummy_pos;
extern "C" { 

void kernel_fdtd_2d(class ap_uint< 512 > ex[600],class ap_uint< 512 > ey[600],class ap_uint< 512 > hz[600],class ap_uint< 512 > _fict_[5])
{
  
#pragma HLS INTERFACE m_axi port=_fict_ offset=slave depth=5 bundle=merlin_gmem_kernel_fdtd_2d_512_0
  
#pragma HLS INTERFACE m_axi port=ex offset=slave depth=600 bundle=merlin_gmem_kernel_fdtd_2d_512_ex
  
#pragma HLS INTERFACE m_axi port=ey offset=slave depth=600 bundle=merlin_gmem_kernel_fdtd_2d_512_ey
  
#pragma HLS INTERFACE m_axi port=hz offset=slave depth=600 bundle=merlin_gmem_kernel_fdtd_2d_512_hz
  
#pragma HLS INTERFACE s_axilite port=_fict_ bundle=control
  
#pragma HLS INTERFACE s_axilite port=ex bundle=control
  
#pragma HLS INTERFACE s_axilite port=ey bundle=control
  
#pragma HLS INTERFACE s_axilite port=hz bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=_fict_
  
#pragma HLS DATA_PACK VARIABLE=hz
  
#pragma HLS DATA_PACK VARIABLE=ey
  
#pragma HLS DATA_PACK VARIABLE=ex
  
#pragma ACCEL interface variable=_fict_ depth=40 max_depth=40
  
#pragma ACCEL interface variable=hz depth=60,80 max_depth=60,80
  
#pragma ACCEL interface variable=ey depth=60,80 max_depth=60,80
  
#pragma ACCEL interface variable=ex depth=60,80 max_depth=60,80
  double _fict_3_0_buf[40];
  
#pragma HLS array_partition variable=_fict_3_0_buf cyclic factor=8 dim=1
  double ex_buf[60][80];
  
#pragma HLS array_partition variable=ex_buf cyclic factor=8 dim=2
  double hz_buf[60][80];
  
#pragma HLS array_partition variable=hz_buf cyclic factor=8 dim=2
  double ey_buf[60][80];
  
#pragma HLS array_partition variable=ey_buf cyclic factor=8 dim=2
  int t;
  int i;
  int j;
// Existing HLS partition: #pragma HLS array_partition variable=ey_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(ey_buf,0,0,(class ap_uint< 512 > *)ey,(0 * 8),sizeof(double ) * ((unsigned long )4800),4800L);
// Existing HLS partition: #pragma HLS array_partition variable=hz_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(hz_buf,0,0,(class ap_uint< 512 > *)hz,(0 * 8),sizeof(double ) * ((unsigned long )4800L),4800L);
// Existing HLS partition: #pragma HLS array_partition variable=ex_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(ex_buf,0,0,(class ap_uint< 512 > *)ex,(0 * 8),sizeof(double ) * ((unsigned long )4800L),4800L);
{
    memcpy_wide_bus_read_double_512(&_fict_3_0_buf[0],(class ap_uint< 512 > *)_fict_,(0 * 8),sizeof(double ) * ((unsigned long )40),40L);
  }
  merlinL7:
  for (t = 0; t < 40; t++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL6:
    for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=ey_buf array inter false
      
#pragma HLS pipeline
      ey_buf[0][j] = _fict_3_0_buf[t];
    }
    merlinL5:
    for (i = 1; i < 60; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_1}
{
      merlinL4:
      for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_1_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=ey_buf array inter false
        
#pragma HLS pipeline
        ey_buf[i][j] = ey_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[- 1 + i][j]);
      }
    }
    merlinL3:
    for (i = 0; i < 60; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_2}
{
      merlinL2:
      for (j = 1; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2_0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_2_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=ex_buf array inter false
        
#pragma HLS pipeline
        ex_buf[i][j] = ex_buf[i][j] - 0.5 * (hz_buf[i][j] - hz_buf[i][- 1 + j]);
      }
    }
    merlinL1:
    for (i = 0; i < 59; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0_3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0_3}
{
      merlinL0:
      for (j = 0; j < 79; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3_0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0_3_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS dependence variable=hz_buf array inter false
        
#pragma HLS pipeline
        hz_buf[i][j] = hz_buf[i][j] - 0.7 * (ex_buf[i][1 + j] - ex_buf[i][j] + ey_buf[1 + i][j] - ey_buf[i][j]);
      }
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=ex_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)ex,ex_buf,0,1,(8 * 1L),sizeof(double ) * ((unsigned long )4799L),4799L);
// Existing HLS partition: #pragma HLS array_partition variable=hz_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)hz,hz_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )4719L),4719L);
// Existing HLS partition: #pragma HLS array_partition variable=ey_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)ey,ey_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )4800),4800L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=_fict_3_0_buf cyclic factor = 8 dim=1
