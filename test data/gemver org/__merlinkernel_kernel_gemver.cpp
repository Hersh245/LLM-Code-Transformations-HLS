#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#define SIZE_1 120
#include "memcpy_512_2d.h"
#undef SIZE_1
#include <string.h> 
#include<hls_stream.h>


#include "merlin_type_define.h"






// Original: #pragma ACCEL kernel
static class hls::stream< double  > __merlinkernel_gemver_z_ch;

void __merlinkernel_gemver_z_streaming(double z[120])
{
  
#pragma HLS INLINE off
  long merlin_it_0;
  merlinL1:
  for (merlin_it_0 = 0L; merlin_it_0 < 120L; ++merlin_it_0) 
// Original: #pragma ACCEL pipeline
{
    
#pragma HLS pipeline
    long merlin_it_1;
    merlinL0:
    for (merlin_it_1 = 0L; merlin_it_1 < ((long )1); ++merlin_it_1) 
// Original: #pragma ACCEL parallel
{
      
#pragma HLS unroll
      double merlin_tmp_2;
      merlin_tmp_2 = z[((long )0) + merlin_it_0 * ((long )1) + merlin_it_1];
      __merlinkernel_gemver_z_ch . write(merlin_tmp_2);
    }
  }
}

void __merlinkernel_gemver_computation(double alpha,double beta,class ap_uint< 512 > A[1800],class ap_uint< 512 > u1[15],class ap_uint< 512 > v1[15],class ap_uint< 512 > u2[15],class ap_uint< 512 > v2[15],class ap_uint< 512 > w[15],class ap_uint< 512 > x[15],class ap_uint< 512 > y[15],double z[120])
{
  
#pragma HLS inline off
  double v2_10_0_buf[120];
  
#pragma HLS array_partition variable=v2_10_0_buf cyclic factor=8 dim=1
  double u2_10_0_buf[120];
  
#pragma HLS array_partition variable=u2_10_0_buf cyclic factor=8 dim=1
  double v1_10_0_buf[120];
  
#pragma HLS array_partition variable=v1_10_0_buf cyclic factor=8 dim=1
  double u1_10_0_buf[120];
  
#pragma HLS array_partition variable=u1_10_0_buf cyclic factor=8 dim=1
  double y_11_0_buf[120];
  
#pragma HLS array_partition variable=y_11_0_buf cyclic factor=8 dim=1
  double w_buf[120];
  
#pragma HLS array_partition variable=w_buf cyclic factor=8 dim=1
  double A_buf[120][120];
  
#pragma HLS array_partition variable=A_buf cyclic factor=8 dim=2
  double x_buf[120];
// Existing HLS partition: #pragma HLS array_partition variable=x_buf cyclic factor = 8 dim=1
  
#pragma HLS array_partition variable=x_buf cyclic factor=8 dim=1
  memcpy_wide_bus_read_double_512(&x_buf[0],(class ap_uint< 512 > *)x,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_120_512(A_buf,0,0,(class ap_uint< 512 > *)A,(0 * 8),sizeof(double ) * ((unsigned long )14400L),14400L);
  int i;
  int j;
{
    memcpy_wide_bus_read_double_512(&u1_10_0_buf[0],(class ap_uint< 512 > *)u1,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  }
{
    memcpy_wide_bus_read_double_512(&v1_10_0_buf[0],(class ap_uint< 512 > *)v1,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  }
{
    memcpy_wide_bus_read_double_512(&u2_10_0_buf[0],(class ap_uint< 512 > *)u2,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  }
{
    memcpy_wide_bus_read_double_512(&v2_10_0_buf[0],(class ap_uint< 512 > *)v2,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  }
  
#pragma scop
  merlinL8:
  for (i = 0; i < 120; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    merlinL7:
    for (j = 0; j < 120; j++) 
// Original: #pragma ACCEL PARALLEL reduction=A FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=A
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=A_buf array inter false
      
#pragma HLS pipeline
      A_buf[i][j] += u1_10_0_buf[i] * v1_10_0_buf[j] + u2_10_0_buf[i] * v2_10_0_buf[j];
    }
  }
{
    memcpy_wide_bus_read_double_512(&y_11_0_buf[0],(class ap_uint< 512 > *)y,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  }
  merlinL6:
  for (i = 0; i < 120; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
    merlinL5:
    for (j = 0; j < 120; j++) 
// Original: #pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=x
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      x_buf[i] += beta * A_buf[j][i] * y_11_0_buf[j];
    }
  }
  merlinL4:
  for (i = 0; i < 120; i++) 
// Original: #pragma ACCEL PARALLEL reduction=x FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=x
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE AUTO
{
    
#pragma HLS dependence variable=x_buf array inter false
    
#pragma HLS pipeline
    x_buf[i] += __merlinkernel_gemver_z_ch . read();
  }
// Existing HLS partition: #pragma HLS array_partition variable=w_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&w_buf[0],(class ap_uint< 512 > *)w,(0 * 8),sizeof(double ) * ((unsigned long )120),120L);
  merlinL3:
  for (i = 0; i < 120; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
{
    merlinL2:
    for (j = 0; j < 120; j++) 
// Original: #pragma ACCEL PARALLEL reduction=w FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=w
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      w_buf[i] += alpha * A_buf[i][j] * x_buf[j];
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=w_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)w,&w_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )120),120L);
  
#pragma endscop
// Existing HLS partition: #pragma HLS array_partition variable=x_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)x,&x_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )120),120L);
// Existing HLS partition: #pragma HLS array_partition variable=A_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_120_512((class ap_uint< 512 > *)A,A_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )14400L),14400L);
}
// Existing HLS partition: #pragma HLS array_partition variable=u1_10_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=u2_10_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=v1_10_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=v2_10_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=y_11_0_buf cyclic factor = 8 dim=1
static int kernel_gemver_dummy_pos;
extern "C" { 

void kernel_gemver(double alpha,double beta,class ap_uint< 512 > A[1800],class ap_uint< 512 > u1[15],class ap_uint< 512 > v1[15],class ap_uint< 512 > u2[15],class ap_uint< 512 > v2[15],class ap_uint< 512 > w[15],class ap_uint< 512 > x[15],class ap_uint< 512 > y[15],double z[120])
{
  
#pragma HLS INTERFACE m_axi port=A offset=slave depth=1800 bundle=merlin_gmem_kernel_gemver_512_A
  
#pragma HLS INTERFACE m_axi port=u1 offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_0
  
#pragma HLS INTERFACE m_axi port=u2 offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_1
  
#pragma HLS INTERFACE m_axi port=v1 offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_2
  
#pragma HLS INTERFACE m_axi port=v2 offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_3
  
#pragma HLS INTERFACE m_axi port=w offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_w
  
#pragma HLS INTERFACE m_axi port=x offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_x
  
#pragma HLS INTERFACE m_axi port=y offset=slave depth=15 bundle=merlin_gmem_kernel_gemver_512_4
  
#pragma HLS INTERFACE m_axi port=z offset=slave depth=120 bundle=merlin_gmem_kernel_gemver_64_0
  
#pragma HLS INTERFACE s_axilite port=A bundle=control
  
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
  
#pragma HLS INTERFACE s_axilite port=beta bundle=control
  
#pragma HLS INTERFACE s_axilite port=u1 bundle=control
  
#pragma HLS INTERFACE s_axilite port=u2 bundle=control
  
#pragma HLS INTERFACE s_axilite port=v1 bundle=control
  
#pragma HLS INTERFACE s_axilite port=v2 bundle=control
  
#pragma HLS INTERFACE s_axilite port=w bundle=control
  
#pragma HLS INTERFACE s_axilite port=x bundle=control
  
#pragma HLS INTERFACE s_axilite port=y bundle=control
  
#pragma HLS INTERFACE s_axilite port=z bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=y
  
#pragma HLS DATA_PACK VARIABLE=x
  
#pragma HLS DATA_PACK VARIABLE=w
  
#pragma HLS DATA_PACK VARIABLE=v2
  
#pragma HLS DATA_PACK VARIABLE=u2
  
#pragma HLS DATA_PACK VARIABLE=v1
  
#pragma HLS DATA_PACK VARIABLE=u1
  
#pragma HLS DATA_PACK VARIABLE=A
  
#pragma ACCEL interface variable=y depth=120 max_depth=120
  
#pragma ACCEL interface variable=x depth=120 max_depth=120
  
#pragma ACCEL interface variable=w depth=120 max_depth=120
  
#pragma ACCEL interface variable=v2 depth=120 max_depth=120
  
#pragma ACCEL interface variable=u2 depth=120 max_depth=120
  
#pragma ACCEL interface variable=v1 depth=120 max_depth=120
  
#pragma ACCEL interface variable=u1 depth=120 max_depth=120
  
#pragma ACCEL interface variable=A depth=120,120 max_depth=120,120
  
#pragma ACCEL interface variable=z max_depth=120 depth=120 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=64
  
#pragma HLS dataflow
  
#pragma HLS stream variable=__merlinkernel_gemver_z_ch
  __merlinkernel_gemver_z_streaming(z);
  __merlinkernel_gemver_computation(alpha,beta,A,u1,v1,u2,v2,w,x,y,0);
}
}
