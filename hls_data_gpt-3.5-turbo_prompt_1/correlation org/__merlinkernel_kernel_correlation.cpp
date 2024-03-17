#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#define SIZE_1 80
#include "memcpy_512_2d.h"
#undef SIZE_1
#include <string.h> 

#include <math.h>
#include "merlin_type_define.h"






// Original: #pragma ACCEL kernel
static int kernel_correlation_dummy_pos;

static void merlin_memcpy_0(double dst[2][6399],int dst_idx_0,int dst_idx_1,double src[6400],int src_idx_0,unsigned int len,unsigned int max_len)
{
  
#pragma HLS inline off
  
#pragma HLS function_instantiate variable=dst_idx_0,dst_idx_1,src_idx_0
  unsigned long i;
  unsigned long total_offset1 = (0 * 2 + dst_idx_0) * 6399 + dst_idx_1;
  unsigned long total_offset2 = 0 * 6400 + src_idx_0;
  merlinL0:
  for (i = 0; i < len / 8; ++i) {
    
#pragma HLS PIPELINE II=1
    
#pragma HLS LOOP_TRIPCOUNT max=6400
    dst[(total_offset1 + i) / 6399][(total_offset1 + i) % 6399] = src[total_offset2 + i];
  }
}

static void merlin_memcpy_1(double dst[6400],int dst_idx_0,double src[2][6399],int src_idx_0,int src_idx_1,unsigned int len,unsigned int max_len)
{
  
#pragma HLS inline off
  
#pragma HLS function_instantiate variable=dst_idx_0,src_idx_0,src_idx_1
  unsigned long i;
  unsigned long total_offset1 = 0 * 6400 + dst_idx_0;
  unsigned long total_offset2 = (0 * 2 + src_idx_0) * 6399 + src_idx_1;
  merlinL1:
  for (i = 0; i < len / 8; ++i) {
    
#pragma HLS PIPELINE II=1
    
#pragma HLS LOOP_TRIPCOUNT max=6400
    dst[total_offset1 + i] = src[(total_offset2 + i) / 6399][(total_offset2 + i) % 6399];
  }
}
extern "C" { 

void kernel_correlation(double float_n,class ap_uint< 512 > data[1000],double corr[6400],class ap_uint< 512 > mean[10],class ap_uint< 512 > stddev[10])
{
  
#pragma HLS INTERFACE m_axi port=corr offset=slave depth=6400 bundle=merlin_gmem_kernel_correlation_64_corr
  
#pragma HLS INTERFACE m_axi port=data offset=slave depth=1000 bundle=merlin_gmem_kernel_correlation_512_data
  
#pragma HLS INTERFACE m_axi port=mean offset=slave depth=10 bundle=merlin_gmem_kernel_correlation_512_mean
  
#pragma HLS INTERFACE m_axi port=stddev offset=slave depth=10 bundle=merlin_gmem_kernel_correlation_512_stddev
  
#pragma HLS INTERFACE s_axilite port=corr bundle=control
  
#pragma HLS INTERFACE s_axilite port=data bundle=control
  
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
  
#pragma HLS INTERFACE s_axilite port=mean bundle=control
  
#pragma HLS INTERFACE s_axilite port=stddev bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=stddev
  
#pragma HLS DATA_PACK VARIABLE=mean
  
#pragma HLS DATA_PACK VARIABLE=data
  
#pragma ACCEL interface variable=stddev depth=80 max_depth=80
  
#pragma ACCEL interface variable=mean depth=80 max_depth=80
  
#pragma ACCEL interface variable=corr depth=80,80 max_depth=80,80
  
#pragma ACCEL interface variable=data depth=100,80 max_depth=100,80
  double corr_buf[2][6399];
  
#pragma HLS array_partition variable=corr_buf complete dim=1
  double stddev_buf[80];
  
#pragma HLS array_partition variable=stddev_buf cyclic factor=8 dim=1
  double data_buf[100][80];
  
#pragma HLS array_partition variable=data_buf cyclic factor=8 dim=2
  double mean_buf[80];
// Existing HLS partition: #pragma HLS array_partition variable=mean_buf cyclic factor = 8 dim=1
  
#pragma HLS array_partition variable=mean_buf cyclic factor=8 dim=1
  memcpy_wide_bus_read_double_512(&mean_buf[0],(class ap_uint< 512 > *)mean,(0 * 8),sizeof(double ) * ((unsigned long )80),80L);
// Existing HLS partition: #pragma HLS array_partition variable=data_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(data_buf,0,0,(class ap_uint< 512 > *)data,(0 * 8),sizeof(double ) * ((unsigned long )8000L),8000L);
// Existing HLS partition: #pragma HLS array_partition variable=stddev_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&stddev_buf[0],(class ap_uint< 512 > *)stddev,(0 * 8),sizeof(double ) * ((unsigned long )80),80L);
  merlin_memcpy_0(corr_buf,0,0,corr,0,sizeof(double ) * ((unsigned long )6400L),51200UL);
  int i;
  int j;
  int k;
  double eps = 0.1;
  merlinL10:
  for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    mean_buf[j] = 0.0;
    merlinL9:
    for (i = 0; i < 100; i++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      mean_buf[j] += data_buf[i][j];
    }
    mean_buf[j] /= float_n;
  }
  merlinL8:
  for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
    stddev_buf[j] = 0.0;
    merlinL7:
    for (i = 0; i < 100; i++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      stddev_buf[j] += pow(data_buf[i][j] - mean_buf[j],(double )2);
    }
{
      stddev_buf[j] /= float_n;
      stddev_buf[j] = sqrt(stddev_buf[j]);
    }
/* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
    if (stddev_buf[j] <= eps) {
      stddev_buf[j] = 1.0;
    }
     else {
      stddev_buf[j] = stddev_buf[j];
    }
  }
/* Center and reduce the column vectors. */
  merlinL6:
/* Center and reduce the column vectors. */
  for (i = 0; i < 100; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
    merlinL5:
    for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=data_buf array inter false
      
#pragma HLS pipeline
      data_buf[i][j] -= mean_buf[j];
      data_buf[i][j] /= sqrt(float_n) * stddev_buf[j];
    }
  }
/* Calculate the m * m correlation matrix. */
  merlinL4:
/* Calculate the m * m correlation matrix. */
  for (i = 0; i < 80 - 1; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L3}
{
    corr_buf[0][((long )i) * 81L] = 1.0;
    merlinL3:
    for (j = i + 1; j < 80; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L7}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L7}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L7}
{
      
#pragma HLS LOOP_TRIPCOUNT max=79
      corr_buf[0][((long )i) * 80L + ((long )j)] = 0.0;
      merlinL2:
      for (k = 0; k < 100; k++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7_0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7_0}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS pipeline
        corr_buf[0][((long )i) * 80L + ((long )j)] += data_buf[k][i] * data_buf[k][j];
      }
      corr_buf[0][((long )j) * 80L + ((long )i)] = corr_buf[0][((long )i) * 80L + ((long )j)];
    }
  }
  corr_buf[1][0] = 1.0;
// Existing HLS partition: #pragma HLS array_partition variable=mean_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)mean,&mean_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )80),80L);
// Existing HLS partition: #pragma HLS array_partition variable=data_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)data,data_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )8000L),8000L);
// Existing HLS partition: #pragma HLS array_partition variable=stddev_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)stddev,&stddev_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )80),80L);
  merlin_memcpy_1(corr,0,corr_buf,0,0,sizeof(double ) * ((unsigned long )6400L),51200UL);
}
}
