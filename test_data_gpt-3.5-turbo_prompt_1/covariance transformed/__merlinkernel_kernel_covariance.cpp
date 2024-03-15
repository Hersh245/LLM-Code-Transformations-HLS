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
static int kernel_covariance_dummy_pos;
extern "C" { 

void kernel_covariance(double float_n,class ap_uint< 512 > data[1000],class ap_uint< 512 > cov[800],class ap_uint< 512 > mean[10])
{
  
#pragma HLS INTERFACE m_axi port=cov offset=slave depth=800 bundle=merlin_gmem_kernel_covariance_512_cov
  
#pragma HLS INTERFACE m_axi port=data offset=slave depth=1000 bundle=merlin_gmem_kernel_covariance_512_data
  
#pragma HLS INTERFACE m_axi port=mean offset=slave depth=10 bundle=merlin_gmem_kernel_covariance_512_mean
  
#pragma HLS INTERFACE s_axilite port=cov bundle=control
  
#pragma HLS INTERFACE s_axilite port=data bundle=control
  
#pragma HLS INTERFACE s_axilite port=float_n bundle=control
  
#pragma HLS INTERFACE s_axilite port=mean bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=mean
  
#pragma HLS DATA_PACK VARIABLE=cov
  
#pragma HLS DATA_PACK VARIABLE=data
  
#pragma ACCEL interface variable=mean depth=80 max_depth=80
  
#pragma ACCEL interface variable=cov depth=80,80 max_depth=80,80
  
#pragma ACCEL interface variable=data depth=100,80 max_depth=100,80
  double cov_buf[80][80];
  
#pragma HLS array_partition variable=cov_buf cyclic factor=8 dim=2
  double data_buf[100][80];
  
#pragma HLS array_partition variable=data_buf cyclic factor=8 dim=2
  double mean_buf[80];
// Existing HLS partition: #pragma HLS array_partition variable=mean_buf cyclic factor = 8 dim=1
  
#pragma HLS array_partition variable=mean_buf cyclic factor=8 dim=1
  memcpy_wide_bus_read_double_512(&mean_buf[0],(class ap_uint< 512 > *)mean,(0 * 8),sizeof(double ) * ((unsigned long )80),80L);
// Existing HLS partition: #pragma HLS array_partition variable=data_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(data_buf,0,0,(class ap_uint< 512 > *)data,(0 * 8),sizeof(double ) * ((unsigned long )8000L),8000L);
  int i;
  int j;
  int k;
// Calculate mean
  merlinL6:
// Calculate mean
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
    merlinL5:
    for (i = 0; i < 100; i++) 
// Original: #pragma ACCEL PARALLEL reduction=mean FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=mean
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      mean_buf[j] += data_buf[i][j];
    }
    mean_buf[j] /= float_n;
  }
// Center the data
  merlinL4:
// Center the data
  for (i = 0; i < 100; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
    merlinL3:
    for (j = 0; j < 80; j++) 
// Original: #pragma ACCEL PARALLEL reduction=data FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=data
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=data_buf array inter false
      
#pragma HLS pipeline
      data_buf[i][j] -= mean_buf[j];
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=cov_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_read_double_2d_80_512(cov_buf,0,0,(class ap_uint< 512 > *)cov,(0 * 8),sizeof(double ) * ((unsigned long )6400L),6400L);
// Calculate covariance
  merlinL2:
// Calculate covariance
  for (i = 0; i < 80; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
    merlinL1:
    for (j = i; j < 80; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L5}
{
      
#pragma HLS LOOP_TRIPCOUNT max=80
      cov_buf[i][j] = 0.0;
      merlinL0:
      for (k = 0; k < 100; k++) 
// Original: #pragma ACCEL PARALLEL reduction=cov FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL REDUCTION SCHEME=auto VARIABLE=cov
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS pipeline
        cov_buf[i][j] += data_buf[k][i] * data_buf[k][j];
      }
{
        cov_buf[i][j] /= float_n - 1.0;
        cov_buf[j][i] = cov_buf[i][j];
      }
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=cov_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)cov,cov_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )6400L),6400L);
// Existing HLS partition: #pragma HLS array_partition variable=mean_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)mean,&mean_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )80),80L);
// Existing HLS partition: #pragma HLS array_partition variable=data_buf cyclic factor = 8 dim=2
  memcpy_wide_bus_write_double_2d_80_512((class ap_uint< 512 > *)data,data_buf,0,0,(8 * 0),sizeof(double ) * ((unsigned long )8000L),8000L);
}
}
