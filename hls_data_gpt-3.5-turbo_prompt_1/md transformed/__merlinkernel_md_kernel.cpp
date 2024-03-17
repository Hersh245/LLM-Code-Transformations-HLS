#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#define SIZE_1 16
#include "memcpy_512_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"





// Original: #pragma ACCEL kernel
static int md_kernel_dummy_pos;
extern "C" { 

void md_kernel(class ap_uint< 512 > force_x[32],class ap_uint< 512 > force_y[32],class ap_uint< 512 > force_z[32],class ap_uint< 512 > position_x[32],class ap_uint< 512 > position_y[32],class ap_uint< 512 > position_z[32],class ap_uint< 512 > NL[256])
{
  
#pragma HLS INTERFACE m_axi port=NL offset=slave depth=256 bundle=merlin_gmem_md_kernel_512_0
  
#pragma HLS INTERFACE m_axi port=force_x offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_force_x
  
#pragma HLS INTERFACE m_axi port=force_y offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_force_y
  
#pragma HLS INTERFACE m_axi port=force_z offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_force_z
  
#pragma HLS INTERFACE m_axi port=position_x offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_position_x
  
#pragma HLS INTERFACE m_axi port=position_y offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_position_y
  
#pragma HLS INTERFACE m_axi port=position_z offset=slave depth=32 bundle=merlin_gmem_md_kernel_512_position_z
  
#pragma HLS INTERFACE s_axilite port=NL bundle=control
  
#pragma HLS INTERFACE s_axilite port=force_x bundle=control
  
#pragma HLS INTERFACE s_axilite port=force_y bundle=control
  
#pragma HLS INTERFACE s_axilite port=force_z bundle=control
  
#pragma HLS INTERFACE s_axilite port=position_x bundle=control
  
#pragma HLS INTERFACE s_axilite port=position_y bundle=control
  
#pragma HLS INTERFACE s_axilite port=position_z bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=NL
  
#pragma HLS DATA_PACK VARIABLE=position_z
  
#pragma HLS DATA_PACK VARIABLE=position_y
  
#pragma HLS DATA_PACK VARIABLE=position_x
  
#pragma HLS DATA_PACK VARIABLE=force_z
  
#pragma HLS DATA_PACK VARIABLE=force_y
  
#pragma HLS DATA_PACK VARIABLE=force_x
  
#pragma ACCEL interface variable=NL depth=4096 max_depth=4096
  
#pragma ACCEL interface variable=position_z depth=256 max_depth=256
  
#pragma ACCEL interface variable=position_y depth=256 max_depth=256
  
#pragma ACCEL interface variable=position_x depth=256 max_depth=256
  
#pragma ACCEL interface variable=force_z depth=256 max_depth=256
  
#pragma ACCEL interface variable=force_y depth=256 max_depth=256
  
#pragma ACCEL interface variable=force_x depth=256 max_depth=256
  int NL_3_0_buf[256][16];
  
#pragma HLS array_partition variable=NL_3_0_buf cyclic factor=16 dim=2
  double position_z_3_0_buf[256];
  
#pragma HLS array_partition variable=position_z_3_0_buf cyclic factor=8 dim=1
  double position_y_3_0_buf[256];
  
#pragma HLS array_partition variable=position_y_3_0_buf cyclic factor=8 dim=1
  double position_x_3_0_buf[256];
  
#pragma HLS array_partition variable=position_x_3_0_buf cyclic factor=8 dim=1
  double force_z_buf[256];
  
#pragma HLS array_partition variable=force_z_buf cyclic factor=8 dim=1
  double force_y_buf[256];
  
#pragma HLS array_partition variable=force_y_buf cyclic factor=8 dim=1
  double force_x_buf[256];
  
#pragma HLS array_partition variable=force_x_buf cyclic factor=8 dim=1
  double delx;
  double dely;
  double delz;
  double r2inv;
  double r6inv;
  double potential;
  double force;
  double j_x;
  double j_y;
  double j_z;
  double i_x;
  double i_y;
  double i_z;
  double fx;
  double fy;
  double fz;
  int i;
  int j;
  int jidx;
// Existing HLS partition: #pragma HLS array_partition variable=force_x_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&force_x_buf[0],(class ap_uint< 512 > *)force_x,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
// Existing HLS partition: #pragma HLS array_partition variable=force_y_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&force_y_buf[0],(class ap_uint< 512 > *)force_y,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
// Existing HLS partition: #pragma HLS array_partition variable=force_z_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_read_double_512(&force_z_buf[0],(class ap_uint< 512 > *)force_z,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
{
    memcpy_wide_bus_read_double_512(&position_x_3_0_buf[0],(class ap_uint< 512 > *)position_x,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
{
    memcpy_wide_bus_read_double_512(&position_y_3_0_buf[0],(class ap_uint< 512 > *)position_y,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
{
    memcpy_wide_bus_read_double_512(&position_z_3_0_buf[0],(class ap_uint< 512 > *)position_z,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
{
    memcpy_wide_bus_read_int_2d_16_512(NL_3_0_buf,0,0,(class ap_uint< 512 > *)NL,(0 * 4),sizeof(int ) * ((unsigned long )4096),4096L);
  }
// Original label: loop_j:for(j = 0;j < 16;j++) {#pragma ACCEL PIPELINE auto{__PIPE__L0}loop_i:for(i = 0;i < 256;i++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;i_x = position_x_3_0_buf[i];i_y = position_y_3_0_buf[i];i_z = position_z_3_0_buf[i];fx =((double )0);fy =((double )0);fz =((double )0);jidx = NL_3_0_buf[i][j];tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;force_x_buf[i] += fx;force_y_buf[i] += fy;force_z_buf[i] += fz;}}
  merlinL1:
// Original label: loop_j:for(j = 0;j < 16;j++) {#pragma ACCEL PIPELINE auto{__PIPE__L0}loop_i:for(i = 0;i < 256;i++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;i_x = position_x_3_0_buf[i];i_y = position_y_3_0_buf[i];i_z = position_z_3_0_buf[i];fx =((double )0);fy =((double )0);fz =((double )0);jidx = NL_3_0_buf[i][j];tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;force_x_buf[i] += fx;force_y_buf[i] += fy;force_z_buf[i] += fz;}}
  for (j = 0; j < 16; j++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=16{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=16{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
// Original label: loop_i:for(i = 0;i < 256;i++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;i_x = position_x_3_0_buf[i];i_y = position_y_3_0_buf[i];i_z = position_z_3_0_buf[i];fx =((double )0);fy =((double )0);fz =((double )0);jidx = NL_3_0_buf[i][j];tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;force_x_buf[i] += fx;force_y_buf[i] += fy;force_z_buf[i] += fz;}
    merlinL0:
// Original label: loop_i:for(i = 0;i < 256;i++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;i_x = position_x_3_0_buf[i];i_y = position_y_3_0_buf[i];i_z = position_z_3_0_buf[i];fx =((double )0);fy =((double )0);fz =((double )0);jidx = NL_3_0_buf[i][j];tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;force_x_buf[i] += fx;force_y_buf[i] += fy;force_z_buf[i] += fz;}
    for (i = 0; i < 256; i++) 
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS dependence variable=force_x_buf array inter false
      
#pragma HLS dependence variable=force_y_buf array inter false
      
#pragma HLS dependence variable=force_z_buf array inter false
      
#pragma HLS pipeline
      double tmp_1;
      double tmp_0;
      double tmp;
      i_x = position_x_3_0_buf[i];
      i_y = position_y_3_0_buf[i];
      i_z = position_z_3_0_buf[i];
      fx = ((double )0);
      fy = ((double )0);
      fz = ((double )0);
      jidx = NL_3_0_buf[i][j];
      tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(::size_t )(jidx * 8));
      j_x = tmp;
      tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(::size_t )(jidx * 8));
      j_y = tmp_0;
      tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(::size_t )(jidx * 8));
      j_z = tmp_1;
      delx = i_x - j_x;
      dely = i_y - j_y;
      delz = i_z - j_z;
      r2inv = 1.0 / (delx * delx + dely * dely + delz * delz);
      r6inv = r2inv * r2inv * r2inv;
      potential = r6inv * (1.5 * r6inv - 2.0);
      force = r2inv * potential;
      fx += delx * force;
      fy += dely * force;
      fz += delz * force;
      force_x_buf[i] += fx;
      force_y_buf[i] += fy;
      force_z_buf[i] += fz;
    }
  }
// Existing HLS partition: #pragma HLS array_partition variable=force_z_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)force_z,&force_z_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )256),256L);
// Existing HLS partition: #pragma HLS array_partition variable=force_y_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)force_y,&force_y_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )256),256L);
// Existing HLS partition: #pragma HLS array_partition variable=force_x_buf cyclic factor = 8 dim=1
  memcpy_wide_bus_write_double_512((class ap_uint< 512 > *)force_x,&force_x_buf[0],(8 * 0),sizeof(double ) * ((unsigned long )256),256L);
}
}
// Existing HLS partition: #pragma HLS array_partition variable=NL_3_0_buf cyclic factor = 16 dim=2
// Existing HLS partition: #pragma HLS array_partition variable=position_x_3_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=position_y_3_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=position_z_3_0_buf cyclic factor = 8 dim=1
