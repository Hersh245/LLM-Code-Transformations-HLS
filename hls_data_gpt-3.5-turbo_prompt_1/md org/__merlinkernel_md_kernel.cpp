#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#include <string.h> 
#include<hls_stream.h>


#include "merlin_type_define.h"





// Original: #pragma ACCEL kernel
static class hls::stream< double  > __merlinmd_kernel_force_x_ch;
static class hls::stream< double  > __merlinmd_kernel_force_y_ch;
static class hls::stream< double  > __merlinmd_kernel_force_z_ch;
static class hls::stream< int  > __merlinmd_kernel_NL_ch;

void __merlinmd_kernel_force_x_streaming(double force_x[256])
{
  
#pragma HLS INLINE off
  long merlin_it_0;
  merlinL5:
  for (merlin_it_0 = 0L; merlin_it_0 < 256L; ++merlin_it_0) 
// Original: #pragma ACCEL pipeline
{
    
#pragma HLS dependence variable=force_x array inter false
    
#pragma HLS pipeline
    long merlin_it_1;
    merlinL4:
    for (merlin_it_1 = 0L; merlin_it_1 < ((long )1); ++merlin_it_1) 
// Original: #pragma ACCEL parallel
{
      
#pragma HLS unroll
      double merlin_tmp_2;
      merlin_tmp_2 = __merlinmd_kernel_force_x_ch . read();
      force_x[((long )0) + merlin_it_0 * ((long )1) + merlin_it_1] = merlin_tmp_2;
    }
  }
}

void __merlinmd_kernel_force_y_streaming(double force_y[256])
{
  
#pragma HLS INLINE off
  long merlin_it_0;
  merlinL7:
  for (merlin_it_0 = 0L; merlin_it_0 < 256L; ++merlin_it_0) 
// Original: #pragma ACCEL pipeline
{
    
#pragma HLS dependence variable=force_y array inter false
    
#pragma HLS pipeline
    long merlin_it_1;
    merlinL6:
    for (merlin_it_1 = 0L; merlin_it_1 < ((long )1); ++merlin_it_1) 
// Original: #pragma ACCEL parallel
{
      
#pragma HLS unroll
      double merlin_tmp_2;
      merlin_tmp_2 = __merlinmd_kernel_force_y_ch . read();
      force_y[((long )0) + merlin_it_0 * ((long )1) + merlin_it_1] = merlin_tmp_2;
    }
  }
}

void __merlinmd_kernel_force_z_streaming(double force_z[256])
{
  
#pragma HLS INLINE off
  long merlin_it_0;
  merlinL9:
  for (merlin_it_0 = 0L; merlin_it_0 < 256L; ++merlin_it_0) 
// Original: #pragma ACCEL pipeline
{
    
#pragma HLS dependence variable=force_z array inter false
    
#pragma HLS pipeline
    long merlin_it_1;
    merlinL8:
    for (merlin_it_1 = 0L; merlin_it_1 < ((long )1); ++merlin_it_1) 
// Original: #pragma ACCEL parallel
{
      
#pragma HLS unroll
      double merlin_tmp_2;
      merlin_tmp_2 = __merlinmd_kernel_force_z_ch . read();
      force_z[((long )0) + merlin_it_0 * ((long )1) + merlin_it_1] = merlin_tmp_2;
    }
  }
}

void __merlinmd_kernel_NL_streaming(int NL[4096])
{
  
#pragma HLS INLINE off
  long merlin_it_0;
  merlinL1:
  for (merlin_it_0 = 0L; merlin_it_0 < 4096L; ++merlin_it_0) 
// Original: #pragma ACCEL pipeline
{
    
#pragma HLS pipeline
    long merlin_it_1;
    merlinL0:
    for (merlin_it_1 = 0L; merlin_it_1 < ((long )1); ++merlin_it_1) 
// Original: #pragma ACCEL parallel
{
      
#pragma HLS unroll
      int merlin_tmp_2;
      merlin_tmp_2 = NL[((long )0) + merlin_it_0 * ((long )1) + merlin_it_1];
      __merlinmd_kernel_NL_ch . write(merlin_tmp_2);
    }
  }
}

void mars_kernel_0_7_node_0_stage_0(int i,int exec,double *fx,double *fy,double *fz,double *i_x,double *i_y,double *i_z,double position_x_8_0_buf[256],double position_y_8_0_buf[256],double position_z_8_0_buf[256])
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
     *i_x = position_x_8_0_buf[i];
     *i_y = position_y_8_0_buf[i];
     *i_z = position_z_8_0_buf[i];
     *fx = ((double )0);
     *fy = ((double )0);
     *fz = ((double )0);
  }
}

void mars_kernel_0_7_node_1_stage_1(int i,int exec,double *delx,double *dely,double *delz,double *force,double *fx,double *fy,double *fz,double i_x,double i_y,double i_z,double *j_x,double *j_y,double *j_z,int *jidx,class ap_uint< 512 > position_x[32],class ap_uint< 512 > position_y[32],class ap_uint< 512 > position_z[32],double *potential,double *r2inv,double *r6inv)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    int j;
    merlinL3:
    for (j = 0; j < 16; j++) 
// Original: #pragma ACCEL PIPELINE AUTO
{
      
#pragma HLS pipeline
      double tmp_1;
      double tmp_0;
      double tmp;
// Get neighbor
       *jidx = __merlinmd_kernel_NL_ch . read();
// Look up x,y,z positions
      tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(::size_t )( *jidx * 8));
       *j_x = tmp;
      tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(::size_t )( *jidx * 8));
       *j_y = tmp_0;
      tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(::size_t )( *jidx * 8));
       *j_z = tmp_1;
// Calc distance
       *delx = i_x -  *j_x;
       *dely = i_y -  *j_y;
       *delz = i_z -  *j_z;
       *r2inv = 1.0 / ( *delx *  *delx +  *dely *  *dely +  *delz *  *delz);
// Assume no cutoff and aways account for all nodes in area
       *r6inv =  *r2inv *  *r2inv *  *r2inv;
       *potential =  *r6inv * (1.5 *  *r6inv - 2.0);
// Sum changes in force
       *force =  *r2inv *  *potential;
       *fx +=  *delx *  *force;
       *fy +=  *dely *  *force;
       *fz +=  *delz *  *force;
    }
// Original label: loop_j:for(j = 0;j < 16;j++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;jidx = __merlinmd_kernel_NL_ch . read();tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;}
  }
}

void mars_kernel_0_7_node_2_stage_2(int i,int exec,double fx)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    __merlinmd_kernel_force_x_ch . write(fx);
//Update forces after all neighbors accounted for.
  }
}

void mars_kernel_0_7_node_3_stage_2(int i,int exec,double fy)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    __merlinmd_kernel_force_y_ch . write(fy);
  }
}

void mars_kernel_0_7_node_4_stage_2(int i,int exec,double fz)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    __merlinmd_kernel_force_z_ch . write(fz);
//printf("dF=%lf,%lf,%lf\n", fx, fy, fz);
  }
}

void mars_kernel_0_7(int mars_i,int mars_init,int mars_cond,double *mars_delx_1,double *mars_dely_1,double *mars_delz_1,double *mars_force_1,double *mars_fx_0,double *mars_fx_1,double *mars_fx_2,double *mars_fy_0,double *mars_fy_1,double *mars_fy_2,double *mars_fz_0,double *mars_fz_1,double *mars_fz_2,double *mars_i_x_0,double *mars_i_x_1,double *mars_i_y_0,double *mars_i_y_1,double *mars_i_z_0,double *mars_i_z_1,double *mars_j_x_1,double *mars_j_y_1,double *mars_j_z_1,int *mars_jidx_1,class ap_uint< 512 > position_x[32],double mars_position_x_8_0_buf_0[256],class ap_uint< 512 > position_y[32],double mars_position_y_8_0_buf_0[256],class ap_uint< 512 > position_z[32],double mars_position_z_8_0_buf_0[256],double *mars_potential_1,double *mars_r2inv_1,double *mars_r6inv_1)
{
  
#pragma HLS INLINE OFF
  mars_kernel_0_7_node_0_stage_0(mars_i - 0,(int )((mars_i >= mars_init + 0) & (mars_i <= mars_cond + 0)),mars_fx_0,mars_fy_0,mars_fz_0,mars_i_x_0,mars_i_y_0,mars_i_z_0,mars_position_x_8_0_buf_0,mars_position_y_8_0_buf_0,mars_position_z_8_0_buf_0);
  mars_kernel_0_7_node_1_stage_1(mars_i - 1,(int )((mars_i >= mars_init + 1) & (mars_i <= mars_cond + 1)),mars_delx_1,mars_dely_1,mars_delz_1,mars_force_1,mars_fx_1,mars_fy_1,mars_fz_1, *mars_i_x_1, *mars_i_y_1, *mars_i_z_1,mars_j_x_1,mars_j_y_1,mars_j_z_1,mars_jidx_1,position_x,position_y,position_z,mars_potential_1,mars_r2inv_1,mars_r6inv_1);
  mars_kernel_0_7_node_2_stage_2(mars_i - 2,(int )((mars_i >= mars_init + 2) & (mars_i <= mars_cond + 2)), *mars_fx_2);
  mars_kernel_0_7_node_3_stage_2(mars_i - 2,(int )((mars_i >= mars_init + 2) & (mars_i <= mars_cond + 2)), *mars_fy_2);
  mars_kernel_0_7_node_4_stage_2(mars_i - 2,(int )((mars_i >= mars_init + 2) & (mars_i <= mars_cond + 2)), *mars_fz_2);
}

void __merlinmd_kernel_computation(double force_x[256],double force_y[256],double force_z[256],class ap_uint< 512 > position_x[32],class ap_uint< 512 > position_y[32],class ap_uint< 512 > position_z[32],int NL[4096])
{
  
#pragma HLS inline off
  double position_z_8_0_buf[256];
  
#pragma HLS array_partition variable=position_z_8_0_buf cyclic factor=8 dim=1
  double position_y_8_0_buf[256];
  
#pragma HLS array_partition variable=position_y_8_0_buf cyclic factor=8 dim=1
  double position_x_8_0_buf[256];
  
#pragma HLS array_partition variable=position_x_8_0_buf cyclic factor=8 dim=1
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
{
    memcpy_wide_bus_read_double_512(&position_x_8_0_buf[0],(class ap_uint< 512 > *)position_x,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
{
    memcpy_wide_bus_read_double_512(&position_y_8_0_buf[0],(class ap_uint< 512 > *)position_y,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
{
    memcpy_wide_bus_read_double_512(&position_z_8_0_buf[0],(class ap_uint< 512 > *)position_z,(0 * 8),sizeof(double ) * ((unsigned long )256),256L);
  }
// Original label: loop_i:for(i = 0;i < 256;i++) {#pragma ACCEL PIPELINE auto{__PIPE__L0}i_x = position_x_8_0_buf[i];i_y = position_y_8_0_buf[i];i_z = position_z_8_0_buf[i];fx =((double )0);fy =((double )0);fz =((double )0);loop_j:for(j = 0;j < 16;j++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;jidx = __merlinmd_kernel_NL_ch . read();tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;}__merlinmd_kernel_force_x_ch . write(fx);__merlinmd_kernel_force_y_ch . write(fy);__merlinmd_kernel_force_z_ch . write(fz);}
  int mars_count_0_7 = 0;
  double mars_kernel_0_7_fx_0 = fx;
  double mars_kernel_0_7_fx_1 = fx;
  double mars_kernel_0_7_fx_2 = fx;
  double mars_kernel_0_7_fy_0 = fy;
  double mars_kernel_0_7_fy_1 = fy;
  double mars_kernel_0_7_fy_2 = fy;
  double mars_kernel_0_7_fz_0 = fz;
  double mars_kernel_0_7_fz_1 = fz;
  double mars_kernel_0_7_fz_2 = fz;
  double mars_kernel_0_7_i_x_0 = i_x;
  double mars_kernel_0_7_i_x_1 = i_x;
  double mars_kernel_0_7_i_x_2 = i_x;
  double mars_kernel_0_7_i_y_0 = i_y;
  double mars_kernel_0_7_i_y_1 = i_y;
  double mars_kernel_0_7_i_y_2 = i_y;
  double mars_kernel_0_7_i_z_0 = i_z;
  double mars_kernel_0_7_i_z_1 = i_z;
  double mars_kernel_0_7_i_z_2 = i_z;
  merlinL2:
  for (i = 0; i < 256 + 2; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
    if (mars_count_0_7 == 0) 
      mars_kernel_0_7(i,0,255,&delx,&dely,&delz,&force,&mars_kernel_0_7_fx_0,&mars_kernel_0_7_fx_1,&mars_kernel_0_7_fx_2,&mars_kernel_0_7_fy_0,&mars_kernel_0_7_fy_1,&mars_kernel_0_7_fy_2,&mars_kernel_0_7_fz_0,&mars_kernel_0_7_fz_1,&mars_kernel_0_7_fz_2,&mars_kernel_0_7_i_x_0,&mars_kernel_0_7_i_x_1,&mars_kernel_0_7_i_y_0,&mars_kernel_0_7_i_y_1,&mars_kernel_0_7_i_z_0,&mars_kernel_0_7_i_z_1,&j_x,&j_y,&j_z,&jidx,position_x,position_x_8_0_buf,position_y,position_y_8_0_buf,position_z,position_z_8_0_buf,&potential,&r2inv,&r6inv);
     else if (mars_count_0_7 == 1) 
      mars_kernel_0_7(i,0,255,&delx,&dely,&delz,&force,&mars_kernel_0_7_fx_2,&mars_kernel_0_7_fx_0,&mars_kernel_0_7_fx_1,&mars_kernel_0_7_fy_2,&mars_kernel_0_7_fy_0,&mars_kernel_0_7_fy_1,&mars_kernel_0_7_fz_2,&mars_kernel_0_7_fz_0,&mars_kernel_0_7_fz_1,&mars_kernel_0_7_i_x_2,&mars_kernel_0_7_i_x_0,&mars_kernel_0_7_i_y_2,&mars_kernel_0_7_i_y_0,&mars_kernel_0_7_i_z_2,&mars_kernel_0_7_i_z_0,&j_x,&j_y,&j_z,&jidx,position_x,position_x_8_0_buf,position_y,position_y_8_0_buf,position_z,position_z_8_0_buf,&potential,&r2inv,&r6inv);
     else 
      mars_kernel_0_7(i,0,255,&delx,&dely,&delz,&force,&mars_kernel_0_7_fx_1,&mars_kernel_0_7_fx_2,&mars_kernel_0_7_fx_0,&mars_kernel_0_7_fy_1,&mars_kernel_0_7_fy_2,&mars_kernel_0_7_fy_0,&mars_kernel_0_7_fz_1,&mars_kernel_0_7_fz_2,&mars_kernel_0_7_fz_0,&mars_kernel_0_7_i_x_1,&mars_kernel_0_7_i_x_2,&mars_kernel_0_7_i_y_1,&mars_kernel_0_7_i_y_2,&mars_kernel_0_7_i_z_1,&mars_kernel_0_7_i_z_2,&j_x,&j_y,&j_z,&jidx,position_x,position_x_8_0_buf,position_y,position_y_8_0_buf,position_z,position_z_8_0_buf,&potential,&r2inv,&r6inv);
    mars_count_0_7++;
    if (mars_count_0_7 == 3) 
      mars_count_0_7 = 0;
  }
// Original label: loop_j:for(j = 0;j < 16;j++) {#pragma ACCEL PIPELINE AUTOdouble tmp_1;double tmp_0;double tmp;jidx = __merlinmd_kernel_NL_ch . read();tmp = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_x,(jidx * 8));j_x = tmp;tmp_0 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_y,(jidx * 8));j_y = tmp_0;tmp_1 = memcpy_wide_bus_single_read_double_512((class ap_uint< 512 > *)position_z,(jidx * 8));j_z = tmp_1;delx = i_x - j_x;dely = i_y - j_y;delz = i_z - j_z;r2inv = 1.0 /(delx * delx + dely * dely + delz * delz);r6inv = r2inv * r2inv * r2inv;potential = r6inv *(1.5 * r6inv - 2.0);force = r2inv * potential;fx += delx * force;fy += dely * force;fz += delz * force;}
//Update forces after all neighbors accounted for.
//printf("dF=%lf,%lf,%lf\n", fx, fy, fz);
}
// Existing HLS partition: #pragma HLS array_partition variable=position_x_8_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=position_y_8_0_buf cyclic factor = 8 dim=1
// Existing HLS partition: #pragma HLS array_partition variable=position_z_8_0_buf cyclic factor = 8 dim=1
static int md_kernel_dummy_pos;
extern "C" { 

void md_kernel(double force_x[256],double force_y[256],double force_z[256],class ap_uint< 512 > position_x[32],class ap_uint< 512 > position_y[32],class ap_uint< 512 > position_z[32],int NL[4096])
{
  
#pragma HLS INTERFACE m_axi port=NL offset=slave depth=4096 bundle=merlin_gmem_md_kernel_32_0
  
#pragma HLS INTERFACE m_axi port=force_x offset=slave depth=256 bundle=merlin_gmem_md_kernel_64_0
  
#pragma HLS INTERFACE m_axi port=force_y offset=slave depth=256 bundle=merlin_gmem_md_kernel_64_1
  
#pragma HLS INTERFACE m_axi port=force_z offset=slave depth=256 bundle=merlin_gmem_md_kernel_64_2
  
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
  
#pragma HLS DATA_PACK VARIABLE=position_z
  
#pragma HLS DATA_PACK VARIABLE=position_y
  
#pragma HLS DATA_PACK VARIABLE=position_x
  
#pragma ACCEL interface variable=position_z depth=256 max_depth=256
  
#pragma ACCEL interface variable=position_y depth=256 max_depth=256
  
#pragma ACCEL interface variable=position_x depth=256 max_depth=256
  
#pragma ACCEL interface variable=force_x max_depth=256 depth=256 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=64
  
#pragma ACCEL interface variable=force_y max_depth=256 depth=256 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=64
  
#pragma ACCEL interface variable=force_z max_depth=256 depth=256 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=64
  
#pragma ACCEL interface variable=NL max_depth=4096 depth=4096 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=32
  
#pragma HLS dataflow
  
#pragma HLS stream variable=__merlinmd_kernel_NL_ch
  __merlinmd_kernel_NL_streaming(NL);
  __merlinmd_kernel_computation(0,0,0,position_x,position_y,position_z,0);
  
#pragma HLS stream variable=__merlinmd_kernel_force_x_ch
  __merlinmd_kernel_force_x_streaming(force_x);
  
#pragma HLS stream variable=__merlinmd_kernel_force_y_ch
  __merlinmd_kernel_force_y_streaming(force_y);
  
#pragma HLS stream variable=__merlinmd_kernel_force_z_ch
  __merlinmd_kernel_force_z_streaming(force_z);
}
}
