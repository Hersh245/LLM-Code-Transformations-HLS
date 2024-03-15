#define __constant
#define __kernel
#define __global
#include "memcpy_512_1d.h"
#define SIZE_1 64
#include "memcpy_512_2d.h"
#undef SIZE_1
#include <string.h> 
#include<hls_stream.h>


#include "merlin_type_define.h"






// Original: #pragma ACCEL kernel
static class hls::stream< int  > __merlinstencil_sol_ch;

void __merlinstencil_sol_streaming(int sol[8192])
{
  
#pragma HLS INLINE off
  merlinL7:
  for (long merlin_it_0 = 0L; merlin_it_0 <= 125L; merlin_it_0 += 1L) {
    long merlin_it_1;
    merlinL6:
    for (merlin_it_1 = 0L; merlin_it_1 < 62L; ++merlin_it_1) 
// Original: #pragma ACCEL pipeline
{
      
#pragma HLS dependence variable=sol array inter false
      
#pragma HLS pipeline
      long merlin_it_2;
      merlinL5:
      for (merlin_it_2 = 0L; merlin_it_2 < ((long )1); ++merlin_it_2) 
// Original: #pragma ACCEL parallel
{
        
#pragma HLS unroll
        int merlin_tmp_3;
        merlin_tmp_3 = __merlinstencil_sol_ch . read();
        sol[merlin_it_0 * ((long )64) + merlin_it_1 * ((long )1) + merlin_it_2] = merlin_tmp_3;
      }
    }
  }
}

void mars_kernel_0_5_node_0_stage_0(int c,int exec,int *temp)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
     *temp = ((int )0);
  }
}

void mars_kernel_0_5_node_1_stage_1(int c,int exec,int filter_10_0_buf[3][3],int *mul,int orig_10_0_buf[128][64],int r,int *temp)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    int k2;
    int k1;
    merlinL4:
    for (k1 = 0; k1 < 3; k1++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{
      
#pragma HLS pipeline
// Original label: stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}
      merlinL3:
// Original label: stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}
      for (k2 = 0; k2 < 3; k2++) 
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        
#pragma HLS unroll
         *mul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];
         *temp +=  *mul;
      }
    }
// Original label: stencil_label3:for(k1 = 0;k1 < 3;k1++) {#pragma ACCEL PIPELINE auto{__PIPE__L2}stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}}
  }
}

void mars_kernel_0_5_node_2_stage_2(int c,int exec,int temp)
{
  
#pragma HLS INLINE OFF
  if (exec == 1) {
    __merlinstencil_sol_ch . write(temp);
  }
}

void mars_kernel_0_5(int mars_c,int mars_init,int mars_cond,int mars_filter_10_0_buf_1[3][3],int *mars_mul_1,int mars_orig_10_0_buf_1[128][64],int mars_r_1,int *mars_temp_0,int *mars_temp_1,int *mars_temp_2)
{
  
#pragma HLS INLINE OFF
  mars_kernel_0_5_node_0_stage_0(mars_c - 0,(int )((mars_c >= mars_init + 0) & (mars_c <= mars_cond + 0)),mars_temp_0);
  mars_kernel_0_5_node_1_stage_1(mars_c - 1,(int )((mars_c >= mars_init + 1) & (mars_c <= mars_cond + 1)),mars_filter_10_0_buf_1,mars_mul_1,mars_orig_10_0_buf_1,mars_r_1,mars_temp_1);
  mars_kernel_0_5_node_2_stage_2(mars_c - 2,(int )((mars_c >= mars_init + 2) & (mars_c <= mars_cond + 2)), *mars_temp_2);
}

static void merlin_memcpy_0(int dst[3][3],int dst_idx_0,int dst_idx_1,int src[9],int src_idx_0,unsigned int len,unsigned int max_len)
{
  
#pragma HLS inline off
  
#pragma HLS function_instantiate variable=dst_idx_0,dst_idx_1,src_idx_0
  unsigned long i;
  unsigned long total_offset1 = (0 * 3 + dst_idx_0) * 3 + dst_idx_1;
  unsigned long total_offset2 = 0 * 9 + src_idx_0;
  merlinL2:
  for (i = 0; i < len / 4; ++i) {
    
#pragma HLS PIPELINE II=1
    
#pragma HLS LOOP_TRIPCOUNT max=9
    dst[(total_offset1 + i) / 3][(total_offset1 + i) % 3] = src[total_offset2 + i];
  }
}

void __merlinstencil_computation(class ap_uint< 512 > orig[512],int sol[8192],int filter[9])
{
  
#pragma HLS inline off
  int orig_10_0_buf[128][64];
  
#pragma HLS array_partition variable=orig_10_0_buf cyclic factor=16 dim=2
  int filter_10_0_buf[3][3];
  
#pragma HLS array_partition variable=filter_10_0_buf complete dim=2
  int r;
  int c;
  int k1;
  int k2;
  int temp;
  int mul;
{
    merlin_memcpy_0(filter_10_0_buf,0,0,filter,0,sizeof(int ) * ((unsigned long )9),36UL);
  }
{
    memcpy_wide_bus_read_int_2d_64_512(orig_10_0_buf,0,0,(class ap_uint< 512 > *)orig,(0 * 4),sizeof(int ) * ((unsigned long )8192),8192L);
  }
// Original label: stencil_label1:for(r = 0;r < 128 - 2;r++) {#pragma ACCEL PIPELINE auto{__PIPE__L0}__merlin_streaming_range(sol,r * 64,62);stencil_label2:for(c = 0;c < 64 - 2;c++) {#pragma ACCEL PIPELINE auto{__PIPE__L1}temp =((int )0);stencil_label3:for(k1 = 0;k1 < 3;k1++) {#pragma ACCEL PIPELINE auto{__PIPE__L2}stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}}__merlinstencil_sol_ch . write(temp);}}
  merlinL1:
// Original label: stencil_label1:for(r = 0;r < 128 - 2;r++) {#pragma ACCEL PIPELINE auto{__PIPE__L0}__merlin_streaming_range(sol,r * 64,62);stencil_label2:for(c = 0;c < 64 - 2;c++) {#pragma ACCEL PIPELINE auto{__PIPE__L1}temp =((int )0);stencil_label3:for(k1 = 0;k1 < 3;k1++) {#pragma ACCEL PIPELINE auto{__PIPE__L2}stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}}__merlinstencil_sol_ch . write(temp);}}
  for (r = 0; r < 128 - 2; r++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L0}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L0}
{
// Original label: stencil_label2:for(c = 0;c < 64 - 2;c++) {#pragma ACCEL PIPELINE auto{__PIPE__L1}temp =((int )0);stencil_label3:for(k1 = 0;k1 < 3;k1++) {#pragma ACCEL PIPELINE auto{__PIPE__L2}stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}}__merlinstencil_sol_ch . write(temp);}
    int mars_count_0_5 = 0;
    int mars_kernel_0_5_temp_0 = temp;
    int mars_kernel_0_5_temp_1 = temp;
    int mars_kernel_0_5_temp_2 = temp;
    merlinL0:
    for (c = 0; c < 64 - 2 + 2; c++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{
      if (mars_count_0_5 == 0) 
        mars_kernel_0_5(c,0,61,filter_10_0_buf,&mul,orig_10_0_buf,r,&mars_kernel_0_5_temp_0,&mars_kernel_0_5_temp_1,&mars_kernel_0_5_temp_2);
       else if (mars_count_0_5 == 1) 
        mars_kernel_0_5(c,0,61,filter_10_0_buf,&mul,orig_10_0_buf,r,&mars_kernel_0_5_temp_2,&mars_kernel_0_5_temp_0,&mars_kernel_0_5_temp_1);
       else 
        mars_kernel_0_5(c,0,61,filter_10_0_buf,&mul,orig_10_0_buf,r,&mars_kernel_0_5_temp_1,&mars_kernel_0_5_temp_2,&mars_kernel_0_5_temp_0);
      mars_count_0_5++;
      if (mars_count_0_5 == 3) 
        mars_count_0_5 = 0;
    }
// Original label: stencil_label3:for(k1 = 0;k1 < 3;k1++) {#pragma ACCEL PIPELINE auto{__PIPE__L2}stencil_label4:for(k2 = 0;k2 < 3;k2++) {#pragma ACCEL PARALLEL COMPLETEmul = filter_10_0_buf[k1][k2] * orig_10_0_buf[k1 + r][c + k2];temp += mul;}}
  }
}
// Existing HLS partition: #pragma HLS array_partition variable=orig_10_0_buf cyclic factor = 16 dim=2
static int stencil_dummy_pos;
extern "C" { 

void stencil(class ap_uint< 512 > orig[512],int sol[8192],int filter[9])
{
  
#pragma HLS INTERFACE m_axi port=filter offset=slave depth=9 bundle=merlin_gmem_stencil_32_0
  
#pragma HLS INTERFACE m_axi port=orig offset=slave depth=512 bundle=merlin_gmem_stencil_512_0
  
#pragma HLS INTERFACE m_axi port=sol offset=slave depth=8192 bundle=merlin_gmem_stencil_32_0
  
#pragma HLS INTERFACE s_axilite port=filter bundle=control
  
#pragma HLS INTERFACE s_axilite port=orig bundle=control
  
#pragma HLS INTERFACE s_axilite port=sol bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=orig
  
#pragma ACCEL interface variable=filter depth=9 max_depth=9
  
#pragma ACCEL interface variable=orig depth=8192 max_depth=8192
  
#pragma ACCEL interface variable=sol max_depth=8192 depth=8192 BURST_OFF EXPLICIT_BUNDLE BUS_BITWIDTH=32
  
#pragma HLS dataflow
  __merlinstencil_computation(orig,0,filter);
  
#pragma HLS stream variable=__merlinstencil_sol_ch
  __merlinstencil_sol_streaming(sol);
}
}
