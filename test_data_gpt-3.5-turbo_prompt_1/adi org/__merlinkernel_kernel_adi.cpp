#define __constant
#define __kernel
#define __global
#include "memcpy_256_1d.h"
#define SIZE_1 60
#include "memcpy_256_2d.h"
#undef SIZE_1
#include <string.h> 

#include "merlin_type_define.h"





// Original: #pragma ACCEL kernel
static int kernel_adi_dummy_pos;
extern "C" { 

void kernel_adi(class ap_uint< 256 > u[900],class ap_uint< 256 > v[900],class ap_uint< 256 > p[900],class ap_uint< 256 > q[900])
{
  
#pragma HLS INTERFACE m_axi port=p offset=slave depth=900 bundle=merlin_gmem_kernel_adi_256_p
  
#pragma HLS INTERFACE m_axi port=q offset=slave depth=900 bundle=merlin_gmem_kernel_adi_256_q
  
#pragma HLS INTERFACE m_axi port=u offset=slave depth=900 bundle=merlin_gmem_kernel_adi_256_u
  
#pragma HLS INTERFACE m_axi port=v offset=slave depth=900 bundle=merlin_gmem_kernel_adi_256_v
  
#pragma HLS INTERFACE s_axilite port=p bundle=control
  
#pragma HLS INTERFACE s_axilite port=q bundle=control
  
#pragma HLS INTERFACE s_axilite port=u bundle=control
  
#pragma HLS INTERFACE s_axilite port=v bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
  
#pragma HLS DATA_PACK VARIABLE=q
  
#pragma HLS DATA_PACK VARIABLE=p
  
#pragma HLS DATA_PACK VARIABLE=v
  
#pragma HLS DATA_PACK VARIABLE=u
  
#pragma ACCEL interface variable=q depth=60,60 max_depth=60,60
  
#pragma ACCEL interface variable=p depth=60,60 max_depth=60,60
  
#pragma ACCEL interface variable=v depth=60,60 max_depth=60,60
  
#pragma ACCEL interface variable=u depth=60,60 max_depth=60,60
  double v_buf[60][60];
  
#pragma HLS array_partition variable=v_buf cyclic factor=4 dim=2
  
#pragma HLS array_partition variable=v_buf complete dim=1
  double u_buf[58][60];
  
#pragma HLS array_partition variable=u_buf complete dim=2
  double q_buf[58][60];
  
#pragma HLS array_partition variable=q_buf complete dim=2
  double p_buf[58][60];
  
#pragma HLS array_partition variable=p_buf complete dim=2
  int t;
  int i;
  int j;
  double DX;
  double DY;
  double DT;
  double B1;
  double B2;
  double mul1;
  double mul2;
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
//#pragma scop
  DX = 1.0 / ((double )60);
  DY = 1.0 / ((double )60);
  DT = 1.0 / ((double )40);
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;
// Existing HLS partition: #pragma HLS array_partition variable=p_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_read_double_2d_60_256(p_buf,0,0,(class ap_uint< 256 > *)p,(60L * 8),sizeof(double ) * ((unsigned long )3479L),3479L);
// Existing HLS partition: #pragma HLS array_partition variable=q_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_read_double_2d_60_256(q_buf,0,0,(class ap_uint< 256 > *)q,(60L * 8),sizeof(double ) * ((unsigned long )3479L),3479L);
// Existing HLS partition: #pragma HLS array_partition variable=u_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_read_double_2d_60_256(u_buf,0,0,(class ap_uint< 256 > *)u,(60L * 8),sizeof(double ) * ((unsigned long )3480L),3480L);
// Existing HLS partition: #pragma HLS array_partition variable=v_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_read_double_2d_60_256(v_buf,0,0,(class ap_uint< 256 > *)v,(1 * 8),sizeof(double ) * ((unsigned long )3598),3598L);
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
    for (i = 1; i < 59; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L1}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L1}
{{
        v_buf[0][- 1 + i] = 1.0;
        p_buf[- 1 + i][0] = 0.0;
        q_buf[- 1 + i][0] = v_buf[0][- 1 + i];
      }
      merlinL4:
      for (j = 1; j < 59; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS pipeline
        p_buf[- 1 + i][j] = -c / (a * p_buf[- 1 + i][- 1 + j] + b);
        q_buf[- 1 + i][j] = (-d * u_buf[- 1 + j][- 1 + i] + (1.0 + 2.0 * d) * u_buf[- 1 + j][i] - f * u_buf[- 1 + j][1 + i] - a * q_buf[- 1 + i][- 1 + j]) / (a * p_buf[- 1 + i][- 1 + j] + b);
      }
      v_buf[59][- 1 + i] = 1.0;
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      merlinL3:
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      for (j = 0; j <= 57; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        
#pragma HLS unroll
        int _in_j_0 = 58 + - 1 * j;
        v_buf[58 + -j][- 1 + i] = p_buf[- 1 + i][58 + -j] * v_buf[59 + -j][- 1 + i] + q_buf[- 1 + i][58 + -j];
      }
      j = 1 + - 1;
    }
//Row Sweep
    merlinL2:
//Row Sweep
    for (i = 1; i < 59; i++) 
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
// Original: #pragma ACCEL TILE FACTOR=auto{__TILE__L2}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
// Original: #pragma ACCEL PIPELINE auto{__PIPE__L2}
{{
        u_buf[- 1 + i][0] = 1.0;
        p_buf[- 1 + i][0] = 0.0;
        q_buf[- 1 + i][0] = u_buf[- 1 + i][0];
      }
      merlinL1:
      for (j = 1; j < 59; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
// Original: #pragma ACCEL PIPELINE AUTO
{
        
#pragma HLS pipeline
        p_buf[- 1 + i][j] = -f / (d * p_buf[- 1 + i][- 1 + j] + e);
        q_buf[- 1 + i][j] = (-a * v_buf[- 1 + i][- 1 + j] + (1.0 + 2.0 * a) * v_buf[i][- 1 + j] - c * v_buf[1 + i][- 1 + j] - d * q_buf[- 1 + i][- 1 + j]) / (d * p_buf[- 1 + i][- 1 + j] + e);
      }
      u_buf[- 1 + i][59] = 1.0;
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      merlinL0:
/* Standardize from: for(j = 60 - 2;j >= 1;j--) {...} */
      for (j = 0; j <= 57; j++) 
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
// Original: #pragma ACCEL PARALLEL COMPLETE
{
        
#pragma HLS unroll
        int _in_j = 58 + - 1 * j;
        u_buf[- 1 + i][58 + -j] = p_buf[- 1 + i][58 + -j] * u_buf[- 1 + i][59 + -j] + q_buf[- 1 + i][58 + -j];
      }
      j = 1 + - 1;
    }
  }
//#pragma endscop
//Column Sweep
// Existing HLS partition: #pragma HLS array_partition variable=v_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_write_double_2d_60_256((class ap_uint< 256 > *)v,v_buf,0,0,(8 * 1),sizeof(double ) * ((unsigned long )3598),3598L);
// Existing HLS partition: #pragma HLS array_partition variable=u_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_write_double_2d_60_256((class ap_uint< 256 > *)u,u_buf,0,0,(8 * 60L),sizeof(double ) * ((unsigned long )3480L),3480L);
// Existing HLS partition: #pragma HLS array_partition variable=q_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_write_double_2d_60_256((class ap_uint< 256 > *)q,q_buf,0,0,(8 * 60L),sizeof(double ) * ((unsigned long )3479L),3479L);
// Existing HLS partition: #pragma HLS array_partition variable=p_buf cyclic factor = 4 dim=2
  memcpy_wide_bus_write_double_2d_60_256((class ap_uint< 256 > *)p,p_buf,0,0,(8 * 60L),sizeof(double ) * ((unsigned long )3479L),3479L);
}
}
