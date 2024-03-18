#pragma ACCEL kernel

void md_kernel(double force_x[256],double force_y[256],double force_z[256],double position_x[256],double position_y[256],double position_z[256],int NL[4096])
{
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
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=16{__TILE__L0} // Tile the outer loop by a factor of 16
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  loop_j:
  for (j = 0; j < 16; j++) {
    loop_i:
    for (i = 0; i < 256; i++) {
      i_x = position_x[i];
      i_y = position_y[i];
      i_z = position_z[i];
      fx = ((double )0);
      fy = ((double )0);
      fz = ((double )0);
      jidx = NL[i * 16 + j];
      j_x = position_x[jidx];
      j_y = position_y[jidx];
      j_z = position_z[jidx];
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
      force_x[i] += fx; // Accumulate forces directly into the output arrays
      force_y[i] += fy;
      force_z[i] += fz;
    }
  }
}
// Rationale for transformations:
// 1. **Loop Tiling**: By tiling the outer loop by a factor of 16, we can improve data locality and potentially increase parallelism by processing multiple iterations of the inner loop at once. This can lead to better utilization of resources in the FPGA.
// 2. **Loop Permutation**: The order of the loops has been switched to loop over `j` first and then `i`. This can help in better exploiting parallelism and pipelining opportunities in the code.
// 3. **Loop Fusion**: The forces are accumulated directly into the output arrays within the inner loop, eliminating the need for separate assignments at the end of each iteration. This can reduce memory accesses and improve performance.