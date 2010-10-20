#include <stdio.h>
#include "ihd.h"

#define TIDE_Y 32

static __global__ void _dx(C *d, C *f, Z n1, Z h2)
{
  Z i = blockDim.x * blockIdx.x + threadIdx.x;
  Z j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i < n1 && j < h2) {
    Z h = i * h2 + j;
    R k = i < n1 / 2 ? i : i - n1;
    C g = f[h]; /* In case the derivative is in-place */
    d[h].r = - k * g.i;
    d[h].i =   k * g.r;
  }
}

static __global__ void _dy(C *d, C *f, Z n1, Z h2)
{
  Z i = blockDim.x * blockIdx.x + threadIdx.x;
  Z j = blockDim.y * blockIdx.y + threadIdx.y;
  if(i < n1 && j < h2) {
    Z h = i * h2 + j;
    R k = j;
    C g = f[h]; /* In case the derivative is in-place */
    d[h].r = - k * g.i;
    d[h].i =   k * g.r;
  }
}

C *der(C *d, C *f, Z i)
{
  uint3 bsz = make_uint3(M / TIDE_Y, TIDE_Y, 1);
  uint3 gsz = make_uint3((N1 - 1) / bsz.x + 1,
                         (N2 / 2) / bsz.y + 1, 1);
  switch(i) {
  case 1 : _dx<<<gsz, bsz>>>(d, f, N1, H2); break;
  case 2 : _dy<<<gsz, bsz>>>(d, f, N1, H2); break;
  default:
    fprintf(stderr, "ERROR: der: direction cannot be %d\n", i);
    exit(-1);
  }
  return d;
}
