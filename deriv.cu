#include <stdio.h>
#include "ihd.h"

static __global__ void _dx(C *d, const C *f, const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const R k = i < n1 / 2 ? i : i - n1;
    const C g = f[h]; /* In case the derivative is in-place */

    d[h].r = - k * g.i;
    d[h].i =   k * g.r;
  }
}

static __global__ void _dy(C *d, const C *f, const Z n1, const Z h2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * h2 + j;

  if(i < n1 && j < h2) {
    const R k = j;
    const C g = f[h]; /* In case the derivative is in-place */

    d[h].r = - k * g.i;
    d[h].i =   k * g.r;
  }
}

C *deriv(C *d, C *f, Z i)
{
  switch(i) {
  case 1 : _dx<<<Hsz, Bsz>>>(d, (const C *)f, N1, H2); break;
  case 2 : _dy<<<Hsz, Bsz>>>(d, (const C *)f, N1, H2); break;
  default:
    fprintf(stderr, "ERROR: deriv: direction cannot be %d\n", i);
    exit(-1);
  }
  return d;
}
