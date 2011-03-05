#include "ihd.h"

static __global__ void _force(R *f, const R fi, const R ki,
                                    const Z n1, const Z n2)
{
  const Z i = blockDim.y * blockIdx.y + threadIdx.y;
  const Z j = blockDim.x * blockIdx.x + threadIdx.x;
  const Z h = i * n2 + j;

  if(i < n1 && j < n2) {
    const R x = (TWO_PI / n1) * i;
    f[h] += fi * ki * cos(ki * x);
  }
}

R *force(R *f, const R fi, const R ki)
{
  if(fi != 0.0 && ki != 0.0)
    _force<<<Gsz, Bsz>>>(f, fi, ki, N1, N2);
  return f;
}
